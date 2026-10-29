"""Train the aggregated multi-view knee MRI CNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from knee_ai.data.knee_dataset import KneeMRIDataset
from knee_ai.models.aggregated_multiview_cnn import (
    AggregatedMultiViewKneeCNN,
)
from knee_ai.training.train import (
    run_epoch,
    save_checkpoint,
    set_random_seeds,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXAMINATION_MANIFEST = (
    PROJECT_ROOT / "data" / "gold" / "manifests" / "train_examination_manifest.csv"
)
SPLIT_MANIFEST = PROJECT_ROOT / "data" / "gold" / "splits" / "examination_splits.csv"
MRI_ROOT = PROJECT_ROOT / "data" / "bronze" / "mri_sample"

CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
REPORT_DIR = PROJECT_ROOT / "reports"


def parse_args() -> argparse.Namespace:
    """Parse and validate training settings."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-slices", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

    if args.num_slices < 1:
        parser.error("--num-slices must be at least 1.")

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")

    return args


def calculate_positive_weights(dataset: KneeMRIDataset) -> torch.Tensor:
    """Calculate label weights from the training dataset only."""
    labels = torch.stack(
        [dataset[index]["labels"] for index in range(len(dataset))]
    ).float()

    positive_counts = labels.sum(dim=0)
    negative_counts = len(dataset) - positive_counts

    if torch.any(positive_counts == 0):
        raise ValueError(
            "Cannot calculate positive weights because at least one "
            "label has no positive training examples."
        )

    weights = negative_counts / positive_counts

    print("\nTraining label counts and positive weights:")

    for index, name in enumerate(("abnormality", "acl", "meniscus")):
        print(
            f"  {name}: "
            f"positives={int(positive_counts[index])}, "
            f"negatives={int(negative_counts[index])}, "
            f"positive_weight={weights[index].item():.4f}"
        )

    return weights


def main() -> None:
    """Train the aggregated model and save its best checkpoint."""
    args = parse_args()
    set_random_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("AGGREGATED MULTI-VIEW CNN TRAINING")
    print("=" * 65)
    print(f"Device: {device}")
    print("Experiment scope: development sample; model comparison.")
    print(f"Random seed: {args.seed}")
    print(f"Learning rate: {args.learning_rate}")

    train_dataset = KneeMRIDataset(
        examination_manifest=EXAMINATION_MANIFEST,
        split_manifest=SPLIT_MANIFEST,
        mri_root=MRI_ROOT,
        split="train",
        num_slices=args.num_slices,
    )

    validation_dataset = KneeMRIDataset(
        examination_manifest=EXAMINATION_MANIFEST,
        split_manifest=SPLIT_MANIFEST,
        mri_root=MRI_ROOT,
        split="validation",
        num_slices=args.num_slices,
    )

    if len(train_dataset) == 0:
        raise ValueError("No training examinations are available.")

    if len(validation_dataset) == 0:
        raise ValueError("No validation examinations are available.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = AggregatedMultiViewKneeCNN(num_slices=args.num_slices).to(device)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"\nModel parameters: {parameter_count:,}")
    print(f"Training examinations: {len(train_dataset)}")
    print(f"Validation examinations: {len(validation_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of slices per view: {args.num_slices}")
    print(f"Epochs: {args.epochs}")

    positive_weights = calculate_positive_weights(train_dataset).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    history: list[dict[str, object]] = []
    best_validation_loss = float("inf")

    checkpoint_path = CHECKPOINT_DIR / "aggregated_full_best.pt"
    history_path = REPORT_DIR / "aggregated_full_training_history.json"

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        validation_result = run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        train_loss = float(train_result["loss"])
        validation_loss = float(validation_result["loss"])

        record: dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_metrics": validation_result["metrics"],
        }

        history.append(record)

        print(
            f"\nEpoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"validation_loss={validation_loss:.4f}"
        )

        print(
            "Validation ROC-AUC:",
            json.dumps(
                validation_result["metrics"],
                sort_keys=True,
            ),
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
                seed=args.seed,
            )

            print("New best aggregated-model checkpoint saved.")

    history_path.write_text(
        json.dumps(history, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("AGGREGATED MODEL TRAINING COMPLETED")
    print("=" * 65)
    print(f"Best validation loss: {best_validation_loss:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training history: {history_path}")
    print(
        "Reminder: these development-sample results are experimental, "
        "not evidence of clinical performance."
    )


if __name__ == "__main__":
    main()
