"""Train an improved baseline CNN with adaptive learning-rate scheduling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from knee_ai.data.knee_dataset import KneeMRIDataset
from knee_ai.models.multiview_cnn import MultiViewKneeCNN
from knee_ai.training.train import run_epoch, save_checkpoint, set_random_seeds

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXAMINATION_MANIFEST = (
    PROJECT_ROOT / "data" / "gold" / "manifests" / "train_examination_manifest.csv"
)
SPLIT_MANIFEST = PROJECT_ROOT / "data" / "gold" / "splits" / "examination_splits.csv"
MRI_ROOT = PROJECT_ROOT / "data" / "bronze" / "mri_sample"

CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "checkpoints" / "baseline_v2_best.pt"
HISTORY_PATH = PROJECT_ROOT / "reports" / "baseline_v2_training_history.json"


def calculate_positive_weights(dataset: KneeMRIDataset) -> torch.Tensor:
    """Calculate negative-to-positive label ratios from training examinations."""
    labels = np.stack(
        [dataset[index]["labels"].numpy() for index in range(len(dataset))]
    )

    positives = labels.sum(axis=0)
    negatives = len(labels) - positives

    if np.any(positives == 0):
        raise ValueError("At least one training label has no positive examples.")

    weights = negatives / positives

    print("\nTraining label distribution:")
    for name, positive, negative, weight in zip(
        ("abnormality", "acl", "meniscus"),
        positives,
        negatives,
        weights,
        strict=True,
    ):
        print(
            f"  {name}: positives={int(positive)}, "
            f"negatives={int(negative)}, pos_weight={weight:.4f}"
        )

    return torch.tensor(weights, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    """Parse experiment settings."""
    parser = argparse.ArgumentParser(
        description="Train Baseline V2 with AdamW and early stopping."
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-slices", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    """Run training and save the best model and experiment history."""
    args = parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")
    if args.lr_patience < 1:
        raise ValueError("--lr-patience must be at least 1.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative.")

    set_random_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("KNEE MRI CLASSIFICATION — BASELINE V2")
    print("=" * 65)
    print(f"Device: {device}")
    print(f"Maximum epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Early-stopping patience: {args.patience}")
    print(f"LR scheduler patience: {args.lr_patience}")
    print(f"Random seed: {args.seed}")

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

    print(f"\nTraining examinations: {len(train_dataset)}")
    print(f"Validation examinations: {len(validation_dataset)}")

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

    positive_weights = calculate_positive_weights(train_dataset).to(device)

    model = MultiViewKneeCNN(num_slices=args.num_slices).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience,
        min_lr=1e-6,
    )

    history: list[dict[str, object]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

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

        validation_loss = float(validation_result["loss"])
        scheduler.step(validation_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        record: dict[str, object] = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": float(train_result["loss"]),
            "validation_loss": validation_loss,
            "train_auc": train_result["metrics"],
            "validation_auc": validation_result["metrics"],
        }
        history.append(record)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Learning rate: {current_lr:.7f}")
        print(f"  Training loss: {train_result['loss']:.4f}")
        print(f"  Validation loss: {validation_loss:.4f}")

        for label_name, metrics in validation_result["metrics"].items():
            auc = metrics["roc_auc"]
            auc_text = "N/A" if auc is None else f"{auc:.4f}"
            print(f"  Validation AUC — {label_name}: {auc_text}")

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                path=CHECKPOINT_PATH,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
                seed=args.seed,
            )
            print(f"  Best checkpoint saved: {CHECKPOINT_PATH.name}")
        else:
            epochs_without_improvement += 1
            print(
                "  No validation-loss improvement for "
                f"{epochs_without_improvement} epoch(s)."
            )

        if epochs_without_improvement >= args.patience:
            print("\nEarly stopping: validation loss did not improve.")
            break

        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY_PATH.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "experiment": "baseline_v2",
                    "configuration": vars(args),
                    "device": str(device),
                    "training_examinations": len(train_dataset),
                    "validation_examinations": len(validation_dataset),
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    "epochs_completed": len(history),
                    "history": history,
                },
                file,
                indent=2,
            )

    # Save the final history even if the run stops after the first epoch.
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "experiment": "baseline_v2",
                "configuration": vars(args),
                "device": str(device),
                "training_examinations": len(train_dataset),
                "validation_examinations": len(validation_dataset),
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "epochs_completed": len(history),
                "history": history,
            },
            file,
            indent=2,
        )

    print("\n" + "=" * 65)
    print("BASELINE V2 TRAINING COMPLETE")
    print("=" * 65)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_validation_loss:.4f}")
    print(f"Checkpoint: {CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"History: {HISTORY_PATH.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
