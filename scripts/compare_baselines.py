"""Compare unweighted and class-weighted multi-view knee MRI CNNs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from knee_ai.data.knee_dataset import KneeMRIDataset
from knee_ai.models.multiview_cnn import MultiViewKneeCNN
from knee_ai.training.train import (
    LABEL_NAMES,
    calculate_metrics,
    move_batch_to_device,
    set_random_seeds,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXAMINATION_MANIFEST = (
    PROJECT_ROOT / "data" / "gold" / "manifests" / "development_manifest.csv"
)
SPLIT_MANIFEST = PROJECT_ROOT / "data" / "gold" / "splits" / "examination_splits.csv"
MRI_ROOT = PROJECT_ROOT / "data" / "bronze" / "mri_sample"

CHECKPOINTS = {
    "unweighted": (
        PROJECT_ROOT / "artifacts" / "checkpoints" / "baseline_unweighted.pt"
    ),
    "class_weighted": (PROJECT_ROOT / "artifacts" / "checkpoints" / "baseline_best.pt"),
}

REPORT_DIR = PROJECT_ROOT / "reports"
NUM_SLICES = 8
BATCH_SIZE = 2
THRESHOLD = 0.5
SEED = 42


def evaluate_checkpoint(
    checkpoint_path: Path,
    loader: DataLoader,
    dataset_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate one checkpoint on the shared validation dataset."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = MultiViewKneeCNN(num_slices=NUM_SLICES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            views, labels = move_batch_to_device(batch, device)
            logits = model(**views)
            probabilities = torch.sigmoid(logits)

            all_targets.append(labels.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())

    targets = np.concatenate(all_targets, axis=0)
    probabilities = np.concatenate(all_probabilities, axis=0)

    true_labels = targets.astype(np.int32)
    predictions = (probabilities >= THRESHOLD).astype(np.int32)

    auc_results = calculate_metrics(targets, probabilities)

    results: dict[str, Any] = {
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_loss": float(checkpoint["validation_loss"]),
        "validation_examinations": dataset_size,
        "classification_threshold": THRESHOLD,
        "metrics": {},
    }

    for index, label_name in enumerate(LABEL_NAMES):
        y_true = true_labels[:, index]
        y_pred = predictions[:, index]

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        results["metrics"][label_name] = {
            "roc_auc": auc_results[label_name]["roc_auc"],
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "actual_positive": int(y_true.sum()),
            "actual_negative": int(len(y_true) - y_true.sum()),
            "predicted_positive": int(y_pred.sum()),
        }

    return results


def main() -> None:
    """Compare both checkpoints and save a separate report."""
    set_random_seeds(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classification threshold: {THRESHOLD}")

    dataset = KneeMRIDataset(
        examination_manifest=EXAMINATION_MANIFEST,
        split_manifest=SPLIT_MANIFEST,
        mri_root=MRI_ROOT,
        split="validation",
        num_slices=NUM_SLICES,
    )

    if len(dataset) == 0:
        raise ValueError("No validation examinations are available.")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    comparison: dict[str, Any] = {
        "experiment": "unweighted_vs_class_weighted_baseline",
        "validation_examinations": len(dataset),
        "num_slices_per_view": NUM_SLICES,
        "classification_threshold": THRESHOLD,
        "models": {},
        "weighted_minus_unweighted": {},
    }

    for model_name, checkpoint_path in CHECKPOINTS.items():
        print(f"\nEvaluating {model_name} model...")
        comparison["models"][model_name] = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            loader=loader,
            dataset_size=len(dataset),
            device=device,
        )

    unweighted_metrics = comparison["models"]["unweighted"]["metrics"]
    weighted_metrics = comparison["models"]["class_weighted"]["metrics"]

    for label_name in LABEL_NAMES:
        comparison["weighted_minus_unweighted"][label_name] = {}

        for metric_name in ("roc_auc", "precision", "recall", "f1_score"):
            unweighted_value = unweighted_metrics[label_name][metric_name]
            weighted_value = weighted_metrics[label_name][metric_name]

            # AUC may be unavailable if a target has only one class.
            if unweighted_value is None or weighted_value is None:
                difference = None
            else:
                difference = float(weighted_value - unweighted_value)

            comparison["weighted_minus_unweighted"][label_name][metric_name] = (
                difference
            )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "baseline_comparison.json"

    report_path.write_text(
        json.dumps(comparison, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 85)
    print("BASELINE MODEL COMPARISON")
    print("=" * 85)

    for label_name in LABEL_NAMES:
        print(f"\nTarget: {label_name}")

        for model_name in ("unweighted", "class_weighted"):
            model_results = comparison["models"][model_name]
            metrics = model_results["metrics"][label_name]

            print(f"\n  Model: {model_name}")
            print(f"    Checkpoint epoch: {model_results['checkpoint_epoch']}")
            print(f"    ROC-AUC:  {metrics['roc_auc']}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-score:  {metrics['f1_score']:.4f}")
            print(
                "    Confusion matrix counts: "
                f"TN={metrics['true_negative']}, "
                f"FP={metrics['false_positive']}, "
                f"FN={metrics['false_negative']}, "
                f"TP={metrics['true_positive']}"
            )

        differences = comparison["weighted_minus_unweighted"][label_name]
        print("\n  Change (weighted - unweighted):")

        for metric_name, difference in differences.items():
            if difference is None:
                print(f"    {metric_name}: unavailable")
            else:
                print(f"    {metric_name}: {difference:+.4f}")

    print(f"\nComparison report saved to: {report_path}")
    print(
        "\nResearch baseline only. These results do not establish clinical readiness."
    )


if __name__ == "__main__":
    main()
