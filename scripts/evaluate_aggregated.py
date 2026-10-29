"""Evaluate the aggregated multi-view knee MRI CNN checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from knee_ai.data.knee_dataset import KneeMRIDataset
from knee_ai.models.aggregated_multiview_cnn import AggregatedMultiViewKneeCNN
from knee_ai.training.train import (
    LABEL_NAMES,
    calculate_metrics,
    move_batch_to_device,
    set_random_seeds,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXAMINATION_MANIFEST = (
    PROJECT_ROOT / "data" / "gold" / "manifests" / "train_examination_manifest.csv"
)
SPLIT_MANIFEST = PROJECT_ROOT / "data" / "gold" / "splits" / "examination_splits.csv"
MRI_ROOT = PROJECT_ROOT / "data" / "bronze" / "mri_sample"

CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "checkpoints" / "aggregated_full_best.pt"
REPORT_PATH = PROJECT_ROOT / "reports" / "aggregated_full_validation_evaluation.json"
NUM_SLICES = 8
BATCH_SIZE = 2
THRESHOLD = 0.5
SEED = 42


def main() -> None:
    """Evaluate the aggregated model on the validation split."""
    set_random_seeds(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("AGGREGATED MULTI-VIEW CNN EVALUATION")
    print("=" * 65)
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Classification threshold: {THRESHOLD}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            "Run scripts/train_aggregated.py first."
        )

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

    model = AggregatedMultiViewKneeCNN().to(device)

    checkpoint: dict[str, Any] = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

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

    predictions = (probabilities >= THRESHOLD).astype(np.int32)
    true_labels = targets.astype(np.int32)

    auc_results = calculate_metrics(targets, probabilities)

    results: dict[str, Any] = {
        "experiment": "aggregated_multiview_cnn",
        "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_loss": float(checkpoint["validation_loss"]),
        "validation_examinations": len(dataset),
        "num_slices_per_view": NUM_SLICES,
        "classification_threshold": THRESHOLD,
        "metrics": {},
    }

    for index, label_name in enumerate(LABEL_NAMES):
        y_true = true_labels[:, index]
        y_pred = predictions[:, index]

        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()

        results["metrics"][label_name] = {
            "roc_auc": auc_results[label_name]["roc_auc"],
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
            "actual_positive": int(y_true.sum()),
            "actual_negative": int(len(y_true) - y_true.sum()),
            "predicted_positive": int(y_pred.sum()),
        }

    print("\nValidation evaluation results")
    print("=" * 65)

    for label_name, metrics in results["metrics"].items():
        print(f"\nTarget: {label_name}")
        print(f"  ROC-AUC:   {metrics['roc_auc']}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-score:  {metrics['f1_score']:.4f}")
        print(f"  Confusion matrix: {metrics['confusion_matrix']}")
        print(f"  Actual positives: {metrics['actual_positive']}")
        print(f"  Predicted positives: {metrics['predicted_positive']}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    REPORT_PATH.write_text(
        json.dumps(results, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"\nEvaluation report saved to: {REPORT_PATH}")
    print(
        "\nResearch experiment only. These results are not evidence "
        "of clinical readiness."
    )


if __name__ == "__main__":
    main()
