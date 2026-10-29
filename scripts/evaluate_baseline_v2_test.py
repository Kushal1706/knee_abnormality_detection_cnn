"""Evaluate Baseline V2 once on the held-out test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from knee_ai.data.knee_dataset import KneeMRIDataset
from knee_ai.models.multiview_cnn import MultiViewKneeCNN
from knee_ai.training.train import (
    LABEL_NAMES,
    calculate_metrics,
    move_batch_to_device,
    set_random_seeds,
)

EXAMINATION_MANIFEST = (
    PROJECT_ROOT / "data" / "gold" / "manifests" / "train_examination_manifest.csv"
)
SPLIT_MANIFEST = PROJECT_ROOT / "data" / "gold" / "splits" / "examination_splits.csv"
MRI_ROOT = PROJECT_ROOT / "data" / "bronze" / "mri_sample"

CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "checkpoints" / "baseline_v2_best.pt"
REPORT_DIR = PROJECT_ROOT / "reports"

PREDICTIONS_PATH = REPORT_DIR / "baseline_v2_test_predictions.csv"
REPORT_PATH = REPORT_DIR / "baseline_v2_test_evaluation.json"

NUM_SLICES = 8
BATCH_SIZE = 2
THRESHOLD = 0.50
SEED = 42
EXPECTED_TEST_EXAMINATIONS = 170


def main() -> int:
    """Evaluate the selected checkpoint without modifying model weights."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing final test reports.",
    )
    args = parser.parse_args()

    # Protect existing final reports from accidental replacement.
    existing = [path for path in (PREDICTIONS_PATH, REPORT_PATH) if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Output files already exist: {names}. "
            "Inspect or back them up before using --overwrite."
        )

    for path in (EXAMINATION_MANIFEST, SPLIT_MANIFEST, CHECKPOINT_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    set_random_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 68)
    print("BASELINE V2 — HELD-OUT TEST EVALUATION")
    print("=" * 68)
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH.name}")
    print(f"Threshold: {THRESHOLD:.2f}")

    dataset = KneeMRIDataset(
        examination_manifest=EXAMINATION_MANIFEST,
        split_manifest=SPLIT_MANIFEST,
        mri_root=MRI_ROOT,
        split="test",
        num_slices=NUM_SLICES,
    )

    if len(dataset) != EXPECTED_TEST_EXAMINATIONS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_EXAMINATIONS} test examinations, "
            f"but dataset contains {len(dataset)}. "
            "Check the split manifest before proceeding."
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    checkpoint: dict[str, Any] = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model = MultiViewKneeCNN(num_slices=NUM_SLICES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_ids: list[str] = []
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            views, labels = move_batch_to_device(batch, device)
            probabilities = torch.sigmoid(model(**views))

            batch_ids = batch["exam_id"]
            if isinstance(batch_ids, torch.Tensor):
                batch_ids = batch_ids.cpu().tolist()

            all_ids.extend(str(exam_id) for exam_id in batch_ids)
            all_targets.append(labels.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())

    targets = np.concatenate(all_targets, axis=0).astype(np.int32)
    probabilities = np.concatenate(all_probabilities, axis=0)

    if len(all_ids) != len(dataset) or len(set(all_ids)) != len(all_ids):
        raise ValueError("Test examination IDs are missing or duplicated.")

    if targets.shape != (len(dataset), len(LABEL_NAMES)):
        raise ValueError(f"Unexpected target shape: {targets.shape}")

    if probabilities.shape != targets.shape:
        raise ValueError("Prediction and target shapes do not match.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions contain NaN or infinite values.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Predicted probabilities are outside [0, 1].")

    predictions_binary = (probabilities >= THRESHOLD).astype(np.int32)

    prediction_df = pd.DataFrame({"exam_id": all_ids})
    for index, label_name in enumerate(LABEL_NAMES):
        prediction_df[f"{label_name}_actual"] = targets[:, index]
        prediction_df[f"{label_name}_probability"] = probabilities[:, index]
        prediction_df[f"{label_name}_predicted"] = predictions_binary[:, index]

    auc_results = calculate_metrics(targets, probabilities)

    metrics: dict[str, Any] = {}

    for index, label_name in enumerate(LABEL_NAMES):
        actual = targets[:, index]
        predicted = predictions_binary[:, index]

        tn, fp, fn, tp = confusion_matrix(
            actual,
            predicted,
            labels=[0, 1],
        ).ravel()

        metrics[label_name] = {
            "roc_auc": auc_results[label_name]["roc_auc"],
            "precision": float(precision_score(actual, predicted, zero_division=0)),
            "recall": float(recall_score(actual, predicted, zero_division=0)),
            "f1_score": float(f1_score(actual, predicted, zero_division=0)),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
            "actual_positive": int(actual.sum()),
            "actual_negative": int(len(actual) - actual.sum()),
            "predicted_positive": int(predicted.sum()),
        }

    report: dict[str, Any] = {
        "experiment": "baseline_v2_final_test_evaluation",
        "evaluation_split": "test",
        "test_examinations": len(dataset),
        "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_loss": float(checkpoint["validation_loss"]),
        "num_slices_per_view": NUM_SLICES,
        "classification_threshold": THRESHOLD,
        "device": str(device),
        "metrics": metrics,
        "limitations": [
            "Research prototype; not clinically validated.",
            "Results apply to this held-out dataset and evaluation setup.",
            "Threshold 0.50 was fixed before this test evaluation.",
        ],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Use exclusive creation so a concurrent process cannot silently
    # replace an existing report after the initial existence check.
    with REPORT_PATH.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    try:
        with PREDICTIONS_PATH.open("x", encoding="utf-8", newline="") as file:
            prediction_df.to_csv(file, index=False)
    except Exception:
        # Avoid leaving a seemingly complete report if CSV writing fails.
        REPORT_PATH.unlink(missing_ok=True)
        raise

    print(f"\nTest examinations: {len(dataset)}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")

    for label_name, result in metrics.items():
        print(f"\n--- {label_name.upper()} ---")
        for metric_name in ("roc_auc", "precision", "recall", "f1_score"):
            print(f"{metric_name}: {result[metric_name]:.4f}")
        print(f"Confusion matrix: {result['confusion_matrix']}")

    print(f"\nMetrics report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Predictions: {PREDICTIONS_PATH.relative_to(PROJECT_ROOT)}")
    print("\nTest evaluation completed. No training was performed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
