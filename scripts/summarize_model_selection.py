from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
INPUT_FILE = REPORT_DIR / "threshold_analysis.csv"

CSV_OUTPUT = REPORT_DIR / "model_selection_summary.csv"
JSON_OUTPUT = REPORT_DIR / "model_selection_summary.json"
NOTES_OUTPUT = REPORT_DIR / "model_selection_notes.md"

GROUP_COLUMNS = ["model", "architecture", "target"]

REQUIRED_COLUMNS = {
    "model",
    "architecture",
    "checkpoint",
    "checkpoint_epoch",
    "target",
    "threshold",
    "roc_auc",
    "precision",
    "recall",
    "f1_score",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
    "actual_positive",
    "actual_negative",
    "predicted_positive",
}


def main() -> int:
    """Generate CSV, JSON, and Markdown selection reports."""
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Report not found: {INPUT_FILE}\nRun scripts/analyze_thresholds.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    if df.empty:
        raise ValueError("The threshold-analysis report is empty.")

    numeric_columns = [
        "threshold",
        "roc_auc",
        "precision",
        "recall",
        "f1_score",
        "checkpoint_epoch",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "actual_positive",
        "actual_negative",
        "predicted_positive",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    # Find the highest validation F1 for each model and prediction task.
    # Preserve every threshold if multiple thresholds tie for best F1.
    maximum_f1 = df.groupby(GROUP_COLUMNS)["f1_score"].transform("max")

    summary = df.loc[(df["f1_score"] - maximum_f1).abs() <= 1e-12].copy()

    summary = summary.sort_values(["target", "model", "threshold"]).reset_index(
        drop=True
    )

    output_columns = [
        "model",
        "architecture",
        "checkpoint",
        "checkpoint_epoch",
        "target",
        "threshold",
        "roc_auc",
        "precision",
        "recall",
        "f1_score",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "actual_positive",
        "actual_negative",
        "predicted_positive",
    ]

    summary = summary[output_columns]

    # Save machine-readable reports.
    summary.to_csv(CSV_OUTPUT, index=False)
    summary.to_json(
        JSON_OUTPUT,
        orient="records",
        indent=2,
        double_precision=8,
    )

    # Generate a readable Markdown report.
    lines = [
        "# Validation-Based Model Selection",
        "",
        "## Purpose",
        "",
        (
            "This report identifies the threshold or thresholds with the "
            "highest validation F1 score for each model and prediction task."
        ),
        "",
        "These are preliminary experimental results, not clinical performance claims.",
        "",
        "## Candidate results",
        "",
        (
            "| Task | Model | Threshold | ROC-AUC | Precision | Recall | "
            "F1 | TN | FP | FN | TP |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.model} | {row.threshold:.2f} | "
            f"{row.roc_auc:.4f} | {row.precision:.4f} | "
            f"{row.recall:.4f} | {row.f1_score:.4f} | "
            f"{row.true_negative} | {row.false_positive} | "
            f"{row.false_negative} | {row.true_positive} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Threshold selection uses validation data only.",
            "- ROC-AUC remains unchanged across thresholds for the same model.",
            (
                "- Precision, recall, F1, and confusion-matrix counts depend "
                "on the selected threshold."
            ),
            (
                "- High recall can come at the cost of many false positives. "
                "Consider FP and FN alongside F1."
            ),
            "- All thresholds tied for the highest F1 are retained.",
            (
                "- Validation results are preliminary, particularly for ACL, "
                "where the number of positive cases is limited."
            ),
            "",
            "## Next step",
            "",
            (
                "Document and freeze the selected model and threshold choices "
                "before evaluating the held-out test set. Do not use test "
                "results to tune these choices."
            ),
            "",
            (
                "This experimental system is not intended for diagnosis or "
                "clinical decision-making."
            ),
            "",
            "## Source",
            "",
            f"- Input: `{INPUT_FILE.relative_to(PROJECT_ROOT).as_posix()}`",
            f"- CSV: `{CSV_OUTPUT.relative_to(PROJECT_ROOT).as_posix()}`",
            f"- JSON: `{JSON_OUTPUT.relative_to(PROJECT_ROOT).as_posix()}`",
        ]
    )

    NOTES_OUTPUT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 65)
    print("MODEL-SELECTION SUMMARY")
    print("=" * 65)
    print(f"\nInput: {INPUT_FILE.relative_to(PROJECT_ROOT)}")
    print(f"CSV:   {CSV_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"JSON:  {JSON_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"Notes: {NOTES_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"\nSelected rows, including ties: {len(summary)}")
    print("\nCandidate results:")
    print(
        summary[
            [
                "target",
                "model",
                "threshold",
                "roc_auc",
                "precision",
                "recall",
                "f1_score",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
