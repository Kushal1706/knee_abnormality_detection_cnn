"""Summarize model selection results for the three-model comparison."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"

INPUT_FILE = REPORT_DIR / "threshold_analysis_v2_comparison.csv"

CSV_OUTPUT = REPORT_DIR / "model_selection_v2_summary.csv"
JSON_OUTPUT = REPORT_DIR / "model_selection_v2_summary.json"
NOTES_OUTPUT = REPORT_DIR / "model_selection_v2_notes.md"

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
    """Generate model-selection reports from the latest threshold analysis."""
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

    # Every model must have been evaluated on the same validation population.
    expected_rows = 170
    if not (df["actual_positive"] + df["actual_negative"]).eq(expected_rows).all():
        raise ValueError(
            "Unexpected validation population size. "
            "Verify the manifest and split before summarizing."
        )

    expected_models = {"baseline", "aggregated", "baseline_v2"}
    actual_models = set(df["model"].unique())

    if actual_models != expected_models:
        raise ValueError(
            f"Expected models {sorted(expected_models)}, "
            f"but found {sorted(actual_models)}."
        )

    # Select the highest validation F1 per model and task.
    # Retain all thresholds tied for the maximum.
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

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(CSV_OUTPUT, index=False)
    summary.to_json(
        JSON_OUTPUT,
        orient="records",
        indent=2,
        double_precision=8,
    )

    lines = [
        "# Model Selection — Baseline Comparison V2",
        "",
        "## Purpose",
        "",
        (
            "This report identifies the validation threshold or thresholds "
            "with the highest F1-score for each model and prediction task."
        ),
        "",
        "## Experiment configuration",
        "",
        "- Validation examinations: 170.",
        "- Models: baseline, aggregated, and baseline_v2.",
        "- Classification thresholds: 0.20 through 0.80.",
        "- Model selection uses validation results only.",
        "- The held-out test split is not used for tuning.",
        "",
        "## Results",
        "",
        (
            "| Task | Model | Threshold | ROC-AUC | Precision | "
            "Recall | F1 | TN | FP | FN | TP |"
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
            (
                "The highest-F1 threshold is a candidate for further "
                "investigation, not an automatically optimal clinical threshold."
            ),
            "",
            (
                "ROC-AUC measures ranking performance and does not change "
                "when the classification threshold changes."
            ),
            "",
            (
                "Precision, recall, F1-score, and confusion-matrix counts "
                "depend on the threshold."
            ),
            "",
            (
                "Thresholds were explored on the validation set. Repeated "
                "selection on the same set may lead to optimistic estimates."
            ),
            "",
            ("These results are preliminary and do not establish clinical readiness."),
            "",
            "## Next step",
            "",
            (
                "Review the selected thresholds and their false-positive "
                "and false-negative counts. Document the model-selection "
                "decision before evaluating the held-out test set."
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

    print("=" * 70)
    print("MODEL-SELECTION SUMMARY — V2 COMPARISON")
    print("=" * 70)
    print(f"Input: {INPUT_FILE.relative_to(PROJECT_ROOT)}")
    print(f"CSV:   {CSV_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"JSON:  {JSON_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"Notes: {NOTES_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"\nSelected rows, including ties: {len(summary)}")
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
                "false_positive",
                "false_negative",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
