# Model Selection — Baseline Comparison V2

## Purpose

This report identifies the validation threshold or thresholds with the highest F1-score for each model and prediction task.

## Experiment configuration

- Validation examinations: 170.
- Models: baseline, aggregated, and baseline_v2.
- Classification thresholds: 0.20 through 0.80.
- Model selection uses validation results only.
- The held-out test split is not used for tuning.

## Results

| Task | Model | Threshold | ROC-AUC | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| abnormality | aggregated | 0.20 | 0.7939 | 0.8059 | 1.0000 | 0.8925 | 0 | 33 | 0 | 137 |
| abnormality | aggregated | 0.30 | 0.7939 | 0.8059 | 1.0000 | 0.8925 | 0 | 33 | 0 | 137 |
| abnormality | baseline | 0.30 | 0.7956 | 0.8261 | 0.9708 | 0.8926 | 5 | 28 | 4 | 133 |
| abnormality | baseline_v2 | 0.20 | 0.8067 | 0.8600 | 0.9416 | 0.8990 | 12 | 21 | 8 | 129 |
| acl | aggregated | 0.50 | 0.6289 | 0.1926 | 0.8387 | 0.3133 | 30 | 109 | 5 | 26 |
| acl | baseline | 0.50 | 0.6616 | 0.3429 | 0.3871 | 0.3636 | 116 | 23 | 19 | 12 |
| acl | baseline_v2 | 0.60 | 0.7433 | 0.7000 | 0.4516 | 0.5490 | 133 | 6 | 17 | 14 |
| meniscus | aggregated | 0.50 | 0.6642 | 0.4098 | 0.8333 | 0.5495 | 38 | 72 | 10 | 50 |
| meniscus | baseline | 0.40 | 0.6805 | 0.3889 | 0.9333 | 0.5490 | 22 | 88 | 4 | 56 |
| meniscus | baseline_v2 | 0.30 | 0.7191 | 0.4355 | 0.9000 | 0.5870 | 40 | 70 | 6 | 54 |

## Interpretation

The highest-F1 threshold is a candidate for further investigation, not an automatically optimal clinical threshold.

ROC-AUC measures ranking performance and does not change when the classification threshold changes.

Precision, recall, F1-score, and confusion-matrix counts depend on the threshold.

Thresholds were explored on the validation set. Repeated selection on the same set may lead to optimistic estimates.

These results are preliminary and do not establish clinical readiness.

## Next step

Review the selected thresholds and their false-positive and false-negative counts. Document the model-selection decision before evaluating the held-out test set.

## Source

- Input: `reports/threshold_analysis_v2_comparison.csv`
- CSV: `reports/model_selection_v2_summary.csv`
- JSON: `reports/model_selection_v2_summary.json`
