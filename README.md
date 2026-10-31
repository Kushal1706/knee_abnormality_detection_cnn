# Knee Abnormality Detection Using Deep Learning

A research prototype for multi-task classification of knee MRI examinations using a multi-view Convolutional Neural Network (CNN).

## 1. Project Overview

Magnetic Resonance Imaging (MRI) provides detailed views of structures inside the knee joint. Reviewing MRI examinations can involve inspecting multiple image slices across different anatomical views.

This project explores deep learning for predicting three binary labels from knee MRI examinations:

- **Abnormality detection**
- **Anterior cruciate ligament (ACL) injury detection**
- **Meniscus injury detection**

The system is an experimental research prototype, not a clinically validated diagnostic tool.

## 2. Objectives

- Train and compare baseline model experiments.
- Evaluate the selected model using validation and held-out test data.
- Develop a CNN based system to identify and classify knee abnormality detection.

## 3. Dataset

The project uses a sample of an MRNet-style knee MRI dataset accessed through **Stanford** Redivis.

The current development dataset contains 1,129 examinations. Each examination has three MRI views and binary labels for abnormality, ACL injury, and meniscus injury.

The examination split is:

| Split | Examinations |
|---|---:|
| Training | 789 |
| Validation | 170 |
| Test | 170 |
| **Total** | **1,129** |

MRI volumes are stored as NumPy arrays. The dataset and generated model checkpoints are not included in the Git repository.

**Data access:** Users must obtain authorized access to the source dataset and place the files in the expected project directories. Follow the dataset provider's terms and conditions.

## 4. Model Architecture

The selected baseline is a multi-view CNN implemented in PyTorch.

1. **Input:** Three MRI views—axial, coronal, and sagittal.
2. **Slice selection:** Eight evenly spaced slices are selected from each view.
3. **Preprocessing:** Pixel intensities are converted to floating-point values and normalized to the range [0, 1].
4. **Feature extraction:** A shared CNN extracts features from the selected slices.
5. **Feature aggregation:** Features from the three views are concatenated.
6. **Classification head:** Fully connected layers produce three output logits, one for each prediction task.
7. **Prediction:** Sigmoid probabilities are compared with a classification threshold to generate binary predictions.

The selected Baseline V2 checkpoint was saved at epoch 10 using validation loss for model selection.

## Model Architecture and Experiments

### 1. MultiViewKneeCNN

The main model, `MultiViewKneeCNN`, is a multi-view convolutional neural network designed to predict three binary labels from knee MRI examinations:

- Abnormality
- Anterior cruciate ligament (ACL) injury
- Meniscus injury

Each examination contains axial, coronal, and sagittal MRI volumes. The preprocessing pipeline selects eight evenly spaced slices from each view and converts them into normalized tensors.

The model uses a shared CNN feature extractor to learn image representations from individual slices. Its convolutional layers use 16, 32, and 64 output channels, with batch normalization, ReLU activation, max pooling, and adaptive average pooling.

The extracted features are combined across the slices and the three MRI views. A fully connected classifier with 128 hidden units then produces three output logits, one for each prediction task.

For the eight-slice configuration, the combined feature representation contains 1,536 features before classification. During inference, sigmoid probabilities and a classification threshold are used to generate binary predictions.

The shared feature extractor allows the model to learn reusable visual features, while the multi-view input provides information from different viewing directions. The three outputs support multi-task learning, allowing a single model to learn the three related classification tasks.

### 2. Baseline and Aggregated Approaches

Two feature-combination approaches were investigated.

**Baseline approach**

The baseline preserves individual slice-level feature representations when combining information from the MRI volume. This allows the classifier to learn from the features extracted from different slices rather than immediately averaging them into one representation per view.

The approach may preserve localized patterns that appear in only a subset of slices. Its trade-off is a larger combined feature representation.

**Aggregated approach**

The aggregated approach averages slice-level features within each MRI view before combining information across the views. This produces a more compact representation and reduces the amount of slice-specific information passed to the classifier.

Aggregation can simplify the model's representation, but averaging may dilute localized patterns that appear in only a few slices.

The two approaches were compared experimentally to investigate how slice-level feature combination affects classification performance.

### 3. Baseline V2 Selection

Baseline V2 was selected as the final experimental checkpoint based on the validation results and the practical computational constraints of the development environment.

The selected checkpoint was saved at epoch 10, using eight slices per view. Its validation ROC-AUC results were:

| Task | Validation ROC-AUC |
|---|---:|
| Abnormality | 0.8067 |
| ACL injury | 0.7433 |
| Meniscus injury | 0.7191 |

An additional 16-slice experiment was stopped after two epochs because of excessive laptop heat and fan noise. The eight-slice configuration provided a more practical balance between MRI slice coverage and computational cost.

Model selection was based on validation performance rather than repeated evaluation on the test set. The selected checkpoint was subsequently evaluated on the held-out test split using a fixed classification threshold of 0.50 for all three tasks.

### 4. Rationale for the Design

The project uses a multi-view CNN because the three MRI views provide complementary visual information. Processing multiple slices helps represent the MRI volume more comprehensively than relying on a single image.

The baseline and aggregated approaches provide a comparison of two feature-combination strategies. Baseline V2 was retained because of its validation performance and practical training cost.

The resulting system is an experimental research prototype. Its predictions are not a clinical diagnosis, and the model has not been clinically validated for use in patient care.

## 5. Technology Stack

- Python 3.12
- PyTorch
- NumPy
- pandas
- scikit-learn
- iterative-stratification
- Matplotlib
- pytest
- Ruff

The development environment used a CPU-only PyTorch build on Windows. Training and inference performance may differ on other hardware.

## 6. Project Structure

```text
knee-abnormality-detection/
├── configs/
├── data/
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── artifacts/
│   ├── checkpoints/
│   └── experiment_logs/
├── reports/
├── scripts/
├── src/
│   └── knee_ai/
│       ├── data/
│       ├── models/
│       ├── preprocessing/
│       ├── training/
│       ├── evaluation/
│       └── inference/
├── tests/
├── docs/
├── pyproject.toml
├── requirements.txt
└── README.md
```

The project follows a medallion-style data organization:

- **Bronze:** Source MRI data and source metadata.
- **Silver:** Validated and quality-checked intermediate data.
- **Gold:** Examination manifests, labels, and data splits used for modelling.

Some directories are created during project setup or execution and may be empty until their corresponding workflows are run.

## 7. Environment Setup

Use Python 3.12 in a virtual environment.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,download]"
```

If PowerShell blocks virtual-environment activation, use the appropriate activation policy permitted on your machine or invoke the environment's Python executable directly.

Confirm the environment:

```powershell
python --version
python scripts\check_environment.py
```

The dataset must be downloaded separately and placed in the directory structure expected by the project scripts. Do not commit the MRI dataset or model checkpoint files to Git.

## 8. Data Preparation and Splits

The repository includes scripts for dataset inspection, metadata validation, manifest construction, data quality checks, and split generation.

Relevant scripts include:

- `scripts/audit_dataset_metadata.py`
- `scripts/validate_dataset_quality.py`
- `scripts/build_examination_manifest.py`
- `scripts/create_dataset_splits.py`
- `scripts/validate_mri_sample.py`

Review each script's arguments and expected inputs before executing it. Some scripts generate or overwrite files, so preserve the existing final manifests and evaluation reports before rerunning data preparation.

## 9. Training and Evaluation

The main training experiment is implemented in:

```text
scripts/train_baseline_v2.py
```

Other experimental scripts are retained for model comparison and investigation. Running a training script can consume substantial CPU time and generate new checkpoints.

The selected checkpoint is expected at:

```text
artifacts/checkpoints/baseline_v2_best.pt
```

The final held-out test evaluation is performed by:

```powershell
python scripts\evaluate_baseline_v2_test.py
```

This script loads the saved checkpoint and evaluates the test split without training. It expects the prepared dataset and checkpoint to be available locally.

The final test outputs are:

```text
reports/baseline_v2_test_evaluation.json
reports/baseline_v2_test_predictions.csv
```

The test evaluation uses a fixed threshold of 0.50 for each task. Do not repeatedly tune the model or thresholds using the held-out test results.

## 10. Final Test Results

The selected Baseline V2 checkpoint was evaluated on 170 held-out examinations.

| Metric | Abnormality | ACL injury | Meniscus injury |
|---|---:|---:|---:|
| ROC-AUC | 0.8620 | 0.7575 | 0.7383 |
| Precision | 0.9691 | 0.3774 | 0.5476 |
| Recall | 0.6861 | 0.6452 | 0.7667 |
| F1-score | 0.8034 | 0.4762 | 0.6389 |

The reported precision, recall, and F1-score use a classification threshold of 0.50. ROC-AUC measures discrimination across thresholds.

These results describe performance on this dataset split only. They do not establish generalization to other hospitals, scanners, populations, or clinical settings.

## 11. Quality Checks

The project includes automated tests for dataset splits, MRI loading, preprocessing, tensor shapes, label handling, and CNN output behaviour.

Run the checks from the repository root:

```powershell
ruff check .
ruff format --check .
pytest -v
```

## 12. Limitations

- The dataset is relatively small for training a medical imaging model.
- The model predicts three binary labels and does not provide detailed localization.
- Performance may be affected by class imbalance, dataset characteristics, and acquisition differences.
- The reported metrics are experimental and require independent external validation.

## 13. Future Work

Potential improvements include:

- Expanding the dataset.
- Evaluating alternative CNN architectures.
- Improving calibration and handling class imbalance.
- Performing external validation on independent data.
