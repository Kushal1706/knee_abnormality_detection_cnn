## Model Architecture and Experiments

### 1. Overview

The project uses a multi-view convolutional neural network (CNN) to perform three binary classification tasks on knee MRI examinations:

- Abnormality detection
- Anterior cruciate ligament (ACL) injury detection
- Meniscus injury detection

Each examination contains MRI volumes from three anatomical views: axial, coronal, and sagittal. The preprocessing pipeline selects eight evenly spaced slices from each view and converts the images into normalized tensors of size 256 × 256.

The model produces three output logits per examination, one for each classification task. During inference, the sigmoid function converts the logits into scores between 0 and 1, which are thresholded to obtain binary predictions.

### 2. Shared Slice Feature Extractor

Both model architectures use the same `SliceFeatureExtractor` implementation defined in `src/knee_ai/models/multiview_cnn.py`.

The feature extractor processes each grayscale MRI slice independently through three convolutional blocks:

| Layer | Configuration | Purpose |
|---|---|---|
| Input | 1 × 256 × 256 | Receives a grayscale MRI slice. |
| Convolution 1 | 1 → 16 channels, 3 × 3 kernel | Learns initial visual patterns. |
| Batch normalization + ReLU | 16 channels | Normalizes activations and introduces non-linearity. |
| Max pooling | 2 × 2 | Reduces spatial dimensions. |
| Convolution 2 | 16 → 32 channels, 3 × 3 kernel | Learns more complex features. |
| Batch normalization + ReLU | 32 channels | Supports stable feature learning. |
| Max pooling | 2 × 2 | Further reduces spatial dimensions. |
| Convolution 3 | 32 → 64 channels, 3 × 3 kernel | Extracts higher-level visual features. |
| Batch normalization + ReLU | 64 channels | Refines the feature representation. |
| Max pooling | 2 × 2 | Reduces the spatial dimensions to 32 × 32. |
| Adaptive average pooling | 1 × 1 per channel | Compresses each channel into one value. |
| Flatten | 64 features | Produces a 64-dimensional feature vector per slice. |

The same feature extractor instance processes slices from all three MRI views. Consequently, the model learns a shared set of visual feature-extraction parameters rather than maintaining a separate CNN for each view.

### 3. Model A: MultiViewKneeCNN

`MultiViewKneeCNN` is the main model used by the selected Baseline V2 experiment.

For each view, the CNN extracts a 64-dimensional feature vector from each of the eight sampled slices. It then flattens and concatenates these slice-level features.

Each view contributes:

8 slices × 64 features = 512 features.

The three views are concatenated to form a feature vector of:

3 views × 8 slices × 64 features = 1,536 features per examination.

This vector passes through a classification head consisting of:

1. A fully connected layer mapping 1,536 features to 128 hidden units.
2. A ReLU activation.
3. Dropout with probability 0.3.
4. A final fully connected layer mapping 128 units to three output logits.

**Why preserve individual slice features?**

A visual pattern may appear in only a subset of MRI slices. By retaining the feature vector from each sampled slice, the classifier can learn relationships between features from different positions in the volume.

The main trade-off is that concatenation creates a larger feature vector and a larger classification head than mean aggregation.

### 4. Model B: AggregatedMultiViewKneeCNN

`AggregatedMultiViewKneeCNN`, implemented in the aggregated model module, uses the same `SliceFeatureExtractor` but combines slice features differently.

After extracting a 64-dimensional feature vector from each slice, it averages those vectors across the eight sampled slices within each view.

Each view therefore contributes 64 features rather than 512.

The three view representations are concatenated:

3 views × 64 features = 192 features per examination.

The classification head consists of:

1. A fully connected layer mapping 192 features to 128 hidden units.
2. A ReLU activation.
3. Dropout with probability 0.3.
4. A final fully connected layer mapping 128 units to three output logits.

**Why use mean aggregation?**

Mean aggregation creates a compact representation of each MRI view. It reduces the number of features passed to the classifier and substantially reduces the size of the classification head.

However, averaging can dilute a distinctive feature that occurs in only a small number of slices. The aggregated model therefore provides an alternative strategy for investigating how slice-level information should be combined.

### 5. Comparison of the Architectures

| Property | MultiViewKneeCNN | AggregatedMultiViewKneeCNN |
|---|---|---|
| Shared slice encoder | Yes | Yes |
| Features per slice | 64 | 64 |
| Slice-combination method | Concatenation | Mean aggregation |
| Features per view with eight slices | 512 | 64 |
| Combined features across three views | 1,536 | 192 |
| Hidden classifier layer | 128 units | 128 units |
| Output tasks | 3 | 3 |
| Main advantage | Preserves slice-specific feature representations | Produces a compact feature representation |
| Main trade-off | Larger classification head | May dilute localized slice features |

The two architectures use the same basic slice encoder but differ in how they combine the resulting features. This comparison helps investigate whether preserving slice-level information or averaging it produces better results for the available dataset.

### 6. Baseline Experiments and Baseline V2

A baseline is a reference model or experimental configuration used to evaluate the performance of a machine-learning approach and compare subsequent changes.

The baseline experiments establish a reference point for knee MRI classification. The aggregated architecture provides an alternative feature-combination strategy, while Baseline V2 improves the training procedure used with `MultiViewKneeCNN`.

According to `scripts/train_baseline_v2.py`, Baseline V2 uses the following configuration by default:

| Setting | Value |
|---|---|
| Architecture | MultiViewKneeCNN |
| Slices per view | 8 |
| Batch size | 2 |
| Optimizer | AdamW |
| Initial learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Maximum epochs | 30 |
| Early-stopping patience | 6 epochs |
| Learning-rate scheduler | ReduceLROnPlateau |
| Scheduler patience | 2 epochs |
| Minimum learning rate | 0.000001 |
| Random seed | 42 |
| Loss function | Weighted binary cross-entropy with logits |

#### Improvements in the Baseline V2 training procedure

**1. Class-imbalance handling**

The training script calculates a separate positive-class weight for each prediction task using the ratio of negative to positive examples in the training split. These weights are passed to `BCEWithLogitsLoss`.

This is intended to reduce the effect of imbalanced label distributions during training.

**2. AdamW optimization**

AdamW updates the model parameters using adaptive learning rates and applies weight decay to help regularize training.

**3. Adaptive learning-rate scheduling**

`ReduceLROnPlateau` reduces the learning rate when the monitored validation loss stops improving, allowing training to make smaller updates when progress slows.

**4. Early stopping and checkpoint selection**

The script tracks validation loss and saves a checkpoint when it improves. Training stops if validation loss fails to improve for the configured patience period.

This reduces unnecessary training and selects the checkpoint using validation data rather than the held-out test set.

**5. Reproducibility**

The script sets random seeds and records the experiment configuration, training history, best epoch, and best validation loss.

### 7. Why Baseline V2 Was Selected

Baseline V2 was retained as the selected experiment based on the validation results and practical computational constraints.

The recorded validation ROC-AUC results were:

| Prediction task | Validation ROC-AUC |
|---|---:|
| Abnormality | 0.8067 |
| ACL injury | 0.7433 |
| Meniscus injury | 0.7191 |

The selected checkpoint was saved at epoch 10, with a validation loss of approximately 0.6748.

The model uses eight slices per view, providing a practical balance between MRI volume coverage and computational cost on the available CPU-based laptop. A separate 16-slice experiment was stopped after two epochs because the laptop became excessively hot and its fans became loud.

Baseline V2 was selected using validation results, and its final performance was subsequently measured on the held-out test split. The test set was not used for model selection or threshold tuning.

### 6. Baseline Experiments

The project uses two training configurations to develop and evaluate the multi-view CNN.

#### Original Baseline

The original baseline, implemented in `scripts/train_baseline.py`, was designed as a short training experiment to validate the end-to-end pipeline.

It uses `MultiViewKneeCNN`, the Adam optimizer, a learning rate of 0.001, a batch size of 2, and a default maximum of two epochs. It uses weighted binary cross-entropy to address differences in label frequencies and saves the checkpoint with the lowest validation loss.

This experiment established that the model, data loader, loss function, optimizer, and validation pipeline could work together. Because it was limited to two epochs, it was primarily a pipeline-validation baseline rather than the final training configuration.

#### Baseline V2

Baseline V2, implemented in `scripts/train_baseline_v2.py`, uses the same `MultiViewKneeCNN` architecture but introduces a more developed training procedure.

Its default configuration includes:

- AdamW optimization with an initial learning rate of 0.0003.
- Weight decay of 0.0001.
- A maximum of 30 epochs.
- A batch size of 2.
- Weighted binary cross-entropy with logits.
- A `ReduceLROnPlateau` scheduler that reduces the learning rate when validation loss stops improving.
- Early stopping after six consecutive epochs without validation-loss improvement.
- Checkpoint selection based on the lowest validation loss.
- Random seed 42 for reproducibility.

The original baseline and Baseline V2 therefore differ primarily in their training configuration and optimization strategy, rather than in their underlying CNN architecture.

### 7. Why Baseline V2 Was Selected

Baseline V2 was selected as the final experimental model based on the validation results and practical computational constraints.

The selected checkpoint was saved at epoch 10, with a validation loss of approximately 0.6748. Its validation ROC-AUC results were:

| Prediction task | Validation ROC-AUC |
|---|---:|
| Abnormality | 0.8067 |
| ACL injury | 0.7433 |
| Meniscus injury | 0.7191 |

An additional experiment using 16 slices per view was stopped after two epochs because the laptop became excessively hot and its fans became loud. The eight-slice configuration provided a more practical balance between MRI slice coverage and computational cost.

Baseline V2 was selected using validation results. Its final performance was then evaluated on 170 held-out test examinations using a fixed classification threshold of 0.50 for all three tasks. The test set was reserved for evaluation rather than model selection.

The selected model is a research prototype. Its results do not establish clinical readiness, and its predictions should not replace professional radiological assessment.

### 8. Final Test Evaluation

The selected checkpoint was evaluated on 170 held-out test examinations using a fixed classification threshold of 0.50 for all three tasks.

| Prediction task | ROC-AUC | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Abnormality | 0.8620 | 0.9691 | 0.6861 | 0.8034 |
| ACL injury | 0.7575 | 0.3774 | 0.6452 | 0.4762 |
| Meniscus injury | 0.7383 | 0.5476 | 0.7667 | 0.6389 |

ROC-AUC measures how well the model ranks positive examinations above negative examinations across classification thresholds. Precision measures the proportion of positive predictions that are correct, while recall measures the proportion of actual positive examinations detected. The F1-score balances precision and recall.

The results indicate different performance levels across the three tasks. In particular, ACL classification has comparatively low precision, indicating that many examinations predicted as ACL-positive were negative in the test labels at the selected threshold.

These findings should be interpreted in the context of the dataset size, label distribution, model design, and fixed threshold. They do not establish clinical readiness.

### 9. Limitations and Responsible Use

This project is an experimental research prototype developed using a limited dataset. Its test results apply to the specific held-out examinations and evaluation procedure used in this project.

The model has not been clinically validated and should not be used as a substitute for professional radiological assessment or as the sole basis for medical decisions. Further work would require broader datasets, external validation, careful calibration, and evaluation of generalization across scanners and clinical settings.