"""Create reproducible examination-level dataset splits."""

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

LABEL_COLUMNS = [
    "abnormal_label",
    "acl_label",
    "meniscus_label",
]

RANDOM_SEED = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15


def create_splits(manifest: pd.DataFrame) -> pd.DataFrame:
    """Assign examinations to train, validation, and test sets."""
    required_columns = ["exam_id", *LABEL_COLUMNS]

    missing_columns = set(required_columns) - set(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing_columns)}"
        )

    data = manifest[required_columns].copy()

    if data["exam_id"].isna().any():
        raise ValueError("Examination IDs must not be missing.")

    if data["exam_id"].duplicated().any():
        raise ValueError("Each examination ID must be unique.")

    if data[LABEL_COLUMNS].isna().any().any():
        raise ValueError("Labels must not contain missing values.")

    labels = data[LABEL_COLUMNS].to_numpy()

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("All labels must be binary values: 0 or 1.")

    if len(data) < 10:
        raise ValueError("Not enough examinations to create dataset splits.")

    first_split = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    remaining_indices, test_indices = next(first_split.split(data[["exam_id"]], labels))

    remaining_data = data.iloc[remaining_indices].reset_index(drop=True)
    remaining_labels = remaining_data[LABEL_COLUMNS].to_numpy()

    relative_validation_size = VALIDATION_SIZE / (1.0 - TEST_SIZE)

    second_split = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=relative_validation_size,
        random_state=RANDOM_SEED + 1,
    )

    train_indices, validation_indices = next(
        second_split.split(
            remaining_data[["exam_id"]],
            remaining_labels,
        )
    )

    train_ids = set(remaining_data.iloc[train_indices]["exam_id"])
    validation_ids = set(remaining_data.iloc[validation_indices]["exam_id"])
    test_ids = set(data.iloc[test_indices]["exam_id"])

    if train_ids & validation_ids:
        raise RuntimeError("Training and validation examinations overlap.")

    if train_ids & test_ids:
        raise RuntimeError("Training and test examinations overlap.")

    if validation_ids & test_ids:
        raise RuntimeError("Validation and test examinations overlap.")

    data["split"] = data["exam_id"].map(
        lambda exam_id: (
            "train"
            if exam_id in train_ids
            else "validation"
            if exam_id in validation_ids
            else "test"
        )
    )

    if data["split"].isna().any():
        raise RuntimeError("Some examinations were not assigned to a split.")

    return data.sort_values("exam_id").reset_index(drop=True)
