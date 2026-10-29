"""PyTorch Dataset for knee MRI examinations."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from knee_ai.data.mri_loader import (
    LABEL_COLUMNS,
    VIEW_COLUMNS,
    load_examination,
)
from knee_ai.preprocessing.mri_preprocessor import preprocess_mri_volume


class KneeMRIDataset(Dataset):
    """Load and preprocess individual knee MRI examinations."""

    def __init__(
        self,
        examination_manifest: Path,
        split_manifest: Path,
        mri_root: Path,
        split: str,
        num_slices: int = 8,
    ) -> None:
        """Initialize the dataset for one split."""
        self.mri_root = Path(mri_root)
        self.num_slices = num_slices

        if num_slices < 1:
            raise ValueError("num_slices must be at least 1.")

        valid_splits = {"train", "validation", "test"}
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split {split!r}. Expected one of {sorted(valid_splits)}."
            )

        # Load the original manifest, which contains MRI paths and labels.
        examination_manifest = pd.read_csv(examination_manifest)

        # Load the split assignment generated earlier.
        split_manifest = pd.read_csv(split_manifest)

        required_examination_columns = {
            "exam_id",
            *VIEW_COLUMNS,
            *LABEL_COLUMNS,
        }
        required_split_columns = {"exam_id", "split"}

        missing_examination_columns = required_examination_columns - set(
            examination_manifest.columns
        )
        missing_split_columns = required_split_columns - set(split_manifest.columns)

        if missing_examination_columns:
            raise ValueError(
                "Examination manifest is missing columns: "
                f"{sorted(missing_examination_columns)}"
            )

        if missing_split_columns:
            raise ValueError(
                f"Split manifest is missing columns: {sorted(missing_split_columns)}"
            )

        if examination_manifest["exam_id"].duplicated().any():
            raise ValueError("Examination manifest contains duplicate IDs.")

        if split_manifest["exam_id"].duplicated().any():
            raise ValueError("Split manifest contains duplicate IDs.")

        if split_manifest["split"].isna().any():
            raise ValueError("Split manifest contains missing split assignments.")

        if not set(split_manifest["split"]).issubset(valid_splits):
            raise ValueError("Split manifest contains an invalid split name.")

        # Use examination IDs for the join; never rely on row order.
        combined = examination_manifest.merge(
            split_manifest[["exam_id", "split"]],
            on="exam_id",
            how="inner",
            validate="one_to_one",
        )

        if len(combined) != len(examination_manifest):
            raise ValueError(
                "The examination and split manifests do not contain "
                "the same examination IDs."
            )

        self.manifest = combined.loc[combined["split"] == split].reset_index(drop=True)

        if self.manifest.empty:
            raise ValueError(f"No examinations found for split {split!r}.")

        if self.manifest[list(LABEL_COLUMNS)].isna().any().any():
            raise ValueError("Selected examinations contain missing labels.")

        labels = self.manifest[list(LABEL_COLUMNS)].to_numpy()
        if not pd.Series(labels.ravel()).isin([0, 1]).all():
            raise ValueError("All labels must be binary values: 0 or 1.")

    def __len__(self) -> int:
        """Return the number of examinations in this split."""
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load and preprocess one examination."""
        row = self.manifest.iloc[index]

        examination = load_examination(
            row=row,
            mri_root=self.mri_root,
        )

        axial = preprocess_mri_volume(
            examination.axial,
            num_slices=self.num_slices,
        )
        coronal = preprocess_mri_volume(
            examination.coronal,
            num_slices=self.num_slices,
        )
        sagittal = preprocess_mri_volume(
            examination.sagittal,
            num_slices=self.num_slices,
        )

        return {
            "exam_id": torch.tensor(examination.exam_id, dtype=torch.int64),
            "axial": axial,
            "coronal": coronal,
            "sagittal": sagittal,
            "labels": torch.tensor(examination.labels, dtype=torch.float32),
        }
