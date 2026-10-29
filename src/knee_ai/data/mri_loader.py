"""Load and calidate MRI examinations from the project manifest."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_COLUMNS = (
    "abnormal_label",
    "acl_label",
    "meniscus_label",
)

VIEW_COLUMNS = (
    "axial",
    "coronal",
    "sagittal",
)


@dataclass
class MRIExamination:
    """Represent one examination and its three MRI volumes"""

    exam_id: int
    axial: np.ndarray
    coronal: np.ndarray
    sagittal: np.ndarray
    labels: np.ndarray


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load the examination manifest and validate its required columns."""
    manifest_path = Path(manifest_path)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Examination manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)

    required_columns = {
        "exam_id",
        *VIEW_COLUMNS,
        *LABEL_COLUMNS,
    }

    missing_columns = required_columns.difference(manifest.columns)

    if missing_columns:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing_columns)}"
        )

    if manifest["exam_id"].isna().any():
        raise ValueError("Manifest contains missing examination IDs.")

    if manifest["exam_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate examination IDs.")

    if manifest[list(LABEL_COLUMNS)].isna().any().any():
        raise ValueError("Manifest contains missing labels.")

    labels = manifest[list(LABEL_COLUMNS)].to_numpy()

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("All labels must be binary values: 0 or 1.")

    return manifest


def load_mri_volume(volume_path: Path) -> np.ndarray:
    """Load one MRI volume and validate its basic structure."""
    volume_path = Path(volume_path)

    if not volume_path.is_file():
        raise FileNotFoundError(f"MRI volume not found: {volume_path}")

    volume = np.load(volume_path, allow_pickle=False)

    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D MRI volume at {volume_path}; received shape {volume.shape}."
        )

    if any(dimension == 0 for dimension in volume.shape):
        raise ValueError(f"MRI volume has an empty dimension: {volume_path}")

    if not np.issubdtype(volume.dtype, np.number):
        raise ValueError(f"MRI volume must contain numeric data: {volume_path}")

    if not np.isfinite(volume).all():
        raise ValueError(f"MRI volume contains NaN or infinity: {volume_path}")

    return volume


def load_examination(
    row: pd.Series,
    mri_root: Path,
) -> MRIExamination:
    """Load the three MRI volumes and labels for one manifest row."""
    mri_root = Path(mri_root)

    volumes = {}

    for view in VIEW_COLUMNS:
        relative_path = Path(str(row[view]))

        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe relative MRI path for {view}: {relative_path}")

        volume_path = mri_root / relative_path
        volumes[view] = load_mri_volume(volume_path)

    labels = row[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)

    return MRIExamination(
        exam_id=int(row["exam_id"]),
        axial=volumes["axial"],
        coronal=volumes["coronal"],
        sagittal=volumes["sagittal"],
        labels=labels,
    )
