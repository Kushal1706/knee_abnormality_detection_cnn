"""Tests for MRI examination loading."""

from pathlib import Path

import numpy as np

from knee_ai.data.mri_loader import (
    LABEL_COLUMNS,
    VIEW_COLUMNS,
    load_examination,
    load_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "gold" / "manifests" / "train_examination_manifest.csv"
)
MRI_ROOT = PROJECT_ROOT / "data" / "bronze" / "mri_sample"


def test_manifest_has_expected_structure() -> None:
    """Check that the manifest contains valid examinations and labels."""
    manifest = load_manifest(MANIFEST_PATH)

    assert len(manifest) == 1129
    assert manifest["exam_id"].is_unique
    assert set(LABEL_COLUMNS).issubset(manifest.columns)


def test_sample_examination_loads_all_three_views() -> None:
    """Check that a downloaded sample examination can be loaded."""
    manifest = load_manifest(MANIFEST_PATH)

    row = manifest.loc[manifest["exam_id"] == 1].iloc[0]
    examination = load_examination(row, MRI_ROOT)

    assert examination.exam_id == 1
    assert examination.labels.shape == (3,)
    assert np.isin(examination.labels, [0.0, 1.0]).all()

    for view in VIEW_COLUMNS:
        volume = getattr(examination, view)

        assert volume.ndim == 3
        assert volume.shape[1:] == (256, 256)
        assert np.isfinite(volume).all()
