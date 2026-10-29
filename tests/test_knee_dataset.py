"""Tests for the PyTorch knee MRI Dataset."""

from pathlib import Path

import pandas as pd
import torch

from knee_ai.data.knee_dataset import KneeMRIDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_MANIFEST = (
    PROJECT_ROOT / "data" / "bronze" / "manifests" / "sample_examination_manifest.csv"
)

SPLIT_MANIFEST = PROJECT_ROOT / "data" / "gold" / "splits" / "examination_splits.csv"

MRI_ROOT = PROJECT_ROOT / "data" / "bronze" / "mri_sample"


def create_dataset(split: str = "train") -> KneeMRIDataset:
    """Create a Dataset using the downloaded MRI sample."""
    return KneeMRIDataset(
        examination_manifest=SAMPLE_MANIFEST,
        split_manifest=SPLIT_MANIFEST,
        mri_root=MRI_ROOT,
        split=split,
        num_slices=8,
    )


def test_dataset_contains_expected_split_examinations() -> None:
    """The Dataset should contain sample exams assigned to the requested split."""
    dataset = create_dataset(split="train")

    sample_manifest = pd.read_csv(SAMPLE_MANIFEST)
    split_manifest = pd.read_csv(SPLIT_MANIFEST)

    sample_ids = set(sample_manifest["exam_id"])

    expected_train_ids = set(
        split_manifest.loc[split_manifest["split"] == "train", "exam_id"]
    )

    expected_ids = sample_ids.intersection(expected_train_ids)
    actual_ids = set(dataset.manifest["exam_id"])

    assert len(sample_ids) == 10
    assert actual_ids == expected_ids
    assert len(dataset) == len(expected_ids)
    assert len(dataset) > 0


def test_dataset_returns_expected_tensor_shapes() -> None:
    """Each examination should produce correctly shaped MRI tensors."""
    dataset = create_dataset()
    sample = dataset[0]

    assert sample["axial"].shape == (8, 1, 256, 256)
    assert sample["coronal"].shape == (8, 1, 256, 256)
    assert sample["sagittal"].shape == (8, 1, 256, 256)

    assert sample["labels"].shape == (3,)
    assert sample["exam_id"].ndim == 0


def test_dataset_returns_finite_normalized_tensors() -> None:
    """MRI tensors should contain finite values between zero and one."""
    dataset = create_dataset()
    sample = dataset[0]

    for view in ("axial", "coronal", "sagittal"):
        tensor = sample[view]

        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()
        assert tensor.min() >= 0
        assert tensor.max() <= 1


def test_dataset_returns_binary_labels() -> None:
    """The three examination labels should be binary floating-point values."""
    dataset = create_dataset()
    sample = dataset[0]

    assert sample["labels"].dtype == torch.float32
    assert torch.all((sample["labels"] == 0) | (sample["labels"] == 1))


def test_dataset_rejects_invalid_split() -> None:
    """Unknown split names should be rejected."""
    try:
        create_dataset("invalid")
    except ValueError as error:
        assert "Invalid split" in str(error)
    else:
        raise AssertionError("Expected invalid split to raise ValueError.")
