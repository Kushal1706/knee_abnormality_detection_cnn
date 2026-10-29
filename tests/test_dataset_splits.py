"""Tests for examination-level dataset splitting."""

import pandas as pd
import pytest

from knee_ai.data.dataset_splits import LABEL_COLUMNS, create_splits


@pytest.fixture
def sample_manifest() -> pd.DataFrame:
    """Create a synthetic manifest with three binary labels."""
    rows = []

    for exam_id in range(120):
        rows.append(
            {
                "exam_id": exam_id,
                "abnormal_label": int(exam_id % 5 != 0),
                "acl_label": int(exam_id % 4 == 0),
                "meniscus_label": int(exam_id % 3 == 0),
            }
        )

    return pd.DataFrame(rows)


def test_every_examination_receives_one_split(
    sample_manifest: pd.DataFrame,
) -> None:
    """Every input examination must appear exactly once."""
    result = create_splits(sample_manifest)

    assert len(result) == len(sample_manifest)
    assert result["exam_id"].is_unique
    assert result["split"].notna().all()

    assert set(result["split"]) == {"train", "validation", "test"}


def test_splits_are_disjoint(sample_manifest: pd.DataFrame) -> None:
    """An examination must not appear in more than one split."""
    result = create_splits(sample_manifest)

    exam_ids = {
        split: set(result.loc[result["split"] == split, "exam_id"])
        for split in ["train", "validation", "test"]
    }

    assert exam_ids["train"].isdisjoint(exam_ids["validation"])
    assert exam_ids["train"].isdisjoint(exam_ids["test"])
    assert exam_ids["validation"].isdisjoint(exam_ids["test"])


def test_split_assignment_is_reproducible(
    sample_manifest: pd.DataFrame,
) -> None:
    """Identical input should produce identical split assignments."""
    first = create_splits(sample_manifest)
    second = create_splits(sample_manifest)

    pd.testing.assert_frame_equal(first, second)


def test_split_proportions_are_approximately_correct(
    sample_manifest: pd.DataFrame,
) -> None:
    """Split sizes should be close to the target proportions."""
    result = create_splits(sample_manifest)

    proportions = result["split"].value_counts(normalize=True)

    assert proportions["train"] == pytest.approx(0.70, abs=0.03)
    assert proportions["validation"] == pytest.approx(0.15, abs=0.03)
    assert proportions["test"] == pytest.approx(0.15, abs=0.03)


def test_missing_label_is_rejected(
    sample_manifest: pd.DataFrame,
) -> None:
    """Missing labels must not silently enter a split."""
    sample_manifest.loc[0, LABEL_COLUMNS[0]] = float("nan")

    with pytest.raises(ValueError, match="Labels must not contain missing values"):
        create_splits(sample_manifest)


def test_duplicate_examination_ids_are_rejected(
    sample_manifest: pd.DataFrame,
) -> None:
    """Duplicate examination IDs must be rejected."""
    sample_manifest.loc[1, "exam_id"] = sample_manifest.loc[0, "exam_id"]

    with pytest.raises(ValueError, match="must be unique"):
        create_splits(sample_manifest)
