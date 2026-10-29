import importlib.util
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADER_PATH = PROJECT_ROOT / "scripts" / "download_mri_development.py"

spec = importlib.util.spec_from_file_location(
    "download_mri_development",
    DOWNLOADER_PATH,
)

if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load downloader module: {DOWNLOADER_PATH}")

downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)

build_development_manifest = downloader.build_development_manifest
SAMPLE_MANIFEST = downloader.SAMPLE_MANIFEST
SPLIT_MANIFEST = downloader.SPLIT_MANIFEST


def test_development_selection_has_expected_sizes() -> None:
    development = build_development_manifest()

    assert len(development) == 200
    assert development["split"].value_counts().to_dict() == {
        "train": 150,
        "validation": 50,
    }


def test_existing_train_and_validation_sample_exams_are_preserved() -> None:
    development = build_development_manifest()

    sample = pd.read_csv(SAMPLE_MANIFEST, dtype={"exam_id": str})
    sample_ids = (
        sample["exam_id"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(4)
    )

    official_splits = pd.read_csv(
        SPLIT_MANIFEST,
        dtype={"exam_id": str},
    )
    official_splits["exam_id"] = (
        official_splits["exam_id"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(4)
    )

    expected_ids = set(
        official_splits.loc[
            official_splits["exam_id"].isin(sample_ids)
            & official_splits["split"].isin(["train", "validation"]),
            "exam_id",
        ]
    )

    assert expected_ids.issubset(set(development["exam_id"]))


def test_development_selection_is_reproducible() -> None:
    first = build_development_manifest().sort_values("exam_id")
    second = build_development_manifest().sort_values("exam_id")

    assert first["exam_id"].tolist() == second["exam_id"].tolist()
    assert first["split"].tolist() == second["split"].tolist()
