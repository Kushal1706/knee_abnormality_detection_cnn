from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import redivis
import requests
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# ---------------------------------------------------------
# 1. Project configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXAMINATION_MANIFEST = (
    PROJECT_ROOT / "data" / "gold" / "manifests" / "train_examination_manifest.csv"
)

SPLIT_MANIFEST = PROJECT_ROOT / "data" / "gold" / "splits" / "examination_splits.csv"

SAMPLE_MANIFEST = (
    PROJECT_ROOT / "data" / "bronze" / "manifests" / "sample_examination_manifest.csv"
)

MRI_ROOT = PROJECT_ROOT / "data" / "bronze" / "mri_sample"

OUTPUT_MANIFEST = (
    PROJECT_ROOT / "data" / "gold" / "manifests" / "development_manifest.csv"
)

REPORT_DIR = PROJECT_ROOT / "data" / "silver" / "quality_reports"

TABLE_ID = "aimi.mrnet_knee_mri_s:4a2c:v1_0.train:5wjf"

VIEWS = ("axial", "coronal", "sagittal")

LABEL_COLUMNS = (
    "abnormal_label",
    "acl_label",
    "meniscus_label",
)

MANIFEST_COLUMNS = (
    "exam_id",
    "axial",
    "coronal",
    "sagittal",
    *LABEL_COLUMNS,
)

TARGET_COUNTS = {
    "train": 150,
    "validation": 50,
}

RANDOM_SEED = 42
MAX_RETRIES = 2


# ---------------------------------------------------------
# 2. Validate and normalize examination IDs
# ---------------------------------------------------------


def normalize_exam_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize examination IDs to four digits."""
    result = frame.copy()

    if "exam_id" not in result.columns:
        raise ValueError("The manifest must contain an exam_id column.")

    if result["exam_id"].isna().any():
        raise ValueError("The manifest contains missing examination IDs.")

    ids = result["exam_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    if not ids.str.fullmatch(r"\d+").all():
        raise ValueError("Examination IDs must contain only digits.")

    result["exam_id"] = ids.str.zfill(4)

    if result["exam_id"].duplicated().any():
        raise ValueError("Duplicate examination IDs were found.")

    return result


# ---------------------------------------------------------
# 3. Build a deterministic development selection
# ---------------------------------------------------------


def build_development_manifest() -> pd.DataFrame:
    """Select reproducible, approximately stratified development examinations."""
    full = pd.read_csv(
        EXAMINATION_MANIFEST,
        dtype={"exam_id": str},
    )

    splits = pd.read_csv(
        SPLIT_MANIFEST,
        dtype={"exam_id": str},
    )

    sample = pd.read_csv(
        SAMPLE_MANIFEST,
        dtype={"exam_id": str},
    )

    full = normalize_exam_ids(full)
    splits = normalize_exam_ids(splits)
    sample = normalize_exam_ids(sample)

    missing_columns = set(MANIFEST_COLUMNS) - set(full.columns)
    if missing_columns:
        raise ValueError(f"Full manifest is missing columns: {sorted(missing_columns)}")

    if "split" not in splits.columns:
        raise ValueError("The split manifest must contain a split column.")

    if not set(splits["split"]).issubset({"train", "validation", "test"}):
        raise ValueError("Unexpected split name in split manifest.")

    if set(full["exam_id"]) != set(splits["exam_id"]):
        raise ValueError(
            "The full manifest and split manifest contain different examination IDs."
        )

    if not set(sample["exam_id"]).issubset(set(full["exam_id"])):
        raise ValueError(
            "The sample manifest contains IDs missing from the full manifest."
        )

    label_columns = ["abnormal_label", "acl_label", "meniscus_label"]

    combined = full.merge(
        splits[["exam_id", "split"]],
        on="exam_id",
        how="inner",
        validate="one_to_one",
    )

    selected_parts = []

    for split_name, target_count in TARGET_COUNTS.items():
        candidates = (
            combined.loc[combined["split"] == split_name]
            .sort_values("exam_id")
            .reset_index(drop=True)
        )

        if len(candidates) < target_count:
            raise ValueError(
                f"Only {len(candidates)} examinations are available for "
                f"{split_name}; {target_count} are required."
            )

        labels = candidates[label_columns].to_numpy(dtype=int)

        if not set(labels.ravel()).issubset({0, 1}):
            raise ValueError(f"Invalid label values found in {split_name} data.")

        # Preserve existing downloaded sample examinations from this split.
        anchor_mask = candidates["exam_id"].isin(sample["exam_id"])
        anchors = candidates.loc[anchor_mask].copy()
        remaining = candidates.loc[~anchor_mask].copy()

        if len(anchors) > target_count:
            raise ValueError(
                f"Existing {split_name} sample has more examinations "
                f"than the target of {target_count}."
            )

        needed = target_count - len(anchors)

        if needed == 0:
            selected = anchors
        else:
            if len(remaining) < needed:
                raise ValueError(f"Not enough remaining examinations for {split_name}.")

            remaining_labels = remaining[label_columns].to_numpy(dtype=int)

            # Generate a reproducible stratified subset of the remaining exams.
            splitter = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                train_size=needed,
                test_size=len(remaining) - needed,
                random_state=RANDOM_SEED,
            )

            selected_indices, _ = next(
                splitter.split(remaining_labels, remaining_labels)
            )

            additional = remaining.iloc[selected_indices]
            selected = pd.concat([anchors, additional], ignore_index=True)

        selected_parts.append(selected.sort_values("exam_id").reset_index(drop=True))

    development = pd.concat(selected_parts, ignore_index=True)

    if development["exam_id"].duplicated().any():
        raise ValueError("The development selection contains duplicate IDs.")

    if "test" in set(development["split"]):
        raise ValueError("Test examinations must not enter development data.")

    for split_name, target_count in TARGET_COUNTS.items():
        actual_count = int((development["split"] == split_name).sum())
        if actual_count != target_count:
            raise ValueError(
                f"Expected {target_count} {split_name} examinations, "
                f"found {actual_count}."
            )

    return development


# ---------------------------------------------------------
# 4. Validate downloaded MRI files
# ---------------------------------------------------------


def validate_npy(path: Path) -> None:
    """Raise an error if an MRI file is missing or invalid."""
    if not path.is_file():
        raise FileNotFoundError(f"MRI file not found: {path}")

    array = np.load(path, allow_pickle=False)

    if array.ndim != 3 or array.shape[0] == 0:
        raise ValueError(f"Invalid MRI volume dimensions: {path}")

    if array.shape[1:] != (256, 256):
        raise ValueError(f"Expected 256x256 slices, got {array.shape}: {path}")

    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"MRI volume must be numeric: {path}")

    if not np.isfinite(array).all():
        raise ValueError(f"MRI volume contains non-finite values: {path}")

    if array.min() == array.max():
        raise ValueError(f"MRI volume is constant: {path}")


# ---------------------------------------------------------
# 5. Download one MRI view safely
# ---------------------------------------------------------


def download_one_file(
    table: object,
    relative_path: str,
    destination: Path,
) -> str:
    """Reuse a valid file or download and validate a replacement."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Never download a file again if it already passes validation.
    if destination.exists():
        try:
            validate_npy(destination)
            return "REUSED"
        except (OSError, ValueError):
            destination.unlink(missing_ok=True)

    temporary_path = destination.with_name(destination.stem + ".download.tmp.npy")

    for attempt in range(1, MAX_RETRIES + 1):
        temporary_path.unlink(missing_ok=True)

        try:
            table.file(relative_path).download(str(temporary_path))
            validate_npy(temporary_path)
            temporary_path.replace(destination)
            return "DOWNLOADED"

        except (
            OSError,
            ValueError,
            RuntimeError,
            TimeoutError,
            requests.RequestException,
        ):
            temporary_path.unlink(missing_ok=True)

            if attempt == MAX_RETRIES:
                raise

            time.sleep(attempt * 2)

        finally:
            # Clean up an incomplete temporary download.
            if temporary_path.exists():
                temporary_path.unlink(missing_ok=True)

    raise RuntimeError(f"Download failed: {relative_path}")


# ---------------------------------------------------------
# 6. Download the development dataset
# ---------------------------------------------------------


def main() -> None:
    development = build_development_manifest()

    print("\nDevelopment selection")
    print("---------------------")

    for split_name in TARGET_COUNTS:
        count = int((development["split"] == split_name).sum())
        print(f"{split_name.capitalize()}: {count} examinations")

    print("\nTest examinations will not be downloaded by this script.")
    print(f"MRI directory: {MRI_ROOT}")

    MRI_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    table = redivis.table(TABLE_ID)
    report_rows = []
    failures = []

    for row in development.to_dict(orient="records"):
        exam_id = row["exam_id"]

        for view in VIEWS:
            relative_path = row[view]
            destination = MRI_ROOT / relative_path

            try:
                status = download_one_file(
                    table,
                    relative_path,
                    destination,
                )

                report_rows.append(
                    {
                        "exam_id": exam_id,
                        "split": row["split"],
                        "view": view,
                        "relative_path": relative_path,
                        "status": status,
                        "error": "",
                    }
                )

                print(f"[{status}] {exam_id} | {view}")

            except (
                OSError,
                ValueError,
                RuntimeError,
                TimeoutError,
                requests.RequestException,
            ) as error:
                failures.append(
                    {
                        "exam_id": exam_id,
                        "view": view,
                        "relative_path": relative_path,
                        "error": str(error),
                    }
                )

                report_rows.append(
                    {
                        "exam_id": exam_id,
                        "split": row["split"],
                        "view": view,
                        "relative_path": relative_path,
                        "status": "FAILED",
                        "error": str(error),
                    }
                )

                print(f"[FAILED] {exam_id} | {view}: {error}")

    report_path = REPORT_DIR / "development_download_report.csv"

    pd.DataFrame(report_rows).to_csv(report_path, index=False)

    if failures:
        print(f"\nDownload failures: {len(failures)}")
        print(f"Report: {report_path}")
        print(
            "The development manifest was not updated because some "
            "MRI files failed. Fix the issue and rerun the script."
        )
        raise SystemExit(1)

    # Confirm that all three views for every selected examination
    # are present and valid before publishing the manifest.
    for row in development.to_dict(orient="records"):
        for view in VIEWS:
            validate_npy(MRI_ROOT / row[view])

    # Do not include the split column here. The existing dataset loader
    # joins this manifest to the official split manifest.
    output_manifest = development[list(MANIFEST_COLUMNS)].copy()

    # Keep IDs compatible with the existing split manifest.
    output_manifest["exam_id"] = (
        output_manifest["exam_id"].str.lstrip("0").replace("", "0")
    )

    output_manifest.to_csv(OUTPUT_MANIFEST, index=False)

    print("\nDownload completed successfully.")
    print(f"Development manifest: {OUTPUT_MANIFEST}")
    print(f"Download report: {report_path}")
    print(f"Total examinations: {len(development)}")
    print(f"Total MRI files expected: {len(development) * len(VIEWS)}")


if __name__ == "__main__":
    main()
