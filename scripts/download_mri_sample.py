"""Download and validate a reproducible sample of labelled MRI examinations."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import redivis

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "gold" / "manifests" / "train_examination_manifest.csv"
)

MRI_ROOT = PROJECT_ROOT / "data" / "bronze" / "mri_sample"

REPORT_DIR = PROJECT_ROOT / "data" / "silver" / "quality_reports"

TABLE_ID = "aimi.mrnet_knee_mri_s:4a2c:v1_0.train:5wjf"

VIEWS = ("axial", "coronal", "sagittal")

SAMPLE_SIZE = 10
RANDOM_SEED = 42
MAX_RETRIES = 2


def select_examinations(manifest: pd.DataFrame) -> pd.DataFrame:
    """Select a reproducible sample, retaining examination 0001."""
    known = manifest.loc[manifest["exam_id"] == "0001"]
    remaining = manifest.loc[manifest["exam_id"] != "0001"]

    if known.empty:
        raise ValueError("Examination 0001 is absent from the labelled manifest.")

    sample_size = min(SAMPLE_SIZE, len(manifest))
    additional_count = sample_size - 1

    additional = remaining.sample(
        n=additional_count,
        random_state=RANDOM_SEED,
    )

    return (
        pd.concat([known, additional], ignore_index=True)
        .sort_values("exam_id")
        .reset_index(drop=True)
    )


def validate_npy(path: Path) -> tuple[bool, str]:
    """Check that a file is a readable, non-empty 3D numeric array."""
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        return False, f"Could not load NumPy array: {exc}"

    if array.size == 0:
        return False, "Array is empty."

    if array.ndim != 3:
        return False, f"Expected 3 dimensions, got {array.ndim}."

    if not np.issubdtype(array.dtype, np.number):
        return False, f"Expected numeric data, got {array.dtype}."

    if not np.isfinite(array).all():
        return False, "Array contains NaN or infinity."

    if array.shape[1:] != (256, 256):
        return False, f"Unexpected slice dimensions: {array.shape[1:]}."

    if array.min() == array.max():
        return False, "All pixel values are identical."

    return True, f"Valid array: shape={array.shape}, dtype={array.dtype}"


def download_one_file(
    table: object,
    relative_path: str,
    destination: Path,
) -> tuple[bool, str]:
    """Download one MRI file and validate the result."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        valid, message = validate_npy(destination)

        if valid:
            return True, f"Already downloaded; {message}"

        print(f"Existing file is invalid; replacing it: {destination}")
        destination.unlink()

    for attempt in range(1, MAX_RETRIES + 1):
        temporary_path = destination.with_name(f"{destination.stem}.download.tmp.npy")

        try:
            if temporary_path.exists():
                temporary_path.unlink()

            print(f"  Download attempt {attempt}/{MAX_RETRIES}")

            remote_file = table.file(relative_path)
            remote_file.download(str(temporary_path))

            if not temporary_path.is_file():
                raise RuntimeError(
                    "Redivis download returned without creating the file."
                )

            valid, message = validate_npy(temporary_path)

            if not valid:
                raise RuntimeError(message)

            temporary_path.replace(destination)

            return True, message

        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            if temporary_path.exists():
                temporary_path.unlink()

            if attempt == MAX_RETRIES:
                return False, f"{type(exc).__name__}: {exc}"

            time.sleep(attempt * 2)

    return False, "Download failed for an unknown reason."


def main() -> None:
    """Download and report the selected MRI sample."""
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    manifest = pd.read_csv(
        MANIFEST_PATH,
        dtype={"exam_id": str},
    )

    sample = select_examinations(manifest)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MRI_ROOT.mkdir(parents=True, exist_ok=True)

    selection_path = REPORT_DIR / "selected_mri_sample.csv"
    sample.to_csv(selection_path, index=False)

    print("=" * 65)
    print("MRNET CONTROLLED SAMPLE DOWNLOAD")
    print("=" * 65)
    print(f"Selected examinations: {len(sample)}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Sample list: {selection_path.relative_to(PROJECT_ROOT)}")

    print("\nConnecting to Redivis...")
    table = redivis.table(TABLE_ID)
    print("Table reference created.")

    results = []

    for _, row in sample.iterrows():
        exam_id = row["exam_id"]
        print(f"\nExamination {exam_id}")

        for view in VIEWS:
            relative_path = row[view]
            expected_name = f"{view}/{exam_id}.npy"

            if relative_path != expected_name:
                raise ValueError(f"Unexpected manifest path: {relative_path}")

            destination = MRI_ROOT / relative_path

            print(f"Downloading/checking {relative_path}")

            success, message = download_one_file(
                table=table,
                relative_path=relative_path,
                destination=destination,
            )

            results.append(
                {
                    "exam_id": exam_id,
                    "view": view,
                    "relative_path": relative_path,
                    "status": "PASS" if success else "FAILED",
                    "message": message,
                    "file_size_bytes": (
                        destination.stat().st_size
                        if success and destination.exists()
                        else None
                    ),
                }
            )

            print(f"  {'PASS' if success else 'FAILED'}: {message}")

            if not success:
                print(
                    "\nDownload stopped after a file failed validation. "
                    "Review the error before continuing."
                )
                break

        if results[-1]["status"] == "FAILED":
            break

    report_path = REPORT_DIR / "sample_download_report.csv"
    pd.DataFrame(results).to_csv(report_path, index=False)

    successful = sum(item["status"] == "PASS" for item in results)
    failed = sum(item["status"] == "FAILED" for item in results)

    print("\n" + "=" * 65)
    print("DOWNLOAD SUMMARY")
    print("=" * 65)
    print(f"Files checked/downloaded successfully: {successful}")
    print(f"Failed files: {failed}")
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")

    if failed:
        raise SystemExit(
            "The download was incomplete. Resolve the reported error before proceeding."
        )


if __name__ == "__main__":
    main()
