"""Validate locally available MRI volumes against the gold manifest."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "gold" / "manifests" / "train_examination_manifest.csv"
)

SAMPLE_PATH = (
    PROJECT_ROOT / "data" / "silver" / "quality_reports" / "selected_mri_sample.csv"
)

MRI_ROOT = PROJECT_ROOT / "data" / "bronze" / "mri_sample"

REPORT_DIR = PROJECT_ROOT / "data" / "silver" / "quality_reports"

REPORT_PATH = REPORT_DIR / "mri_sample_quality_report.csv"

VIEWS = ("axial", "coronal", "sagittal")


def load_manifest() -> pd.DataFrame:
    """Load and validate the examination manifest."""
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    manifest = pd.read_csv(
        MANIFEST_PATH,
        dtype={"exam_id": str},
    )

    required_columns = {
        "exam_id",
        "axial",
        "coronal",
        "sagittal",
        "abnormal_label",
        "acl_label",
        "meniscus_label",
    }

    missing_columns = required_columns - set(manifest.columns)

    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

    if manifest["exam_id"].isna().any():
        raise ValueError("Manifest contains missing examination IDs.")

    if manifest["exam_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate examination IDs.")

    return manifest


def load_selected_sample(manifest: pd.DataFrame) -> pd.DataFrame:
    """Load the exact examination sample selected by the downloader."""
    if not SAMPLE_PATH.is_file():
        raise FileNotFoundError(
            f"Selected sample file not found: {SAMPLE_PATH}\n"
            "Run scripts/download_mri_sample.py first."
        )

    selected = pd.read_csv(
        SAMPLE_PATH,
        dtype={"exam_id": str},
    )

    if "exam_id" not in selected.columns:
        raise ValueError("The selected sample CSV must contain an 'exam_id' column.")

    if selected["exam_id"].isna().any():
        raise ValueError("The selected sample contains missing examination IDs.")

    if selected["exam_id"].duplicated().any():
        raise ValueError("The selected sample contains duplicate examination IDs.")

    if selected.empty:
        raise ValueError("The selected sample contains no examinations.")

    # Ensure that every selected examination belongs to the labelled manifest.
    manifest_ids = set(manifest["exam_id"])
    selected_ids = set(selected["exam_id"])

    unknown_ids = selected_ids - manifest_ids

    if unknown_ids:
        raise ValueError(
            "Selected examinations are absent from the gold manifest: "
            f"{sorted(unknown_ids)}"
        )

    # Use the gold manifest as the source of MRI paths and labels.
    sample = manifest.loc[manifest["exam_id"].isin(selected_ids)].copy()

    # Preserve the exact order recorded by the downloader.
    sample = sample.set_index("exam_id").loc[selected["exam_id"].tolist()].reset_index()

    return sample


def inspect_volume(
    exam_id: str,
    view: str,
    relative_path: str,
) -> dict[str, object]:
    """Inspect one locally available MRI volume."""
    file_path = MRI_ROOT / relative_path

    result: dict[str, object] = {
        "exam_id": exam_id,
        "view": view,
        "relative_path": relative_path,
        "exists": file_path.is_file(),
        "file_size_bytes": None,
        "shape": None,
        "dimensions": None,
        "dtype": None,
        "slice_count": None,
        "height": None,
        "width": None,
        "minimum": None,
        "maximum": None,
        "mean": None,
        "std": None,
        "all_finite": None,
        "status": None,
        "error": None,
    }

    if not file_path.is_file():
        result["status"] = "MISSING"
        result["error"] = "File is not present locally."
        return result

    result["file_size_bytes"] = file_path.stat().st_size

    try:
        volume = np.load(file_path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        result["status"] = "LOAD_ERROR"
        result["error"] = str(exc)
        return result

    result["shape"] = str(volume.shape)
    result["dimensions"] = volume.ndim
    result["dtype"] = str(volume.dtype)

    if volume.ndim == 3:
        result["slice_count"] = volume.shape[0]
        result["height"] = volume.shape[1]
        result["width"] = volume.shape[2]

    if volume.size == 0:
        result["status"] = "EMPTY_ARRAY"
        result["error"] = "The array contains no elements."
        return result

    if not np.issubdtype(volume.dtype, np.number):
        result["status"] = "INVALID_DTYPE"
        result["error"] = "The array is not numeric."
        return result

    result["all_finite"] = bool(np.isfinite(volume).all())

    if not result["all_finite"]:
        result["status"] = "NON_FINITE_VALUES"
        result["error"] = "The array contains NaN or infinity."
        return result

    result["minimum"] = float(volume.min())
    result["maximum"] = float(volume.max())
    result["mean"] = float(volume.mean())
    result["std"] = float(volume.std())

    if volume.ndim != 3:
        result["status"] = "UNEXPECTED_DIMENSIONS"
        result["error"] = f"Expected a 3D volume; received {volume.ndim} dimensions."
        return result

    if any(dimension == 0 for dimension in volume.shape):
        result["status"] = "INVALID_SHAPE"
        result["error"] = "One or more array dimensions are zero."
        return result

    if volume.shape[1:] != (256, 256):
        result["status"] = "UNEXPECTED_IMAGE_SIZE"
        result["error"] = f"Expected 256x256 slices; found {volume.shape[1:]}."
        return result

    if volume.min() == volume.max():
        result["status"] = "CONSTANT_VOLUME"
        result["error"] = "All voxel values are identical."
        return result

    result["status"] = "PASS"

    return result


def main() -> None:
    """Run the quality checks and save a CSV report."""
    print("=" * 65)
    print("MRI DATASET QUALITY CHECK")
    print("=" * 65)

    manifest = load_manifest()
    sample = load_selected_sample(manifest)

    print(f"Labelled examinations in manifest: {len(manifest):,}")
    print(f"Examinations selected for inspection: {len(sample):,}")
    print(f"Local MRI directory: {MRI_ROOT}")
    print(f"Sample definition: {SAMPLE_PATH.relative_to(PROJECT_ROOT)}")

    report_rows = []

    for _, row in sample.iterrows():
        exam_id = row["exam_id"]

        print(f"\nExam {exam_id}")

        for view in VIEWS:
            result = inspect_volume(
                exam_id=exam_id,
                view=view,
                relative_path=row[view],
            )

            report_rows.append(result)

            print(f"  {view:8} {result['status']:24} {result['shape'] or '-'}")

    report = pd.DataFrame(report_rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    status_counts = report["status"].value_counts()

    passing = int((report["status"] == "PASS").sum())
    missing = int((report["status"] == "MISSING").sum())
    other_issues = len(report) - passing - missing

    print("\n" + "=" * 65)
    print("QUALITY CHECK SUMMARY")
    print("=" * 65)

    print(f"Files inspected: {len(report):,}")
    print(f"Files passing checks: {passing:,}")
    print(f"Missing files: {missing:,}")
    print(f"Other issues: {other_issues:,}")

    print("\nStatus counts:")
    print(status_counts.to_string())

    print(f"\nReport saved to: {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    if other_issues > 0 or missing > 0:
        print("\nWARNING: Some MRI files did not pass the quality checks.")
    else:
        print("\nSUCCESS: All selected MRI files passed the quality checks.")


if __name__ == "__main__":
    main()
