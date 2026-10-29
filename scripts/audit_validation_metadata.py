"""Audit validation MRI index and label metadata without overwriting reports."""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_ROOT / "data" / "bronze" / "source_metadata"
REPORT_DIR = PROJECT_ROOT / "data" / "silver" / "quality_reports"

INDEX_FILE = METADATA_DIR / "valid.csv"

LABEL_FILES = {
    "abnormal": METADATA_DIR / "valid_abnormal.csv",
    "acl": METADATA_DIR / "valid_acl.csv",
    "meniscus": METADATA_DIR / "valid_meniscus.csv",
}

SUMMARY_FILE = REPORT_DIR / "validation_metadata_audit_summary.csv"
DETAIL_FILE = REPORT_DIR / "validation_metadata_audit_details.csv"
LABEL_MERGE_FILE = REPORT_DIR / "validation_label_alignment.csv"


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV while preserving examination IDs as strings."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

    return pd.read_csv(path, dtype=str)


def normalize_ids(values: pd.Series) -> pd.Series:
    """Normalize numeric examination IDs to four-digit strings."""
    ids = values.astype("string").str.strip()
    numeric = ids.str.fullmatch(r"\d+").fillna(False)
    ids.loc[numeric] = ids.loc[numeric].str.zfill(4)
    return ids


def main() -> int:
    """Audit validation IDs, views, and labels."""
    print("=" * 68)
    print("MRNet VALIDATION METADATA AUDIT")
    print("=" * 68)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    destinations = [SUMMARY_FILE, DETAIL_FILE, LABEL_MERGE_FILE]
    existing = [path for path in destinations if path.exists()]

    if existing:
        raise FileExistsError(
            "Audit cancelled to protect existing reports:\n"
            + "\n".join(str(path) for path in existing)
        )

    index_df = load_csv(INDEX_FILE)

    if "file_name" not in index_df.columns:
        raise ValueError("valid.csv is missing the file_name column.")

    # Extract examination IDs and MRI views from paths such as
    # coronal/1249.npy.
    paths = index_df["file_name"].astype("string").str.strip()
    parts = paths.str.extract(r"^(axial|coronal|sagittal)/([^/]+)\.npy$")
    index_df["view"] = parts[0]
    index_df["exam_id"] = normalize_ids(parts[1])

    invalid_paths = index_df["view"].isna() | index_df["exam_id"].isna()

    if invalid_paths.any():
        print("\nWARNING: Invalid MRI paths found:")
        print(index_df.loc[invalid_paths, "file_name"].head(10).to_string())

    valid_index = index_df.loc[~invalid_paths].copy()

    if valid_index.empty:
        raise ValueError("No valid examination IDs could be extracted from valid.csv.")

    index_ids = set(valid_index["exam_id"].dropna())
    duplicate_paths = int(valid_index["file_name"].duplicated().sum())

    print(f"\nIndex file rows: {len(index_df):,}")
    print(f"Valid MRI path rows: {len(valid_index):,}")
    print(f"Unique examinations in index: {len(index_ids):,}")
    print(f"Duplicate file paths: {duplicate_paths:,}")

    print("\nMRI view counts:")
    print(valid_index["view"].value_counts().sort_index().to_string())

    # One row per examination, with counts for each MRI view.
    view_counts = (
        valid_index.pivot_table(
            index="exam_id",
            columns="view",
            values="file_name",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(columns=["axial", "coronal", "sagittal"], fill_value=0)
        .reset_index()
    )

    expected_views = ["axial", "coronal", "sagittal"]
    view_counts["missing_views"] = view_counts.apply(
        lambda row: ",".join(view for view in expected_views if row[view] == 0),
        axis=1,
    )

    view_counts["duplicate_view_files"] = view_counts[expected_views].gt(1).any(axis=1)

    print(
        "Examinations with all three views: "
        f"{(view_counts['missing_views'] == '').sum():,}"
    )
    print(
        "Examinations missing one or more views: "
        f"{(view_counts['missing_views'] != '').sum():,}"
    )

    label_frames = {}
    summary_rows = []
    detail_frames = []

    for task, path in LABEL_FILES.items():
        labels_df = load_csv(path)

        if len(labels_df.columns) != 2:
            raise ValueError(
                f"{path.name} must contain exactly two columns; "
                f"found {list(labels_df.columns)}"
            )

        id_column, label_column = labels_df.columns

        ids = normalize_ids(labels_df[id_column])
        labels = labels_df[label_column].astype("string").str.strip()

        missing_ids = ids.isna() | ids.eq("")
        missing_labels = labels.isna() | labels.eq("")
        invalid_labels = ~labels.isin({"0", "1"}) & ~missing_labels
        duplicate_ids = ids.duplicated(keep=False) & ~missing_ids

        label_id_set = set(ids.dropna()) - {""}
        ids_not_in_index = sorted(label_id_set - index_ids)
        index_ids_without_labels = sorted(index_ids - label_id_set)

        print(f"\n--- {task.upper()} ---")
        print(f"Rows: {len(labels_df):,}")
        print(f"Unique examination IDs: {ids.nunique(dropna=True):,}")
        print(f"Missing IDs: {int(missing_ids.sum()):,}")
        print(f"Duplicate ID rows: {int(duplicate_ids.sum()):,}")
        print(f"Missing labels: {int(missing_labels.sum()):,}")
        print(f"Invalid binary labels: {int(invalid_labels.sum()):,}")
        print(f"Label IDs not found in index: {len(ids_not_in_index):,}")
        print(f"Index IDs without this label: {len(index_ids_without_labels):,}")
        print("Label counts:")
        print(labels.value_counts(dropna=False).sort_index().to_string())

        label_frames[task] = pd.DataFrame(
            {
                "exam_id": ids,
                f"{task}_label": labels,
            }
        )

        summary_rows.append(
            {
                "task": task,
                "rows": len(labels_df),
                "unique_ids": ids.nunique(dropna=True),
                "missing_ids": int(missing_ids.sum()),
                "duplicate_id_rows": int(duplicate_ids.sum()),
                "missing_labels": int(missing_labels.sum()),
                "invalid_binary_labels": int(invalid_labels.sum()),
                "label_ids_not_in_index": len(ids_not_in_index),
                "index_ids_without_labels": len(index_ids_without_labels),
            }
        )

        detail_frames.append(
            pd.DataFrame(
                {
                    "task": task,
                    "exam_id": ids,
                    "label": labels,
                    "id_in_index": ids.isin(index_ids),
                    "duplicate_id": duplicate_ids,
                    "missing_id": missing_ids,
                    "missing_label": missing_labels,
                    "invalid_label": invalid_labels,
                }
            )
        )

    # Combine label tables by examination ID, never by row position.
    merged = None

    for task, frame in label_frames.items():
        if frame["exam_id"].duplicated().any():
            raise ValueError(
                f"{task} contains duplicate examination IDs. "
                "Resolve duplicates before joining label tables."
            )

        merged = (
            frame.copy()
            if merged is None
            else merged.merge(
                frame,
                on="exam_id",
                how="outer",
                validate="one_to_one",
            )
        )

    assert merged is not None

    merged["id_in_index"] = merged["exam_id"].isin(index_ids)
    label_columns = ["abnormal_label", "acl_label", "meniscus_label"]
    merged["has_all_three_labels"] = merged[label_columns].notna().all(axis=1)

    print("\n--- CROSS-TASK ALIGNMENT ---")
    print(f"Unique IDs across label tables: {len(merged):,}")
    print(f"IDs with all three labels: {int(merged['has_all_three_labels'].sum()):,}")
    print(
        "IDs missing one or more labels: "
        f"{int((~merged['has_all_three_labels']).sum()):,}"
    )
    print(
        f"Label IDs absent from the MRI index: {int((~merged['id_in_index']).sum()):,}"
    )

    summary_df = pd.DataFrame(summary_rows)
    details_df = pd.concat(detail_frames, ignore_index=True)

    summary_df.to_csv(SUMMARY_FILE, index=False)
    details_df.to_csv(DETAIL_FILE, index=False)
    merged.to_csv(LABEL_MERGE_FILE, index=False)

    print("\nReports saved:")
    print(f"  {SUMMARY_FILE.relative_to(PROJECT_ROOT)}")
    print(f"  {DETAIL_FILE.relative_to(PROJECT_ROOT)}")
    print(f"  {LABEL_MERGE_FILE.relative_to(PROJECT_ROOT)}")
    print("\nValidation metadata audit completed.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        OSError,
        pd.errors.ParserError,
    ) as exc:
        print(f"\nAUDIT ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
