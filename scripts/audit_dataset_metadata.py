"""Audit the MRNet index and examination-level label tables."""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_ROOT / "data" / "bronze" / "source_metadata"
REPORT_DIR = PROJECT_ROOT / "data" / "silver" / "quality_reports"

INDEX_FILE = METADATA_DIR / "train.csv"

LABEL_FILES = {
    "abnormal": METADATA_DIR / "train_abnormal.csv",
    "acl": METADATA_DIR / "train_acl.csv",
    "meniscus": METADATA_DIR / "train_meniscus.csv",
}


def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV values as strings to preserve leading zeros."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

    return pd.read_csv(path, dtype=str)


def load_labels(task: str, path: Path) -> pd.DataFrame:
    """Load and standardize one binary label table."""
    df = load_csv(path)

    if "_0000" not in df.columns:
        raise ValueError(
            f"{path.name} has no '_0000' examination ID column. "
            f"Columns found: {list(df.columns)}"
        )

    label_columns = [column for column in df.columns if column != "_0000"]

    if len(label_columns) != 1:
        raise ValueError(
            f"Expected one label column in {path.name}; found {label_columns}"
        )

    label_column = label_columns[0]

    result = pd.DataFrame(
        {
            "exam_id": df["_0000"].str.strip(),
            f"{task}_label": df[label_column].str.strip(),
        }
    )

    if result["exam_id"].isna().any() or result["exam_id"].eq("").any():
        raise ValueError(f"{path.name} contains missing examination IDs.")

    if result["exam_id"].duplicated().any():
        duplicates = result.loc[
            result["exam_id"].duplicated(keep=False), "exam_id"
        ].unique()
        raise ValueError(
            f"{path.name} contains duplicate examination IDs: "
            f"{duplicates[:10].tolist()}"
        )

    invalid = ~result[f"{task}_label"].isin({"0", "1"})

    if invalid.any():
        values = result.loc[invalid, f"{task}_label"].unique()
        raise ValueError(
            f"{path.name} contains missing or invalid binary labels: "
            f"{values[:10].tolist()}"
        )

    return result


def main() -> int:
    """Run checks and save detailed audit reports."""
    print("=" * 65)
    print("MRNet DATASET METADATA AUDIT")
    print("=" * 65)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    index_df = load_csv(INDEX_FILE)

    required_columns = {"file_id", "file_name"}
    missing_columns = required_columns - set(index_df.columns)

    if missing_columns:
        raise ValueError(
            f"Index is missing required columns: {sorted(missing_columns)}"
        )

    print(f"\nIndex rows: {len(index_df):,}")
    print(f"Columns: {list(index_df.columns)}")
    print(f"Unique Redivis file IDs: {index_df['file_id'].nunique():,}")
    print(f"Duplicate Redivis file IDs: {index_df['file_id'].duplicated().sum():,}")

    if index_df["file_id"].isna().any():
        raise ValueError("Index contains missing Redivis file IDs.")

    if index_df["file_name"].isna().any():
        raise ValueError("Index contains missing filenames.")

    # Extract view and examination ID from paths such as sagittal/0060.npy.
    path_parts = index_df["file_name"].str.extract(
        r"^(?P<view>[^/]+)/(?P<exam_id>\d+)\.npy$"
    )

    malformed = path_parts.isna().any(axis=1)

    if malformed.any():
        examples = index_df.loc[malformed, "file_name"].head(10).tolist()
        raise ValueError(f"Malformed MRI filenames detected: {examples}")

    index_df = index_df.copy()
    index_df["view"] = path_parts["view"]
    index_df["exam_id"] = path_parts["exam_id"]

    print("\nFiles by view:")
    print(index_df["view"].value_counts().sort_index().to_string())

    print(f"\nUnique examination IDs in file index: {index_df['exam_id'].nunique():,}")

    # There should be no more than one file for each examination/view pair.
    duplicate_view_pairs = index_df.duplicated(subset=["exam_id", "view"], keep=False)

    if duplicate_view_pairs.any():
        print("\nWARNING: Duplicate examination/view pairs detected:")
        print(
            index_df.loc[duplicate_view_pairs, ["exam_id", "view", "file_name"]]
            .sort_values(["exam_id", "view"])
            .head(20)
            .to_string(index=False)
        )
    else:
        print("Duplicate examination/view pairs: 0")

    # Create one row per examination with a separate filename for each view.
    view_table = index_df.pivot(
        index="exam_id",
        columns="view",
        values="file_name",
    ).reset_index()

    view_columns = [column for column in view_table.columns if column != "exam_id"]

    print("\nExaminations by available view combination:")
    combinations = (
        view_table[view_columns]
        .notna()
        .astype(int)
        .astype(str)
        .agg("".join, axis=1)
        .value_counts()
        .sort_index()
    )

    for combination, count in combinations.items():
        print(f"  Presence pattern {combination}: {count:,}")

    # Load labels and compare their IDs with filename-derived examination IDs.
    index_exam_ids = set(index_df["exam_id"])
    label_tables = {}
    summary_rows = []

    for task, path in LABEL_FILES.items():
        labels = load_labels(task, path)
        label_exam_ids = set(labels["exam_id"])

        label_only = sorted(label_exam_ids - index_exam_ids)
        index_only = sorted(index_exam_ids - label_exam_ids)

        print(f"\n--- {task.upper()} ---")
        print(f"Label rows: {len(labels):,}")
        print(f"Unique label IDs: {len(label_exam_ids):,}")
        print(f"Label IDs without MRI files: {len(label_only):,}")
        print(f"MRI examination IDs without labels: {len(index_only):,}")
        print(
            f"Label counts:\n"
            f"{labels[f'{task}_label'].value_counts().sort_index().to_string()}"
        )

        if label_only:
            print(f"Example label IDs without MRI files: {label_only[:10]}")

        if index_only:
            print(f"Example MRI IDs without labels: {index_only[:10]}")

        label_tables[task] = labels

        summary_rows.append(
            {
                "task": task,
                "label_rows": len(labels),
                "unique_label_ids": len(label_exam_ids),
                "label_ids_without_mri": len(label_only),
                "mri_ids_without_label": len(index_only),
            }
        )

    merged_labels = None

    for labels in label_tables.values():
        if merged_labels is None:
            merged_labels = labels.copy()
        else:
            merged_labels = merged_labels.merge(
                labels,
                on="exam_id",
                how="outer",
                validate="one_to_one",
            )

    assert merged_labels is not None

    print("\n--- CROSS-TASK LABEL CHECK ---")
    print(f"Unique IDs across label tables: {len(merged_labels):,}")
    print(f"IDs with all three labels: {merged_labels.notna().all(axis=1).sum():,}")
    print(f"IDs missing at least one label: {merged_labels.isna().any(axis=1).sum():,}")

    # Save a combined audit summary and an examination-level view manifest.
    summary_path = REPORT_DIR / "metadata_audit_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    manifest_path = REPORT_DIR / "examination_view_manifest.csv"
    view_table.to_csv(manifest_path, index=False)

    print(f"\nAudit summary: {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"View manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    print("\nAudit completed. Investigate discrepancies before training.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"\nAUDIT ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
