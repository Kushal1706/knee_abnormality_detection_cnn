"""Build a validated examination-level manifest from MRNet metadata."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_ROOT / "data" / "bronze" / "source_metadata"
OUTPUT_DIR = PROJECT_ROOT / "data" / "gold" / "manifests"

INDEX_FILE = METADATA_DIR / "train.csv"

LABEL_FILES = {
    "abnormal": METADATA_DIR / "train_abnormal.csv",
    "acl": METADATA_DIR / "train_acl.csv",
    "meniscus": METADATA_DIR / "train_meniscus.csv",
}


def read_labels(task: str, path: Path) -> pd.DataFrame:
    """Load a label table and standardize its columns."""
    df = pd.read_csv(path, dtype=str)

    if "_0000" not in df.columns:
        raise ValueError(f"Missing examination ID column in {path.name}.")

    label_columns = [column for column in df.columns if column != "_0000"]

    if len(label_columns) != 1:
        raise ValueError(f"Unexpected label columns in {path.name}.")

    label_column = label_columns[0]

    result = pd.DataFrame(
        {
            "exam_id": df["_0000"].str.strip(),
            f"{task}_label": df[label_column].str.strip(),
        }
    )

    if result["exam_id"].isna().any() or result["exam_id"].eq("").any():
        raise ValueError(f"Missing examination IDs in {path.name}.")

    if result["exam_id"].duplicated().any():
        raise ValueError(f"Duplicate examination IDs in {path.name}.")

    if not result[f"{task}_label"].isin({"0", "1"}).all():
        raise ValueError(f"Invalid binary labels in {path.name}.")

    result[f"{task}_label"] = result[f"{task}_label"].astype("int8")

    return result


def main() -> None:
    """Create and validate the combined manifest."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    index_df = pd.read_csv(INDEX_FILE, dtype=str)

    required_columns = {"file_id", "file_name"}
    if not required_columns.issubset(index_df.columns):
        raise ValueError("The MRI index is missing required columns.")

    if index_df["file_id"].duplicated().any():
        raise ValueError("Duplicate Redivis file IDs found.")

    if index_df["file_name"].duplicated().any():
        raise ValueError("Duplicate MRI filenames found.")

    extracted = index_df["file_name"].str.extract(
        r"^(?P<view>[^/]+)/(?P<exam_id>\d+)\.npy$"
    )

    if extracted.isna().any(axis=None):
        raise ValueError("One or more MRI filenames have an unexpected format.")

    index_df = index_df.copy()
    index_df["view"] = extracted["view"]
    index_df["exam_id"] = extracted["exam_id"]

    allowed_views = {"axial", "coronal", "sagittal"}

    if set(index_df["view"]) != allowed_views:
        raise ValueError(f"Unexpected MRI views found: {sorted(set(index_df['view']))}")

    if index_df.duplicated(subset=["exam_id", "view"]).any():
        raise ValueError("Duplicate examination/view combinations found.")

    views = index_df.pivot(
        index="exam_id",
        columns="view",
        values="file_name",
    ).reset_index()

    for view in sorted(allowed_views):
        if view not in views.columns:
            raise ValueError(f"Missing view column: {view}")

        if views[view].isna().any():
            raise ValueError(f"Some examinations are missing the {view} view.")

    labels = None

    for task, path in LABEL_FILES.items():
        task_labels = read_labels(task, path)

        if labels is None:
            labels = task_labels
        else:
            labels = labels.merge(
                task_labels,
                on="exam_id",
                how="outer",
                validate="one_to_one",
            )

    if labels is None:
        raise ValueError("No label tables were loaded.")

    label_columns = ["abnormal_label", "acl_label", "meniscus_label"]

    if labels[label_columns].isna().any().any():
        raise ValueError("Some examinations are missing one or more labels.")

    if labels["exam_id"].duplicated().any():
        raise ValueError("Duplicate examination IDs in combined labels.")

    # Require every labelled examination to have all three MRI views.
    manifest = labels.merge(
        views,
        on="exam_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = manifest["_merge"] != "both"

    if unmatched.any():
        missing_ids = manifest.loc[unmatched, "exam_id"].tolist()
        raise ValueError(f"Labelled examinations without MRI files: {missing_ids[:10]}")

    manifest = manifest.drop(columns="_merge")

    column_order = [
        "exam_id",
        "axial",
        "coronal",
        "sagittal",
        "abnormal_label",
        "acl_label",
        "meniscus_label",
    ]

    manifest = manifest[column_order].sort_values("exam_id")

    output_path = OUTPUT_DIR / "train_examination_manifest.csv"
    manifest.to_csv(output_path, index=False)

    print("=" * 60)
    print("EXAMINATION MANIFEST CREATED")
    print("=" * 60)
    print(f"Labelled examinations: {len(manifest):,}")
    print(f"Output: {output_path.relative_to(PROJECT_ROOT)}")
    print("\nLabel distributions:")
    for column in label_columns:
        print(f"\n{column}:")
        print(manifest[column].value_counts().sort_index().to_string())

    print(
        "\nValidation passed: every manifest row has three MRI views and three labels."
    )


if __name__ == "__main__":
    main()
