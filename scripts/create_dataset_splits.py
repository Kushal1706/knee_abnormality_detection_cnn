"""Generate and report examination-level dataset splits."""

from pathlib import Path

import pandas as pd

from knee_ai.data.dataset_splits import LABEL_COLUMNS, create_splits

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_MANIFEST = (
    PROJECT_ROOT / "data" / "gold" / "manifests" / "train_examination_manifest.csv"
)

OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "gold" / "splits"


def report_split_statistics(splits: pd.DataFrame) -> None:
    """Print split sizes and positive-label distributions."""
    print("\nExamination counts")
    print("------------------")

    counts = splits["split"].value_counts()

    for split_name in ["train", "validation", "test"]:
        count = int(counts.get(split_name, 0))
        percentage = 100 * count / len(splits)

        print(f"{split_name:12s}: {count:4d} examinations ({percentage:5.2f}%)")

    print("\nPositive-label counts and percentages")
    print("-------------------------------------")

    for split_name in ["train", "validation", "test"]:
        subset = splits[splits["split"] == split_name]

        print(f"\n{split_name.upper()} — {len(subset)} examinations")

        for label in LABEL_COLUMNS:
            positive_count = int(subset[label].sum())
            positive_percentage = 100 * subset[label].mean()

            print(
                f"  {label:18s}: "
                f"{positive_count:4d} positive "
                f"({positive_percentage:5.2f}%)"
            )


def main() -> None:
    """Generate the split manifest and print its statistics."""
    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(f"Examination manifest not found: {INPUT_MANIFEST}")

    manifest = pd.read_csv(INPUT_MANIFEST)

    print(f"Input manifest: {INPUT_MANIFEST}")
    print(f"Labelled examinations: {len(manifest)}")

    splits = create_splits(manifest)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIRECTORY / "examination_splits.csv"
    splits.to_csv(output_path, index=False)

    report_split_statistics(splits)

    print(f"\nSaved split manifest to:\n{output_path}")


if __name__ == "__main__":
    main()
