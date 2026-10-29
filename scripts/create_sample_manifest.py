"""Create a development manifest for the locally downloaded MRI sample."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_MANIFEST = (
    PROJECT_ROOT / "data" / "gold" / "manifests" / "train_examination_manifest.csv"
)

SAMPLE_DIRECTORY = PROJECT_ROOT / "data" / "bronze" / "mri_sample"
OUTPUT_MANIFEST = (
    PROJECT_ROOT / "data" / "bronze" / "manifests" / "sample_examination_manifest.csv"
)


def main() -> None:
    """Filter the original manifest to examinations with all three MRI views."""
    manifest = pd.read_csv(INPUT_MANIFEST)

    required_views = ["axial", "coronal", "sagittal"]

    sample_ids = set()

    for view in required_views:
        view_directory = SAMPLE_DIRECTORY / view

        if not view_directory.is_dir():
            raise FileNotFoundError(f"MRI view directory not found: {view_directory}")

        for path in view_directory.glob("*.npy"):
            if path.stem.isdigit():
                sample_ids.add(int(path.stem))

    # Keep only examinations that have files for all three views.
    complete_ids = set()

    for exam_id in sample_ids:
        if all(
            (SAMPLE_DIRECTORY / view / f"{exam_id:04d}.npy").is_file()
            for view in required_views
        ):
            complete_ids.add(exam_id)

    sample_manifest = manifest.loc[manifest["exam_id"].isin(complete_ids)].copy()

    sample_manifest = sample_manifest.sort_values("exam_id").reset_index(drop=True)

    if sample_manifest.empty:
        raise ValueError("No complete MRI examinations were found in the sample.")

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    sample_manifest.to_csv(OUTPUT_MANIFEST, index=False)

    print(f"Complete sample examinations: {len(sample_manifest)}")
    print(f"Saved sample manifest: {OUTPUT_MANIFEST}")
    print(sample_manifest[["exam_id", *required_views]].to_string(index=False))


if __name__ == "__main__":
    main()
