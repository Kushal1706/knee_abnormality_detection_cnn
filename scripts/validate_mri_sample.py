"""Validate the downloaded MRI sample."""

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_DIR = PROJECT_ROOT / "data" / "bronze" / "mri_sample"

VIEWS = ("axial", "coronal", "sagittal")


def main() -> None:
    """Load and inspect each downloaded MRI array."""
    print("=" * 60)
    print("MRI SAMPLE VALIDATION")
    print("=" * 60)

    for view in VIEWS:
        file_path = SAMPLE_DIR / view / "0001.npy"

        print(f"\nView: {view}")
        print(f"File: {file_path.relative_to(PROJECT_ROOT)}")

        if not file_path.is_file():
            print("ERROR: File does not exist.")
            continue
        print(f"File size: {file_path.stat().st_size:,}bytes")

        try:
            array = np.load(file_path, allow_pickle=False)
        except (OSError, ValueError, EOFError) as exc:
            print(f"Error: Could not load Numpy array: {exc}")
            continue

        print(f"Shape: {array.shape}")
        print(f"Dimensions: {array.ndim}")
        print(f"Data type: {array.dtype}")
        print(f"Minimum intensity: {array.min()}")
        print(f"Maximum intensity: {array.max()}")
        print(f"Mean intensity: {array.mean():.2f}")
        print(f"Standard deviation: {array.std():.2f}")
        print(f"All values finite: {np.isfinite(array).all()}")
        print(f"Array memory: {array.nbytes:,} bytes")

        if array.ndim != 3:
            print("WARNING: Expected a three-dimensional MRI volume.")

        if array.size == 0:
            print("WARNING: The array is empty.")

    print("\nValidation inspection completed.")


if __name__ == "__main__":
    main()
