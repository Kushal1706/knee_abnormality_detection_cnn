"""Preprocessing utilities for knee MRI volumes."""

import numpy as np
import torch


def select_evenly_spaced_slices(
    volume: np.ndarray,
    num_slices: int = 8,
) -> np.ndarray:
    """Select evenly spaced slices from a 3D MRI volume."""
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D MRI volume; received shape {volume.shape}.")

    if volume.shape[0] == 0:
        raise ValueError("Cannot select slices from an empty volume.")

    if num_slices < 1:
        raise ValueError("num_slices must be at least 1.")

    indices = np.linspace(
        0,
        volume.shape[0] - 1,
        num=num_slices,
    )
    indices = np.rint(indices).astype(np.intp)

    return volume[indices]


def scale_mri_intensities(slices: np.ndarray) -> np.ndarray:
    """Scale uint8 MRI intensities to the range [0, 1]."""
    if slices.size == 0:
        raise ValueError("Cannot scale an empty array.")

    if not np.issubdtype(slices.dtype, np.number):
        raise ValueError("MRI intensities must be numeric.")

    if not np.isfinite(slices).all():
        raise ValueError("MRI intensities contain NaN or infinity.")

    if np.any(slices < 0) or np.any(slices > 255):
        raise ValueError("Expected MRI intensities in the range [0, 255].")

    return slices.astype(np.float32) / 255.0


def preprocess_mri_volume(
    volume: np.ndarray,
    num_slices: int = 8,
) -> torch.Tensor:
    """Convert an MRI volume into a normalized tensor of selected slices."""
    selected_slices = select_evenly_spaced_slices(
        volume,
        num_slices=num_slices,
    )

    scaled_slices = scale_mri_intensities(selected_slices)

    # Add a single channel dimension for grayscale MRI images.
    tensor = torch.from_numpy(scaled_slices).unsqueeze(1)

    return tensor
