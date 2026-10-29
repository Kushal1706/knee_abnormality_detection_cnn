"""Tests for MRI preprocessing."""

import numpy as np
import pytest
import torch

from knee_ai.preprocessing.mri_preprocessor import (
    preprocess_mri_volume,
    scale_mri_intensities,
    select_evenly_spaced_slices,
)


def test_evenly_spaced_slice_selection() -> None:
    """Select the requested number of slices across a volume."""
    volume = np.zeros((30, 256, 256), dtype=np.uint8)

    selected = select_evenly_spaced_slices(volume, num_slices=8)

    assert selected.shape == (8, 256, 256)


def test_intensity_scaling() -> None:
    """Scale 8-bit image values into [0, 1]."""
    pixels = np.array([0, 128, 255], dtype=np.uint8)

    scaled = scale_mri_intensities(pixels)

    assert scaled.dtype == np.float32
    assert scaled[0] == 0.0
    assert scaled[1] == pytest.approx(128 / 255)
    assert scaled[2] == 1.0


def test_preprocessing_returns_tensor() -> None:
    """Return the expected tensor shape and numerical range."""
    volume = np.zeros((30, 256, 256), dtype=np.uint8)

    tensor = preprocess_mri_volume(volume, num_slices=8)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (8, 1, 256, 256)
    assert tensor.dtype == torch.float32
    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0


def test_preprocessing_rejects_invalid_volume_shape() -> None:
    """Reject an array that is not a 3D volume."""
    volume = np.zeros((256, 256), dtype=np.uint8)

    with pytest.raises(ValueError, match="3D MRI volume"):
        preprocess_mri_volume(volume)
