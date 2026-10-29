"""Tests for the multi-view knee MRI CNN."""

import pytest
import torch

from knee_ai.models.multiview_cnn import MultiViewKneeCNN


def make_inputs(batch_size: int = 2, num_slices: int = 8):
    """Create synthetic inputs with the expected MRI tensor shapes."""
    shape = (batch_size, num_slices, 1, 256, 256)

    return {
        "axial": torch.rand(shape),
        "coronal": torch.rand(shape),
        "sagittal": torch.rand(shape),
    }


def test_model_returns_three_logits_per_examination():
    model = MultiViewKneeCNN(num_slices=8)
    inputs = make_inputs()

    outputs = model(**inputs)

    assert outputs.shape == (2, 3)


def test_model_supports_batch_size_one():
    model = MultiViewKneeCNN(num_slices=8)
    inputs = make_inputs(batch_size=1)

    outputs = model(**inputs)

    assert outputs.shape == (1, 3)


def test_model_supports_a_different_slice_count():
    model = MultiViewKneeCNN(num_slices=4)
    inputs = make_inputs(batch_size=1, num_slices=4)

    outputs = model(**inputs)

    assert outputs.shape == (1, 3)


def test_model_rejects_incorrect_slice_count():
    model = MultiViewKneeCNN(num_slices=8)
    inputs = make_inputs(num_slices=4)

    with pytest.raises(ValueError, match="must contain 8 slices"):
        model(**inputs)


def test_model_backpropagates_gradients():
    model = MultiViewKneeCNN(num_slices=8)
    inputs = make_inputs()
    targets = torch.rand(2, 3)

    outputs = model(**inputs)
    loss = torch.nn.BCEWithLogitsLoss()(outputs, targets)
    loss.backward()

    assert model.classifier[-1].weight.grad is not None
    assert torch.isfinite(model.classifier[-1].weight.grad).all()
