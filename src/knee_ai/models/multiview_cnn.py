"""Compact multi-view CNN for knee MRI multi-label classification."""

from __future__ import annotations

import torch
from torch import nn


class SliceFeatureExtractor(nn.Module):
    """Extract visual features from batches of grayscale MRI slices."""

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Input: (batch, 1, 256, 256)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # (batch, 16, 128, 128)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # (batch, 32, 64, 64)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # (batch, 64, 32, 32)
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, slices: torch.Tensor) -> torch.Tensor:
        """Return one feature vector per MRI slice."""
        return self.features(slices)


class MultiViewKneeCNN(nn.Module):
    """Classify knee MRI examinations using three anatomical views."""

    VIEW_NAMES = ("axial", "coronal", "sagittal")
    NUM_LABELS = 3

    def __init__(
        self,
        num_slices: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if num_slices < 1:
            raise ValueError("num_slices must be at least 1.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.num_slices = num_slices

        # The same extractor processes all three views.
        self.slice_encoder = SliceFeatureExtractor()

        # Each view contributes num_slices * 64 features.
        fused_features = len(self.VIEW_NAMES) * num_slices * 64

        self.classifier = nn.Sequential(
            nn.Linear(fused_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, self.NUM_LABELS),
        )

    def forward(
        self,
        axial: torch.Tensor,
        coronal: torch.Tensor,
        sagittal: torch.Tensor,
    ) -> torch.Tensor:
        """Return three classification logits per examination."""
        views = {
            "axial": axial,
            "coronal": coronal,
            "sagittal": sagittal,
        }

        batch_size: int | None = None
        encoded_views: list[torch.Tensor] = []

        for view_name in self.VIEW_NAMES:
            volume = views[view_name]

            if volume.ndim != 5:
                raise ValueError(
                    f"{view_name} must have shape "
                    "(batch, slices, channels, height, width)."
                )

            if volume.shape[1] != self.num_slices:
                raise ValueError(f"{view_name} must contain {self.num_slices} slices.")

            if volume.shape[2] != 1:
                raise ValueError(f"{view_name} must contain grayscale slices.")

            if volume.shape[3] != 256 or volume.shape[4] != 256:
                raise ValueError(f"{view_name} slices must be 256 x 256.")

            if batch_size is None:
                batch_size = volume.shape[0]
            elif volume.shape[0] != batch_size:
                raise ValueError("All views must have the same batch size.")

            # Combine batch and slice dimensions to encode slices together.
            flattened = volume.reshape(
                -1,
                volume.shape[2],
                volume.shape[3],
                volume.shape[4],
            )

            features = self.slice_encoder(flattened)

            # Restore the examination and slice dimensions.
            features = features.reshape(
                volume.shape[0],
                self.num_slices,
                -1,
            )

            encoded_views.append(features.flatten(start_dim=1))

        fused = torch.cat(encoded_views, dim=1)

        return self.classifier(fused)
