"""Multi-view knee MRI CNN with mean aggregation across slices."""

from __future__ import annotations

import torch
from torch import nn

from knee_ai.models.multiview_cnn import SliceFeatureExtractor


class AggregatedMultiViewKneeCNN(nn.Module):
    """Classify knee MRI examinations using mean-pooled slice features."""

    VIEW_NAMES = ("axial", "coronal", "sagittal")
    NUM_LABELS = 3
    FEATURES_PER_SLICE = 64
    IMAGE_SIZE = 256

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
        self.slice_encoder = SliceFeatureExtractor()

        # Mean aggregation produces 64 features per view.
        fused_features = len(self.VIEW_NAMES) * self.FEATURES_PER_SLICE

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

        batch_size = axial.shape[0]
        aggregated_views = []

        for view_name in self.VIEW_NAMES:
            volume = views[view_name]

            if volume.ndim != 5:
                raise ValueError(
                    f"{view_name} must have shape "
                    "(batch, slices, channels, height, width)."
                )

            if volume.shape[0] != batch_size:
                raise ValueError("All views must have the same batch size.")

            if volume.shape[1:] != (
                self.num_slices,
                1,
                self.IMAGE_SIZE,
                self.IMAGE_SIZE,
            ):
                raise ValueError(
                    f"{view_name} has invalid shape {tuple(volume.shape)}. "
                    f"Expected (batch, {self.num_slices}, 1, "
                    f"{self.IMAGE_SIZE}, {self.IMAGE_SIZE})."
                )

            # Combine batch and slice dimensions for shared CNN processing.
            flattened = volume.reshape(
                -1,
                volume.shape[2],
                volume.shape[3],
                volume.shape[4],
            )

            # One 64-feature vector per slice.
            features = self.slice_encoder(flattened)

            # Restore batch and slice dimensions.
            features = features.reshape(
                batch_size,
                self.num_slices,
                self.FEATURES_PER_SLICE,
            )

            # Average each feature across the sampled slices.
            view_features = features.mean(dim=1)
            aggregated_views.append(view_features)

        # 64 features × 3 views = 192 features per examination.
        fused = torch.cat(aggregated_views, dim=1)

        return self.classifier(fused)
