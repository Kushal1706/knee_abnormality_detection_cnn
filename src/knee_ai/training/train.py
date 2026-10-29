"""Training and validation utilities for the knee MRI CNN."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

VIEW_NAMES = ("axial", "coronal", "sagittal")
LABEL_NAMES = ("abnormality", "acl", "meniscus")


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds to improve experiment reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Extract MRI views and labels and move them to the selected device."""
    views = {
        name: batch[name].to(device=device, dtype=torch.float32) for name in VIEW_NAMES
    }
    labels = batch["labels"].to(device=device, dtype=torch.float32)

    return views, labels


def calculate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, dict[str, float | None]]:
    """Calculate per-label ROC-AUC where both classes are present."""
    results: dict[str, dict[str, float | None]] = {}

    for index, label_name in enumerate(LABEL_NAMES):
        true_values = targets[:, index]
        predicted_values = probabilities[:, index]

        if np.unique(true_values).size < 2:
            auc: float | None = None
        else:
            auc = float(roc_auc_score(true_values, predicted_values))

        results[label_name] = {"roc_auc": auc}

    return results


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Run one training or evaluation epoch."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_examinations = 0
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch in loader:
            views, labels = move_batch_to_device(batch, device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(**views)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_examinations += batch_size

            probabilities = torch.sigmoid(logits)

            all_targets.append(labels.detach().cpu().numpy())
            all_probabilities.append(probabilities.detach().cpu().numpy())

    if total_examinations == 0:
        raise ValueError("The DataLoader contains no examinations.")

    targets = np.concatenate(all_targets, axis=0)
    probabilities = np.concatenate(all_probabilities, axis=0)

    return {
        "loss": total_loss / total_examinations,
        "metrics": calculate_metrics(targets, probabilities),
        "targets": targets,
        "probabilities": probabilities,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    seed: int,
) -> None:
    """Save model and optimizer states plus experiment metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": validation_loss,
            "seed": seed,
        },
        path,
    )
