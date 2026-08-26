"""Defines TrainConfig, the settings for a single Trainer run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    """Settings for one Trainer run, independent of model architecture.

    Attributes:
        epochs: Maximum number of training epochs.
        lr: Learning rate for the optimizer.
        weight_decay: Weight decay (L2 penalty) for the optimizer.
        early_stopping_patience: Epochs without validation-loss improvement
            allowed before training stops early.
        checkpoint_path: File path the best model checkpoint is saved to.
        num_classes: Number of output classes, used when computing metrics.
    """

    epochs: int
    lr: float
    weight_decay: float
    early_stopping_patience: int
    checkpoint_path: Path
    num_classes: int
