"""Defines SimpleCnnConfig, the hyperparameter section for the from-scratch CNN."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class SimpleCnnConfig:
    """Hyperparameters for training the from-scratch simple CNN.

    Attributes:
        image_size: Height and width (in pixels) input images are resized to.
        grayscale: Whether input images are single-channel grayscale.
        epochs: Maximum number of training epochs.
        lr: Initial learning rate for the Adam optimizer.
        weight_decay: Weight decay (L2 penalty) for the Adam optimizer.
        early_stopping_patience: Epochs without validation-loss improvement
            allowed before training stops early.
        lr_scheduler_patience: Epochs without validation-loss improvement
            allowed before the learning rate is reduced.
        lr_scheduler_factor: Multiplier applied to the learning rate each
            time it's reduced.
        checkpoint_name: Filename used to save the best model checkpoint.
    """

    image_size: int
    grayscale: bool
    epochs: int
    lr: float
    weight_decay: float
    early_stopping_patience: int
    lr_scheduler_patience: int
    lr_scheduler_factor: float
    checkpoint_name: str
