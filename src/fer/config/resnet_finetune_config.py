"""Defines ResnetFinetuneConfig, the hyperparameter section for fine-tuning ResNet18."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class ResnetFinetuneConfig:
    """Hyperparameters for fine-tuning the pretrained ResNet18.

    Attributes:
        image_size: Height and width (in pixels) input images are resized to.
        epochs: Maximum number of training epochs.
        lr: Learning rate for the Adam optimizer.
        weight_decay: Weight decay (L2 penalty) for the Adam optimizer.
        early_stopping_patience: Epochs without validation-loss improvement
            allowed before training stops early.
        unfrozen_layers: Names of the top-level ResNet18 submodules left
            trainable; every other parameter is frozen.
        checkpoint_name: Filename used to save the best model checkpoint.
    """

    image_size: int
    epochs: int
    lr: float
    weight_decay: float
    early_stopping_patience: int
    unfrozen_layers: list[str]
    checkpoint_name: str
