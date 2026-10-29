"""Builds a fresh, untrained model instance for one hyperparameter-search trial."""

from __future__ import annotations

import torch.nn as nn

from fer.config import Config
from fer.constants import MODEL_TYPE_RESNET, MODEL_TYPE_SIMPLE_CNN
from fer.data.dataset import ModelType
from fer.models.resnet_finetune import build_resnet18_finetune
from fer.models.simple_cnn import SimpleCNN


def build_model_for_tuning(model_type: ModelType, config: Config) -> nn.Module:
    """Instantiate a fresh, untrained model of the given architecture.

    Mirrors the model-building logic in scripts/train_cnn.py and
    scripts/train_resnet.py, duplicated here (rather than imported from
    scripts/) because scripts/*.py are standalone CLI entry points, not an
    importable package. Each grid-search trial needs its own freshly
    initialized model so one trial's learned weights never leak into the next.

    Args:
        model_type: Either "simple_cnn" or "resnet".
        config: Loaded project configuration.

    Returns:
        An untrained model instance ready to pass to Trainer.

    Raises:
        ValueError: If model_type is not recognized.
    """
    if model_type == MODEL_TYPE_SIMPLE_CNN:
        return SimpleCNN(num_classes=config.data.num_classes, in_channels=1)
    if model_type == MODEL_TYPE_RESNET:
        return build_resnet18_finetune(
            num_classes=config.data.num_classes,
            unfrozen_layers=config.resnet_finetune.unfrozen_layers,
        )
    raise ValueError(f"Unknown model_type: {model_type}")
