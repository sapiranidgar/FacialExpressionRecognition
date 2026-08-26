"""Builds a ResNet18 classifier fine-tuned from ImageNet-pretrained weights."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

from fer.constants import RESNET_DEFAULT_UNFROZEN_LAYERS


def build_resnet18_finetune(num_classes: int = 7, unfrozen_layers: list[str] | None = None) -> nn.Module:
    """Load ImageNet-pretrained ResNet18 and freeze all but the given layers.

    The classifier head (`fc`) is always replaced to match `num_classes`, and
    is trainable regardless of `unfrozen_layers` since it must be relearned
    for this task's label space.

    Args:
        num_classes: Number of output emotion classes.
        unfrozen_layers: Names of top-level submodules (e.g. "layer4", "fc")
            left trainable; everything else is frozen. Defaults to
            RESNET_DEFAULT_UNFROZEN_LAYERS.

    Returns:
        The modified ResNet18 model, ready for fine-tuning.
    """
    if unfrozen_layers is None:
        unfrozen_layers = RESNET_DEFAULT_UNFROZEN_LAYERS

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(layer) for layer in unfrozen_layers)

    return model
