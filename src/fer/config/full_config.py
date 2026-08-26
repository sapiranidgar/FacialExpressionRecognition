"""Defines Config, the top-level object aggregating every configuration section."""

from __future__ import annotations

import dataclasses

from fer.config.data_config import DataConfig
from fer.config.paths_config import PathsConfig
from fer.config.resnet_finetune_config import ResnetFinetuneConfig
from fer.config.simple_cnn_config import SimpleCnnConfig


@dataclasses.dataclass
class Config:
    """Top-level configuration aggregating every configuration section.

    Instances are produced by fer.config.loader.load_config and passed around
    the codebase as the single source of truth for paths and hyperparameters.

    Attributes:
        seed: Global random seed for reproducibility.
        paths: Filesystem paths configuration.
        data: Dataset and dataloader configuration.
        simple_cnn: Hyperparameters for the from-scratch CNN.
        resnet_finetune: Hyperparameters for the fine-tuned ResNet18.
    """

    seed: int
    paths: PathsConfig
    data: DataConfig
    simple_cnn: SimpleCnnConfig
    resnet_finetune: ResnetFinetuneConfig
