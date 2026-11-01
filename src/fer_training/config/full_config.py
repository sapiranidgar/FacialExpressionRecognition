"""Defines Config, the top-level object aggregating every configuration section."""

from __future__ import annotations

import dataclasses

from common.preprocessing import PreprocessingConfig

from fer_training.config.data_config import DataConfig
from fer_training.config.paths_config import PathsConfig
from fer_training.config.resnet_finetune_config import ResnetFinetuneConfig
from fer_training.config.simple_cnn_config import SimpleCnnConfig


@dataclasses.dataclass
class Config:
    """Top-level configuration aggregating every configuration section.

    Instances are produced by fer_training.config.loader.load_config and
    passed around fer_training's code as the single source of truth for
    paths and hyperparameters.

    Attributes:
        seed: Global random seed for reproducibility.
        paths: Filesystem paths configuration.
        data: Dataset and dataloader configuration.
        simple_cnn: Hyperparameters for the from-scratch CNN.
        resnet_finetune: Hyperparameters for the fine-tuned ResNet18.
        preprocessing: Train-time augmentation and class-imbalance settings
            (common's plain settings dataclass - fer_training just loads it
            from YAML, common's own code never reads config.yaml itself).
    """

    seed: int
    paths: PathsConfig
    data: DataConfig
    simple_cnn: SimpleCnnConfig
    resnet_finetune: ResnetFinetuneConfig
    preprocessing: PreprocessingConfig
