"""Writes the winning grid-search hyperparameters back into configs/config.yaml, preserving comments."""

from __future__ import annotations

from pathlib import Path

from common.data.dataset import ModelType
from ruamel.yaml import YAML

from fer_training.constants import CONFIG_KEY_LR, CONFIG_KEY_WEIGHT_DECAY, MODEL_TYPE_TO_CONFIG_SECTION


def write_best_hyperparameters(model_type: ModelType, lr: float, weight_decay: float, config_path: Path) -> None:
    """Patch one model's lr and weight_decay in config.yaml, preserving every comment.

    Uses ruamel.yaml's round-trip mode, which loads the file's full
    structure - including comments and key ordering - and only overwrites
    the two specific scalar values being tuned. Everything else in the
    file, including every explanatory comment, is left untouched.

    Args:
        model_type: Either "simple_cnn" or "resnet" - determines which
            top-level section of config.yaml is patched.
        lr: The winning learning rate to write.
        weight_decay: The winning weight decay to write.
        config_path: Path to the config.yaml file to update in place.

    Raises:
        KeyError: If model_type has no known config.yaml section.
    """
    section = MODEL_TYPE_TO_CONFIG_SECTION[model_type]

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path, "r") as f:
        data = yaml.load(f)

    data[section][CONFIG_KEY_LR] = lr
    data[section][CONFIG_KEY_WEIGHT_DECAY] = weight_decay

    with open(config_path, "w") as f:
        yaml.dump(data, f)
