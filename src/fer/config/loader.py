"""Loads configs/config.yaml from disk into a typed Config object."""

from __future__ import annotations

from pathlib import Path

import yaml

from fer.config.data_config import DataConfig
from fer.config.full_config import Config
from fer.config.paths_config import PathsConfig
from fer.config.resnet_finetune_config import ResnetFinetuneConfig
from fer.config.simple_cnn_config import SimpleCnnConfig
from fer.constants import CONFIG_FILE_RELATIVE_PATH

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / CONFIG_FILE_RELATIVE_PATH


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Load and parse the YAML config file into a typed Config object.

    Relative paths inside the "paths" section are resolved against the repo
    root (the directory containing configs/, src/, scripts/), regardless of
    the current working directory the caller runs from.

    Args:
        path: Path to the YAML config file. Defaults to configs/config.yaml
            at the repo root.

    Returns:
        A populated Config instance.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    paths_raw = raw["paths"]
    paths = PathsConfig(
        data_dir=REPO_ROOT / paths_raw["data_dir"],
        train_dir=REPO_ROOT / paths_raw["train_dir"],
        test_dir=REPO_ROOT / paths_raw["test_dir"],
        checkpoints_dir=REPO_ROOT / paths_raw["checkpoints_dir"],
        results_dir=REPO_ROOT / paths_raw["results_dir"],
    )

    return Config(
        seed=raw["seed"],
        paths=paths,
        data=DataConfig(**raw["data"]),
        simple_cnn=SimpleCnnConfig(**raw["simple_cnn"]),
        resnet_finetune=ResnetFinetuneConfig(**raw["resnet_finetune"]),
    )
