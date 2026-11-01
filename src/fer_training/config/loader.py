"""Loads configs/config.yaml from disk into a typed Config object."""

from __future__ import annotations

from pathlib import Path

import yaml
from artifacts.checkpoints import CHECKPOINTS_DIR
from artifacts.data import DATA_DIR
from artifacts.results import RESULTS_DIR
from common.preprocessing import PreprocessingConfig

from fer_training.config.data_config import DataConfig
from fer_training.config.full_config import Config
from fer_training.config.paths_config import PathsConfig
from fer_training.config.resnet_finetune_config import ResnetFinetuneConfig
from fer_training.config.simple_cnn_config import SimpleCnnConfig
from fer_training.constants import (
    CONFIG_FILE_RELATIVE_PATH,
    FER2013_SUBDIR_NAME,
    TEST_SUBDIR_NAME,
    TRAIN_SUBDIR_NAME,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / CONFIG_FILE_RELATIVE_PATH


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Load and parse the YAML config file into a typed Config object.

    Paths are derived from the `artifacts` package's shared location
    constants (not read from config.yaml) - artifacts.data.DATA_DIR,
    artifacts.checkpoints.CHECKPOINTS_DIR, artifacts.results.RESULTS_DIR -
    so there's one source of truth for where data/checkpoints/results live,
    shared with every other package in the repo.

    Args:
        path: Path to the YAML config file. Defaults to configs/config.yaml
            at the repo root.

    Returns:
        A populated Config instance.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    dataset_dir = DATA_DIR / FER2013_SUBDIR_NAME
    paths = PathsConfig(
        data_dir=dataset_dir,
        train_dir=dataset_dir / TRAIN_SUBDIR_NAME,
        test_dir=dataset_dir / TEST_SUBDIR_NAME,
        checkpoints_dir=CHECKPOINTS_DIR,
        results_dir=RESULTS_DIR,
    )

    return Config(
        seed=raw["seed"],
        paths=paths,
        data=DataConfig(**raw["data"]),
        simple_cnn=SimpleCnnConfig(**raw["simple_cnn"]),
        resnet_finetune=ResnetFinetuneConfig(**raw["resnet_finetune"]),
        preprocessing=PreprocessingConfig(**raw["preprocessing"]),
    )
