"""Typed configuration objects and the YAML loader for fer_training.

Re-exports every config dataclass plus load_config so callers can keep
writing `from fer_training.config import Config, load_config` regardless of
which file within this package actually defines them.
"""

from fer_training.config.data_config import DataConfig
from fer_training.config.full_config import Config
from fer_training.config.loader import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config
from fer_training.config.paths_config import PathsConfig
from fer_training.config.resnet_finetune_config import ResnetFinetuneConfig
from fer_training.config.simple_cnn_config import SimpleCnnConfig

__all__ = [
    "Config",
    "DataConfig",
    "PathsConfig",
    "ResnetFinetuneConfig",
    "SimpleCnnConfig",
    "load_config",
    "DEFAULT_CONFIG_PATH",
    "REPO_ROOT",
]
