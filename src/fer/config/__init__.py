"""Typed configuration objects and the YAML loader for the fer package.

Re-exports every config dataclass plus load_config so callers can keep
writing `from fer.config import Config, load_config` regardless of which
file within this package actually defines them.
"""

from fer.config.data_config import DataConfig
from fer.config.full_config import Config
from fer.config.loader import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config
from fer.config.paths_config import PathsConfig
from fer.config.resnet_finetune_config import ResnetFinetuneConfig
from fer.config.simple_cnn_config import SimpleCnnConfig

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
