"""Default hyperparameter grids searched per model architecture."""

from __future__ import annotations

import itertools

from fer.constants import MODEL_TYPE_RESNET, MODEL_TYPE_SIMPLE_CNN
from fer.data.dataset import ModelType

DEFAULT_GRIDS: dict[str, dict[str, list[float]]] = {
    MODEL_TYPE_SIMPLE_CNN: {
        "lr": [0.0005, 0.001, 0.002],
        "weight_decay": [1e-5, 1e-4, 1e-3],
    },
    MODEL_TYPE_RESNET: {
        "lr": [0.0001, 0.0005, 0.001],
        "weight_decay": [1e-5, 1e-4, 1e-3],
    },
}


def get_default_grid(model_type: ModelType) -> dict[str, list[float]]:
    """Look up the default search grid for one model architecture.

    Each model's grid is centered on its current config.yaml value for that
    hyperparameter, since SimpleCNN (trained from scratch) and the
    ResNet18 fine-tune (adapting pretrained weights) operate at different
    learning-rate scales.

    Args:
        model_type: Either "simple_cnn" or "resnet".

    Returns:
        A dict with "lr" and "weight_decay" keys, each a list of values to try.

    Raises:
        KeyError: If model_type has no default grid defined.
    """
    return DEFAULT_GRIDS[model_type]


def build_param_combinations(grid: dict[str, list[float]]) -> list[tuple[float, float]]:
    """Expand a grid into every (lr, weight_decay) combination.

    Args:
        grid: A dict with "lr" and "weight_decay" keys, each a list of values.

    Returns:
        A list of (lr, weight_decay) tuples - the Cartesian product of the
        grid's two dimensions.
    """
    return list(itertools.product(grid["lr"], grid["weight_decay"]))
