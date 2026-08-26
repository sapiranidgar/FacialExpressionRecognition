"""Defines PathsConfig, the filesystem-paths section of the project configuration."""

from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass
class PathsConfig:
    """Filesystem paths for the dataset, model checkpoints, and evaluation results.

    Attributes:
        data_dir: Root directory containing the downloaded dataset.
        train_dir: Directory holding the training split, one subfolder per class.
        test_dir: Directory holding the test split, one subfolder per class.
        checkpoints_dir: Directory where trained model weights are saved.
        results_dir: Directory where evaluation metrics and plots are saved.
    """

    data_dir: Path
    train_dir: Path
    test_dir: Path
    checkpoints_dir: Path
    results_dir: Path
