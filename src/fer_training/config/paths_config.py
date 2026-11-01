"""Defines PathsConfig, the filesystem-paths section of the project configuration."""

from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass
class PathsConfig:
    """Filesystem paths for the dataset, model checkpoints, and evaluation results.

    These resolve to the shared locations defined by the `artifacts` package
    (artifacts.data.DATA_DIR, artifacts.checkpoints.CHECKPOINTS_DIR,
    artifacts.results.RESULTS_DIR) - see fer_training.config.loader - rather
    than being independently configurable, so there's a single source of
    truth for where other packages (fer_server, attacks, defenses) find the
    same data/checkpoints.

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
