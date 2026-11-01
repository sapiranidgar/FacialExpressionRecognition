"""Defines DatasetConfig, the plain settings object for building datasets/dataloaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.preprocessing.preprocessing_config import PreprocessingConfig


@dataclass
class DatasetConfig:
    """Settings for building datasets and dataloaders, independent of any YAML schema.

    Callers (e.g. fer_training's scripts) construct this from whatever their
    own configuration system looks like; common.data.dataset itself never
    reads a config file directly.

    Attributes:
        image_size: Height and width (in pixels) images are resized to -
            model-specific, so the caller picks the right value per model type.
        train_dir: Directory holding the training split, one subfolder per class.
        test_dir: Directory holding the test split, one subfolder per class.
        class_names: Class names in label-index order (alphabetical, matching
            torchvision.datasets.ImageFolder's own sorting of subfolder names).
        val_fraction: Fraction of the training split carved out for validation.
        seed: Random seed for the train/val split, for reproducibility.
        batch_size: Batch size used for all dataloaders.
        num_workers: Number of worker processes used for data loading.
        num_classes: Number of emotion classes in the dataset.
        preprocessing: Train-time augmentation and class-imbalance settings.
    """

    image_size: int
    train_dir: Path
    test_dir: Path
    class_names: list[str]
    val_fraction: float
    seed: int
    batch_size: int
    num_workers: int
    num_classes: int
    preprocessing: PreprocessingConfig
