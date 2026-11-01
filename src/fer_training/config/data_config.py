"""Defines DataConfig, the dataset/dataloader section of the project configuration."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class DataConfig:
    """Dataset and dataloader settings shared by every model.

    Attributes:
        num_classes: Number of emotion classes in the dataset.
        class_names: Class names in label-index order (alphabetical, matching
            torchvision.datasets.ImageFolder's own sorting of subfolder names).
        val_fraction: Fraction of the training split carved out for validation.
        batch_size: Batch size used for all dataloaders.
        num_workers: Number of worker processes used for data loading.
    """

    num_classes: int
    class_names: list[str]
    val_fraction: float
    batch_size: int
    num_workers: int
