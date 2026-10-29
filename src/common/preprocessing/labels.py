"""Extracts integer labels from an ImageFolder-backed dataset without decoding images."""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset, Subset


def extract_labels(dataset: Dataset) -> np.ndarray:
    """Read every sample's integer label directly from dataset metadata.

    Works with a torchvision ImageFolder or a Subset of one (e.g. the
    train/val split produced by common.data.dataset.build_datasets), without
    calling __getitem__ - which would decode and transform every image just
    to read its label.

    Args:
        dataset: An ImageFolder, or a Subset wrapping one.

    Returns:
        An array of integer class labels, one per sample in `dataset`, in
        the same order the dataset yields them.
    """
    if isinstance(dataset, Subset):
        base_labels = np.array(dataset.dataset.targets)
        return base_labels[dataset.indices]
    return np.array(dataset.targets)
