"""Builds a WeightedRandomSampler that oversamples FER2013's rare classes."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from fer.preprocessing.class_frequency import compute_inverse_frequency_weights
from fer.preprocessing.labels import extract_labels


def build_class_balanced_sampler(dataset: Dataset, num_classes: int, power: float = 1.0) -> WeightedRandomSampler:
    """Build a sampler that oversamples rare classes so each batch is more balanced.

    An alternative to (not normally combined with)
    fer.preprocessing.class_weights.compute_class_weights: this corrects
    imbalance in the DataLoader itself rather than the loss function, so
    every batch already contains a more even mix of classes.

    Pass the result as the DataLoader's `sampler` argument, and set
    `shuffle=False` - a DataLoader can't take both a sampler and shuffle=True.

    Args:
        dataset: The training dataset (or Subset), yielding (image, label) pairs.
        num_classes: Total number of classes.
        power: Exponent softening the raw inverse-frequency ratio - see
            fer.preprocessing.class_frequency.compute_inverse_frequency_weights.

    Returns:
        A WeightedRandomSampler with one weight per sample (rare classes
        weighted higher) and `num_samples` equal to the dataset size, drawn
        with replacement.
    """
    labels = extract_labels(dataset)
    class_weights = compute_inverse_frequency_weights(labels, num_classes, power=power)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )
