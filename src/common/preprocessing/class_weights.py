"""Computes per-class loss weights to counter FER2013's class imbalance."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from common.preprocessing.class_frequency import compute_inverse_frequency_weights
from common.preprocessing.labels import extract_labels


def compute_class_weights(dataset: Dataset, num_classes: int, power: float = 1.0) -> torch.Tensor:
    """Compute inverse-frequency class weights for a weighted CrossEntropyLoss.

    Pass the result as `nn.CrossEntropyLoss(weight=...)` so the loss
    penalizes missing a rare class as much as missing a common one, instead
    of the model learning it can mostly ignore rare classes and still score
    well on average.

    An alternative to (not normally combined with)
    common.preprocessing.class_balanced_sampler.build_class_balanced_sampler,
    which corrects the same imbalance in the DataLoader instead of the loss.

    Args:
        dataset: The training dataset (or Subset), yielding (image, label) pairs.
        num_classes: Total number of classes.
        power: Exponent softening the raw inverse-frequency ratio - see
            common.preprocessing.class_frequency.compute_inverse_frequency_weights.

    Returns:
        A float32 tensor of shape (num_classes,) suitable for
        nn.CrossEntropyLoss's `weight` argument.
    """
    labels = extract_labels(dataset)
    weights = compute_inverse_frequency_weights(labels, num_classes, power=power)
    return torch.tensor(weights, dtype=torch.float32)
