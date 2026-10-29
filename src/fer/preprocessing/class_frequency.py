"""Computes inverse-class-frequency weights shared by the two imbalance-handling techniques."""

from __future__ import annotations

import numpy as np


def compute_inverse_frequency_weights(labels: np.ndarray, num_classes: int, power: float = 1.0) -> np.ndarray:
    """Compute a per-class weight inversely proportional to how often it appears.

    A class with half as many samples as average gets roughly twice the raw
    weight (power=1.0), so rare classes (e.g. FER2013's "disgust", ~1.5% of
    the data) are not effectively ignored by whatever consumes these
    weights. Raw inverse-frequency weighting can overcorrect on a severely
    imbalanced dataset - it fixed FER2013's "disgust" recall but tanked its
    precision, since the loss started pushing the model to over-predict it.
    `power` softens this: 1.0 is raw inverse frequency, 0.5 takes the square
    root (a gentler boost), 0.0 disables weighting entirely (all weights 1).

    Args:
        labels: Integer class labels for every sample, e.g. from
            fer.preprocessing.labels.extract_labels.
        num_classes: Total number of classes.
        power: Exponent applied to the raw inverse-frequency ratio.

    Returns:
        A float array of shape (num_classes,). Classes absent from `labels`
        are treated as if they had one sample, to avoid division by zero.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    raw_weights = counts.sum() / (num_classes * counts)
    return raw_weights**power
