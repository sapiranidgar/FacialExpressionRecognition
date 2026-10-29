"""Random horizontal flip augmentation."""

from __future__ import annotations

from torchvision import transforms


def build_horizontal_flip(probability: float) -> transforms.RandomHorizontalFlip:
    """Build a random horizontal flip transform.

    Flipping a face left-right essentially never changes its perceived
    emotion, so this is a safe, cheap way to double the effective variety of
    FER2013's small training set.

    Args:
        probability: Probability of flipping each image.

    Returns:
        A torchvision RandomHorizontalFlip transform, operating on a PIL image.
    """
    return transforms.RandomHorizontalFlip(p=probability)
