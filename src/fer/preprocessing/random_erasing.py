"""Random erasing (cutout-style) augmentation."""

from __future__ import annotations

from torchvision import transforms


def build_random_erasing(probability: float) -> transforms.RandomErasing:
    """Build a random erasing transform.

    Randomly blanks out a small rectangular patch of the image tensor,
    forcing the model to rely on more than one facial region rather than a
    single most-discriminative patch (e.g. only the mouth) - a cheap way to
    reduce overfitting on a small dataset.

    Args:
        probability: Probability of erasing a patch on each image.

    Returns:
        A torchvision RandomErasing transform. Operates on a tensor, so it
        must be placed after ToTensor() (and is typically placed after
        Normalize()) in the pipeline.
    """
    return transforms.RandomErasing(p=probability)
