"""Random rotation augmentation."""

from __future__ import annotations

from torchvision import transforms


def build_random_rotation(degrees: float) -> transforms.RandomRotation:
    """Build a random rotation transform.

    Small rotations simulate head tilt and camera angle variation without
    distorting the expression itself, unlike more aggressive geometric
    augmentations.

    Args:
        degrees: Maximum rotation angle, in degrees, applied in either direction.

    Returns:
        A torchvision RandomRotation transform, operating on a PIL image.
    """
    return transforms.RandomRotation(degrees=degrees)
