"""Composes the individual augmentation techniques into PIL-space and tensor-space groups.

Split this way because torchvision augmentations only work on one
representation each: RandomHorizontalFlip/RandomRotation expect a PIL image,
while RandomErasing expects a tensor. fer.data.dataset inserts the PIL-space
group after resizing (before ToTensor) and the tensor-space group after
normalization.
"""

from __future__ import annotations

from torchvision import transforms

from fer.config import PreprocessingConfig
from fer.preprocessing.horizontal_flip import build_horizontal_flip
from fer.preprocessing.random_erasing import build_random_erasing
from fer.preprocessing.random_rotation import build_random_rotation


def build_pil_augmentations(config: PreprocessingConfig) -> list[transforms.Transform]:
    """Build the augmentations that must run on a PIL image, before ToTensor().

    Args:
        config: The preprocessing section of the project configuration.

    Returns:
        A list of transforms to insert after resizing and before ToTensor().
        Empty if augmentation is disabled.
    """
    if not config.use_augmentation:
        return []
    return [
        build_horizontal_flip(config.horizontal_flip_probability),
        build_random_rotation(config.rotation_degrees),
    ]


def build_tensor_augmentations(config: PreprocessingConfig) -> list[transforms.Transform]:
    """Build the augmentations that must run on a tensor, after ToTensor()/Normalize().

    Args:
        config: The preprocessing section of the project configuration.

    Returns:
        A list of transforms to append at the end of the pipeline. Empty if
        augmentation is disabled.
    """
    if not config.use_augmentation:
        return []
    return [build_random_erasing(config.random_erasing_probability)]
