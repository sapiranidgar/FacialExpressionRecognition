"""Defines PreprocessingConfig, the data-preprocessing section of the project configuration."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class PreprocessingConfig:
    """Settings controlling train-time augmentation and class-imbalance handling.

    Augmentation is applied only to the training split - never to
    validation or test - since it exists to prevent the model from
    memorizing the exact training images, not to change what it's evaluated
    against.

    use_class_weights and use_weighted_sampler are two alternative ways to
    counter FER2013's class imbalance (its "disgust" class has roughly 16x
    fewer samples than "happy"). They are not usually combined, since doing
    so double-corrects for the same imbalance.

    Attributes:
        use_augmentation: Whether to apply horizontal flip, rotation, and
            random erasing to training images.
        horizontal_flip_probability: Probability of flipping each training image.
        rotation_degrees: Maximum random rotation angle (degrees), in either direction.
        random_erasing_probability: Probability of erasing a random patch of
            each training image (applied after normalization).
        use_class_weights: Whether to weight the training loss by inverse
            class frequency.
        use_weighted_sampler: Whether to oversample rare classes in the
            training DataLoader instead.
        class_weight_power: Exponent softening the raw inverse-frequency
            ratio used by both use_class_weights and use_weighted_sampler.
            1.0 is raw inverse frequency (can overcorrect - boosting a rare
            class's recall at the cost of its precision, since the model
            starts over-predicting it); 0.5 takes the square root for a
            gentler boost; 0.0 disables weighting entirely.
    """

    use_augmentation: bool
    horizontal_flip_probability: float
    rotation_degrees: float
    random_erasing_probability: float
    use_class_weights: bool
    use_weighted_sampler: bool
    class_weight_power: float
