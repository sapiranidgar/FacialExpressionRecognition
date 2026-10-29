"""Train-only data augmentation and class-imbalance handling.

Re-exports PreprocessingConfig and the augmentation-pipeline builders so
callers can write `from common.preprocessing import PreprocessingConfig`.
"""

from common.preprocessing.augmentation_pipeline import build_pil_augmentations, build_tensor_augmentations
from common.preprocessing.preprocessing_config import PreprocessingConfig

__all__ = [
    "PreprocessingConfig",
    "build_pil_augmentations",
    "build_tensor_augmentations",
]
