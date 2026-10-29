import numpy as np
import pytest
import torch
from torch.utils.data import Subset
from torchvision import transforms

from fer.config import PreprocessingConfig
from fer.preprocessing.augmentation_pipeline import build_pil_augmentations, build_tensor_augmentations
from fer.preprocessing.class_balanced_sampler import build_class_balanced_sampler
from fer.preprocessing.class_frequency import compute_inverse_frequency_weights
from fer.preprocessing.class_weights import compute_class_weights
from fer.preprocessing.labels import extract_labels


class _FakeImageFolder:
    """Minimal stand-in for torchvision.datasets.ImageFolder exposing only .targets."""

    def __init__(self, targets: list[int]):
        self.targets = targets


def _enabled_preprocessing_config() -> PreprocessingConfig:
    return PreprocessingConfig(
        use_augmentation=True,
        horizontal_flip_probability=0.5,
        rotation_degrees=15,
        random_erasing_probability=0.25,
        use_class_weights=True,
        use_weighted_sampler=False,
        class_weight_power=0.5,
    )


def _disabled_preprocessing_config() -> PreprocessingConfig:
    return PreprocessingConfig(
        use_augmentation=False,
        horizontal_flip_probability=0.5,
        rotation_degrees=15,
        random_erasing_probability=0.25,
        use_class_weights=False,
        use_weighted_sampler=False,
        class_weight_power=0.5,
    )


def test_extract_labels_from_plain_dataset():
    fake = _FakeImageFolder([0, 1, 2, 0, 1])
    labels = extract_labels(fake)
    assert labels.tolist() == [0, 1, 2, 0, 1]


def test_extract_labels_from_subset_reindexes_correctly():
    fake = _FakeImageFolder([0, 1, 2, 0, 1])
    subset = Subset(fake, [4, 0, 2])
    labels = extract_labels(subset)
    assert labels.tolist() == [1, 0, 2]


def test_compute_inverse_frequency_weights_balanced_classes_get_equal_weight():
    labels = np.array([0, 1, 0, 1, 0, 1])
    weights = compute_inverse_frequency_weights(labels, num_classes=2)
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(1.0)


def test_compute_inverse_frequency_weights_power_zero_disables_weighting():
    labels = np.array([0] * 9 + [1])
    weights = compute_inverse_frequency_weights(labels, num_classes=2, power=0.0)
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(1.0)


def test_compute_inverse_frequency_weights_power_softens_the_boost():
    labels = np.array([0] * 9 + [1])
    raw_weights = compute_inverse_frequency_weights(labels, num_classes=2, power=1.0)
    softened_weights = compute_inverse_frequency_weights(labels, num_classes=2, power=0.5)
    assert softened_weights[1] == pytest.approx(raw_weights[1] ** 0.5)
    # softening should narrow the gap between the rare and common class weight
    assert (softened_weights[1] - softened_weights[0]) < (raw_weights[1] - raw_weights[0])


def test_compute_inverse_frequency_weights_rare_class_gets_higher_weight():
    # class 0 appears 9x, class 1 appears 1x
    labels = np.array([0] * 9 + [1])
    weights = compute_inverse_frequency_weights(labels, num_classes=2)
    assert weights[1] > weights[0]


def test_compute_class_weights_returns_float_tensor_of_correct_shape():
    fake = _FakeImageFolder([0, 0, 0, 1, 2])
    weights = compute_class_weights(fake, num_classes=3)
    assert isinstance(weights, torch.Tensor)
    assert weights.dtype == torch.float32
    assert weights.shape == (3,)
    # class 0 is most common -> lowest weight
    assert weights[0] < weights[1]
    assert weights[0] < weights[2]


def test_build_class_balanced_sampler_weights_rare_class_samples_higher():
    fake = _FakeImageFolder([0] * 9 + [1])
    sampler = build_class_balanced_sampler(fake, num_classes=2)
    assert sampler.num_samples == 10
    weights = list(sampler.weights)
    # the single class-1 sample (last one) should have a higher weight than any class-0 sample
    assert weights[-1] > weights[0]


def test_build_pil_augmentations_empty_when_disabled():
    assert build_pil_augmentations(_disabled_preprocessing_config()) == []


def test_build_tensor_augmentations_empty_when_disabled():
    assert build_tensor_augmentations(_disabled_preprocessing_config()) == []


def test_build_pil_augmentations_returns_flip_and_rotation_when_enabled():
    steps = build_pil_augmentations(_enabled_preprocessing_config())
    step_types = [type(s) for s in steps]
    assert transforms.RandomHorizontalFlip in step_types
    assert transforms.RandomRotation in step_types


def test_build_tensor_augmentations_returns_erasing_when_enabled():
    steps = build_tensor_augmentations(_enabled_preprocessing_config())
    step_types = [type(s) for s in steps]
    assert transforms.RandomErasing in step_types
