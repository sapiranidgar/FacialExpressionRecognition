"""Builds transform pipelines, datasets, and dataloaders for FER2013."""

from __future__ import annotations

from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from fer.config import Config
from fer.constants import (
    GRAYSCALE_NORM_MEAN,
    GRAYSCALE_NORM_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_TYPE_RESNET,
    MODEL_TYPE_SIMPLE_CNN,
)
from fer.preprocessing.augmentation_pipeline import build_pil_augmentations, build_tensor_augmentations
from fer.preprocessing.class_balanced_sampler import build_class_balanced_sampler
from fer.preprocessing.class_weights import compute_class_weights

ModelType = Literal[MODEL_TYPE_SIMPLE_CNN, MODEL_TYPE_RESNET]


def get_transforms(model_type: ModelType, config: Config, train: bool = False) -> transforms.Compose:
    """Build the image transform pipeline for the given model type.

    The simple CNN consumes single-channel grayscale images at its native
    resolution; the ResNet expects 3-channel images resized to its expected
    input size and normalized with ImageNet statistics. When `train` is
    True and config.preprocessing.use_augmentation is set, horizontal flip
    and rotation are inserted before ToTensor() and random erasing is
    appended after Normalize() - augmentation is never applied when `train`
    is False, so validation/test images always see the plain pipeline.

    Args:
        model_type: Either "simple_cnn" or "resnet".
        config: Loaded project configuration.
        train: Whether this pipeline is for the training split.

    Returns:
        A torchvision transform pipeline mapping a PIL image to a tensor.

    Raises:
        ValueError: If model_type is not recognized.
    """
    pil_augmentations = build_pil_augmentations(config.preprocessing) if train else []
    tensor_augmentations = build_tensor_augmentations(config.preprocessing) if train else []

    if model_type == MODEL_TYPE_SIMPLE_CNN:
        size = config.simple_cnn.image_size
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((size, size)),
                *pil_augmentations,
                transforms.ToTensor(),
                transforms.Normalize(mean=GRAYSCALE_NORM_MEAN, std=GRAYSCALE_NORM_STD),
                *tensor_augmentations,
            ]
        )
    if model_type == MODEL_TYPE_RESNET:
        size = config.resnet_finetune.image_size
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((size, size)),
                *pil_augmentations,
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                *tensor_augmentations,
            ]
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def build_datasets(model_type: ModelType, config: Config) -> tuple[Dataset, Dataset, Dataset]:
    """Build the train/val/test datasets for the given model type.

    The train split is loaded twice - once with the augmented pipeline, once
    with the plain one - and divided into train/val with matching indices
    (same seed, same underlying file order) according to
    config.data.val_fraction, so training images get augmented while
    validation images, drawn from the exact same files, don't.

    Args:
        model_type: Either "simple_cnn" or "resnet".
        config: Loaded project configuration.

    Returns:
        A (train, val, test) tuple of torch Datasets.

    Raises:
        ValueError: If the downloaded dataset's class folders don't match
            config.data.class_names.
    """
    train_transform = get_transforms(model_type, config, train=True)
    eval_transform = get_transforms(model_type, config, train=False)

    full_train_augmented = datasets.ImageFolder(str(config.paths.train_dir), transform=train_transform)
    full_train_plain = datasets.ImageFolder(str(config.paths.train_dir), transform=eval_transform)
    test = datasets.ImageFolder(str(config.paths.test_dir), transform=eval_transform)

    if full_train_augmented.classes != config.data.class_names:
        raise ValueError(
            f"Dataset classes {full_train_augmented.classes} do not match "
            f"config.data.class_names {config.data.class_names}. "
            "Update configs/config.yaml to match the downloaded dataset's folder names."
        )

    val_size = int(len(full_train_augmented) * config.data.val_fraction)
    train_size = len(full_train_augmented) - val_size
    generator = torch.Generator().manual_seed(config.seed)
    train, val_split = random_split(full_train_augmented, [train_size, val_size], generator=generator)
    val = Subset(full_train_plain, val_split.indices)

    return train, val, test


def build_dataloaders(
    model_type: ModelType, config: Config
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor | None]:
    """Build the train/val/test dataloaders for the given model type.

    If config.preprocessing.use_weighted_sampler is set, the training
    DataLoader draws samples via a class-balanced WeightedRandomSampler
    instead of plain shuffling. If config.preprocessing.use_class_weights is
    set, per-class loss weights are computed from the training split and
    returned for the caller to pass into Trainer.

    Args:
        model_type: Either "simple_cnn" or "resnet".
        config: Loaded project configuration.

    Returns:
        A (train_loader, val_loader, test_loader, class_weights) tuple.
        class_weights is None unless config.preprocessing.use_class_weights
        is set.
    """
    train, val, test = build_datasets(model_type, config)
    batch_size = config.data.batch_size
    num_workers = config.data.num_workers
    num_classes = config.data.num_classes

    weight_power = config.preprocessing.class_weight_power

    sampler = None
    shuffle = True
    if config.preprocessing.use_weighted_sampler:
        sampler = build_class_balanced_sampler(train, num_classes, power=weight_power)
        shuffle = False

    class_weights = None
    if config.preprocessing.use_class_weights:
        class_weights = compute_class_weights(train, num_classes, power=weight_power)

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers
    )
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_weights
