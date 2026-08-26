"""Builds transform pipelines, datasets, and dataloaders for FER2013."""

from __future__ import annotations

from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset, random_split
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

ModelType = Literal[MODEL_TYPE_SIMPLE_CNN, MODEL_TYPE_RESNET]


def get_transforms(model_type: ModelType, config: Config) -> transforms.Compose:
    """Build the image transform pipeline for the given model type.

    The simple CNN consumes single-channel grayscale images at its native
    resolution; the ResNet expects 3-channel images resized to its expected
    input size and normalized with ImageNet statistics.

    Args:
        model_type: Either "simple_cnn" or "resnet".
        config: Loaded project configuration.

    Returns:
        A torchvision transform pipeline mapping a PIL image to a tensor.

    Raises:
        ValueError: If model_type is not recognized.
    """
    if model_type == MODEL_TYPE_SIMPLE_CNN:
        size = config.simple_cnn.image_size
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=GRAYSCALE_NORM_MEAN, std=GRAYSCALE_NORM_STD),
            ]
        )
    if model_type == MODEL_TYPE_RESNET:
        size = config.resnet_finetune.image_size
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def build_datasets(model_type: ModelType, config: Config) -> tuple[Dataset, Dataset, Dataset]:
    """Build the train/val/test datasets for the given model type.

    The train split is loaded once and randomly divided into train/val
    according to config.data.val_fraction, using config.seed for
    reproducibility.

    Args:
        model_type: Either "simple_cnn" or "resnet".
        config: Loaded project configuration.

    Returns:
        A (train, val, test) tuple of torch Datasets.

    Raises:
        ValueError: If the downloaded dataset's class folders don't match
            config.data.class_names.
    """
    transform = get_transforms(model_type, config)

    full_train = datasets.ImageFolder(str(config.paths.train_dir), transform=transform)
    test = datasets.ImageFolder(str(config.paths.test_dir), transform=transform)

    if full_train.classes != config.data.class_names:
        raise ValueError(
            f"Dataset classes {full_train.classes} do not match "
            f"config.data.class_names {config.data.class_names}. "
            "Update configs/config.yaml to match the downloaded dataset's folder names."
        )

    val_size = int(len(full_train) * config.data.val_fraction)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(config.seed)
    train, val = random_split(full_train, [train_size, val_size], generator=generator)

    return train, val, test


def build_dataloaders(model_type: ModelType, config: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build the train/val/test dataloaders for the given model type.

    Args:
        model_type: Either "simple_cnn" or "resnet".
        config: Loaded project configuration.

    Returns:
        A (train_loader, val_loader, test_loader) tuple of torch DataLoaders.
    """
    train, val, test = build_datasets(model_type, config)
    batch_size = config.data.batch_size
    num_workers = config.data.num_workers

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
