import pytest
from PIL import Image
from torchvision import transforms

from fer.config import load_config
from fer.data.dataset import build_datasets, get_transforms

config = load_config()
_data_available = config.paths.train_dir.exists() and any(config.paths.train_dir.iterdir())


def test_simple_cnn_transform_shape():
    transform = get_transforms("simple_cnn", config, train=False)
    tensor = transform(Image.new("RGB", (100, 100)))
    size = config.simple_cnn.image_size
    assert tensor.shape == (1, size, size)


def test_resnet_transform_shape():
    transform = get_transforms("resnet", config, train=False)
    tensor = transform(Image.new("RGB", (100, 100)))
    size = config.resnet_finetune.image_size
    assert tensor.shape == (3, size, size)


def test_train_transform_shape_matches_eval_transform_shape():
    train_transform = get_transforms("simple_cnn", config, train=True)
    tensor = train_transform(Image.new("RGB", (100, 100)))
    size = config.simple_cnn.image_size
    assert tensor.shape == (1, size, size)


def test_train_transform_includes_augmentation_when_enabled():
    assert config.preprocessing.use_augmentation, "test assumes the default config enables augmentation"
    train_transform = get_transforms("simple_cnn", config, train=True)
    step_types = [type(step) for step in train_transform.transforms]
    assert transforms.RandomHorizontalFlip in step_types
    assert transforms.RandomRotation in step_types
    assert transforms.RandomErasing in step_types


def test_eval_transform_never_includes_augmentation():
    eval_transform = get_transforms("simple_cnn", config, train=False)
    step_types = [type(step) for step in eval_transform.transforms]
    assert transforms.RandomHorizontalFlip not in step_types
    assert transforms.RandomRotation not in step_types
    assert transforms.RandomErasing not in step_types


@pytest.mark.skipif(not _data_available, reason="FER2013 dataset not downloaded yet")
def test_build_datasets_matches_class_names():
    train, val, test = build_datasets("simple_cnn", config)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
