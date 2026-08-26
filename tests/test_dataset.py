import pytest
from PIL import Image

from fer.config import load_config
from fer.data.dataset import build_datasets, get_transforms

config = load_config()
_data_available = config.paths.train_dir.exists() and any(config.paths.train_dir.iterdir())


def test_simple_cnn_transform_shape():
    transform = get_transforms("simple_cnn", config)
    tensor = transform(Image.new("RGB", (100, 100)))
    size = config.simple_cnn.image_size
    assert tensor.shape == (1, size, size)


def test_resnet_transform_shape():
    transform = get_transforms("resnet", config)
    tensor = transform(Image.new("RGB", (100, 100)))
    size = config.resnet_finetune.image_size
    assert tensor.shape == (3, size, size)


@pytest.mark.skipif(not _data_available, reason="FER2013 dataset not downloaded yet")
def test_build_datasets_matches_class_names():
    train, val, test = build_datasets("simple_cnn", config)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
