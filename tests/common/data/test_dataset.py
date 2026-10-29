import pytest
from PIL import Image
from torchvision import transforms

from artifacts.data import DATA_DIR
from common.data.dataset import build_datasets, get_transforms
from common.data.dataset_config import DatasetConfig
from common.preprocessing import PreprocessingConfig

# Integration-test fixture only: common.data.dataset itself doesn't know or
# care what dataset is downloaded, but FER2013 (downloaded by fer_training)
# is the real data available in this repo to test build_datasets against.
_FER2013_TRAIN_DIR = DATA_DIR / "fer2013" / "train"
_FER2013_TEST_DIR = DATA_DIR / "fer2013" / "test"
_data_available = _FER2013_TRAIN_DIR.exists() and any(_FER2013_TRAIN_DIR.iterdir())

_CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def _preprocessing_config(use_augmentation: bool) -> PreprocessingConfig:
    return PreprocessingConfig(
        use_augmentation=use_augmentation,
        horizontal_flip_probability=0.5,
        rotation_degrees=15,
        random_erasing_probability=0.25,
        use_class_weights=True,
        use_weighted_sampler=False,
        class_weight_power=0.5,
    )


def _dataset_config(image_size: int, use_augmentation: bool = True) -> DatasetConfig:
    return DatasetConfig(
        image_size=image_size,
        train_dir=_FER2013_TRAIN_DIR,
        test_dir=_FER2013_TEST_DIR,
        class_names=_CLASS_NAMES,
        val_fraction=0.1,
        seed=42,
        batch_size=64,
        num_workers=0,
        num_classes=7,
        preprocessing=_preprocessing_config(use_augmentation),
    )


def test_simple_cnn_transform_shape():
    transform = get_transforms("simple_cnn", image_size=48, train=False, preprocessing=_preprocessing_config(True))
    tensor = transform(Image.new("RGB", (100, 100)))
    assert tensor.shape == (1, 48, 48)


def test_resnet_transform_shape():
    transform = get_transforms("resnet", image_size=224, train=False, preprocessing=_preprocessing_config(True))
    tensor = transform(Image.new("RGB", (100, 100)))
    assert tensor.shape == (3, 224, 224)


def test_train_transform_shape_matches_eval_transform_shape():
    train_transform = get_transforms(
        "simple_cnn", image_size=48, train=True, preprocessing=_preprocessing_config(True)
    )
    tensor = train_transform(Image.new("RGB", (100, 100)))
    assert tensor.shape == (1, 48, 48)


def test_train_transform_includes_augmentation_when_enabled():
    train_transform = get_transforms(
        "simple_cnn", image_size=48, train=True, preprocessing=_preprocessing_config(True)
    )
    step_types = [type(step) for step in train_transform.transforms]
    assert transforms.RandomHorizontalFlip in step_types
    assert transforms.RandomRotation in step_types
    assert transforms.RandomErasing in step_types


def test_eval_transform_never_includes_augmentation():
    eval_transform = get_transforms(
        "simple_cnn", image_size=48, train=False, preprocessing=_preprocessing_config(True)
    )
    step_types = [type(step) for step in eval_transform.transforms]
    assert transforms.RandomHorizontalFlip not in step_types
    assert transforms.RandomRotation not in step_types
    assert transforms.RandomErasing not in step_types


@pytest.mark.skipif(not _data_available, reason="FER2013 dataset not downloaded yet")
def test_build_datasets_matches_class_names():
    train, val, test = build_datasets("simple_cnn", _dataset_config(image_size=48))
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
