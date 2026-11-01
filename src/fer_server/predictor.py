"""Turns raw image bytes into an emotion prediction using the served model.

Deliberately free of any FastAPI code, so the same prediction logic can be
exercised directly (e.g. in tests) without going through HTTP.
"""

from __future__ import annotations

import io
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import transforms

from common.constants import DEVICE_CPU, FER_CLASS_NAMES
from common.data.dataset import get_transforms
from common.models.resnet_finetune import build_resnet18_finetune
from common.preprocessing.preprocessing_config import PreprocessingConfig

from fer_server.constants import (
    MODEL_CHECKPOINT_PATH,
    MODEL_IMAGE_SIZE,
    MODEL_TYPE,
    MSG_CHECKPOINT_MISSING_TEMPLATE,
    MSG_INVALID_IMAGE,
)
from fer_server.schemas import PredictionResponse

# get_transforms only reads augmentation settings when train=True, which
# inference never uses - so these values are inert placeholders.
_INFERENCE_PREPROCESSING = PreprocessingConfig(
    use_augmentation=False,
    horizontal_flip_probability=0.0,
    rotation_degrees=0.0,
    random_erasing_probability=0.0,
    use_class_weights=False,
    use_weighted_sampler=False,
    class_weight_power=0.0,
)


class InvalidImageError(ValueError):
    """Raised when uploaded bytes can't be decoded as an image."""


def build_inference_transform() -> transforms.Compose:
    """Build the eval-mode transform the served model was evaluated with (no augmentation).

    Returns:
        The same pipeline training used for its validation/test splits.
    """
    return get_transforms(MODEL_TYPE, MODEL_IMAGE_SIZE, train=False, preprocessing=_INFERENCE_PREPROCESSING)


def load_model(checkpoint_path: Path = MODEL_CHECKPOINT_PATH) -> nn.Module:
    """Build the served architecture and load its trained weights.

    Args:
        checkpoint_path: Path to the saved state_dict.

    Returns:
        The model in eval() mode with trained weights loaded.

    Raises:
        FileNotFoundError: If no checkpoint exists at checkpoint_path.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(MSG_CHECKPOINT_MISSING_TEMPLATE.format(path=checkpoint_path))

    # Skip the ImageNet download - the checkpoint overwrites every weight.
    model = build_resnet18_finetune(num_classes=len(FER_CLASS_NAMES), pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE_CPU, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


class Predictor:
    """Wraps a trained model and its preprocessing to classify single face images."""

    def __init__(self, model: nn.Module, transform: transforms.Compose, class_names: list[str]) -> None:
        """Store the model (switched to eval mode), its input transform, and class names.

        Args:
            model: Classifier producing one logit per class.
            transform: Pipeline mapping a PIL image to the model's input tensor.
            class_names: Class names in label-index order.
        """
        self.model = model.eval()
        self.transform = transform
        self.class_names = class_names

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path = MODEL_CHECKPOINT_PATH) -> Predictor:
        """Build a Predictor for the served model loaded from its saved checkpoint.

        Args:
            checkpoint_path: Path to the saved state_dict.

        Returns:
            A Predictor ready to serve requests.
        """
        return cls(load_model(checkpoint_path), build_inference_transform(), FER_CLASS_NAMES)

    def predict(self, image_bytes: bytes) -> PredictionResponse:
        """Classify the facial expression in one image.

        The image is assumed to already be a face crop (like FER2013's
        images); it's converted to grayscale and resized by the transform,
        with no face detection.

        Args:
            image_bytes: Encoded image in any format PIL can decode.

        Returns:
            The top emotion, its confidence, and every class's probability.

        Raises:
            InvalidImageError: If image_bytes can't be decoded as an image.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except (UnidentifiedImageError, OSError) as e:
            raise InvalidImageError(MSG_INVALID_IMAGE) from e

        batch = self.transform(image).unsqueeze(0)
        with torch.inference_mode():
            probs = torch.softmax(self.model(batch), dim=1).squeeze(0)

        top_idx = int(probs.argmax())
        return PredictionResponse(
            emotion=self.class_names[top_idx],
            confidence=float(probs[top_idx]),
            probabilities={name: float(p) for name, p in zip(self.class_names, probs)},
        )
