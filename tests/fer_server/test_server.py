import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch import nn

from common.constants import FER_CLASS_NAMES
from fer_server.constants import (
    HEALTH_STATUS_OK,
    MODEL_IMAGE_SIZE,
    MODEL_NAME,
    ROUTE_HEALTH,
    ROUTE_PREDICT,
    UPLOAD_FIELD_NAME,
)
from fer_server.main import app, get_predictor
from fer_server.predictor import Predictor, build_inference_transform


def make_stub_predictor() -> Predictor:
    # Tiny random model standing in for the (gitignored) trained checkpoint.
    torch.manual_seed(0)
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, len(FER_CLASS_NAMES)))
    return Predictor(model, build_inference_transform(), FER_CLASS_NAMES)


@pytest.fixture
def client():
    app.dependency_overrides[get_predictor] = make_stub_predictor
    # Not used as a context manager, so the lifespan hook (which loads the
    # real checkpoint) never runs.
    yield TestClient(app)
    app.dependency_overrides.clear()


def encode_image(mode: str, size: tuple[int, int], fmt: str) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color=128 if mode == "L" else (128, 64, 32)).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("mode", "size", "fmt"),
    [("L", (48, 48), "PNG"), ("RGB", (300, 200), "JPEG")],
)
def test_predict_returns_structured_answer(client, mode, size, fmt):
    files = {UPLOAD_FIELD_NAME: (f"face.{fmt.lower()}", encode_image(mode, size, fmt))}
    response = client.post(ROUTE_PREDICT, files=files)

    assert response.status_code == 200
    body = response.json()
    assert set(body["probabilities"]) == set(FER_CLASS_NAMES)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert body["emotion"] == max(body["probabilities"], key=body["probabilities"].get)
    assert body["confidence"] == pytest.approx(body["probabilities"][body["emotion"]])


def test_predict_rejects_non_image_bytes(client):
    files = {UPLOAD_FIELD_NAME: ("face.png", b"definitely not an image")}
    response = client.post(ROUTE_PREDICT, files=files)
    assert response.status_code == 400


def test_predict_requires_file_field(client):
    response = client.post(ROUTE_PREDICT)
    assert response.status_code == 422


def test_health_reports_served_model(client):
    response = client.get(ROUTE_HEALTH)
    assert response.status_code == 200
    assert response.json() == {"status": HEALTH_STATUS_OK, "model": MODEL_NAME}


def test_inference_transform_matches_model_input():
    image = Image.new("L", (48, 48))
    assert build_inference_transform()(image).shape == (3, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
