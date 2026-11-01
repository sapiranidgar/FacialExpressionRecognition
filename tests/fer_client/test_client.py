import json

import pytest
import requests

from fer_client import client as client_module
from fer_client.client import FerClient, FerClientError, Prediction
from fer_client.constants import ROUTE_HEALTH, ROUTE_PREDICT, UPLOAD_FIELD_NAME

BASE_URL = "http://fer.test"
PREDICTION_BODY = {
    "emotion": "happy",
    "confidence": 0.9,
    "probabilities": {"happy": 0.9, "sad": 0.1},
}


def make_response(status_code: int, json_body: dict) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(json_body).encode()
    return response


class FakeSend:
    """Stands in for requests.get/post: records the call and returns a canned response or raises."""

    def __init__(self, response: requests.Response | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, **kwargs) -> requests.Response:
        self.calls.append((url, kwargs))
        if self.error is not None:
            raise self.error
        return self.response


def test_predict_from_bytes_parses_prediction(monkeypatch):
    fake = FakeSend(make_response(200, PREDICTION_BODY))
    monkeypatch.setattr(client_module.requests, "post", fake)

    prediction = FerClient(BASE_URL).predict(b"image-bytes")

    assert prediction == Prediction(**PREDICTION_BODY)
    url, kwargs = fake.calls[0]
    assert url == f"{BASE_URL}{ROUTE_PREDICT}"
    assert kwargs["files"][UPLOAD_FIELD_NAME][1] == b"image-bytes"


def test_predict_from_path_sends_file_contents(monkeypatch, tmp_path):
    image_path = tmp_path / "face.png"
    image_path.write_bytes(b"file-bytes")
    fake = FakeSend(make_response(200, PREDICTION_BODY))
    monkeypatch.setattr(client_module.requests, "post", fake)

    FerClient(BASE_URL).predict(image_path)

    assert fake.calls[0][1]["files"][UPLOAD_FIELD_NAME][1] == b"file-bytes"


def test_predict_raises_with_server_detail_on_400(monkeypatch):
    detail = "Uploaded file could not be decoded as an image."
    monkeypatch.setattr(client_module.requests, "post", FakeSend(make_response(400, {"detail": detail})))

    with pytest.raises(FerClientError, match=detail):
        FerClient(BASE_URL).predict(b"garbage")


def test_predict_raises_when_server_unreachable(monkeypatch):
    fake = FakeSend(error=requests.ConnectionError("refused"))
    monkeypatch.setattr(client_module.requests, "post", fake)

    with pytest.raises(FerClientError, match="unreachable"):
        FerClient(BASE_URL).predict(b"image-bytes")


def test_health_true_when_server_answers(monkeypatch):
    fake = FakeSend(make_response(200, {"status": "ok", "model": "m"}))
    monkeypatch.setattr(client_module.requests, "get", fake)

    assert FerClient(BASE_URL).health() is True
    assert fake.calls[0][0] == f"{BASE_URL}{ROUTE_HEALTH}"


def test_health_false_when_server_unreachable(monkeypatch):
    monkeypatch.setattr(client_module.requests, "get", FakeSend(error=requests.ConnectionError("refused")))

    assert FerClient(BASE_URL).health() is False
