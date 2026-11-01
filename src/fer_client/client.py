"""Thin HTTP client for the FER server's prediction endpoint.

Depends only on `requests` - no torch, no model-building, just HTTP calls -
so anything that wants to query a running server (the web UI, a demo
script, a black-box attack) can use it without the full ML stack.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import requests

from fer_client.constants import (
    MSG_SERVER_ERROR_TEMPLATE,
    MSG_SERVER_UNREACHABLE_TEMPLATE,
    REQUEST_TIMEOUT_SECONDS,
    RESPONSE_KEY_CONFIDENCE,
    RESPONSE_KEY_EMOTION,
    RESPONSE_KEY_ERROR_DETAIL,
    RESPONSE_KEY_PROBABILITIES,
    ROUTE_HEALTH,
    ROUTE_PREDICT,
    SERVER_URL,
    UPLOAD_FIELD_NAME,
    UPLOAD_FILE_NAME,
)


class FerClientError(Exception):
    """Raised when the server can't be reached or rejects a request."""


@dataclasses.dataclass(frozen=True)
class Prediction:
    """The server's prediction for one face image.

    Attributes:
        emotion: Name of the most probable emotion class.
        confidence: Probability of `emotion`.
        probabilities: Probability of every emotion class, keyed by class name.
    """

    emotion: str
    confidence: float
    probabilities: dict[str, float]


class FerClient:
    """Sends images to a running fer_server and parses its predictions."""

    def __init__(self, base_url: str = SERVER_URL, timeout: float = REQUEST_TIMEOUT_SECONDS) -> None:
        """Store where the server lives and how long to wait for it.

        Args:
            base_url: Server root URL, e.g. "http://localhost:8000".
            timeout: Seconds to wait for each request before giving up.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def predict(self, image: str | Path | bytes) -> Prediction:
        """Classify the facial expression in one image.

        Args:
            image: A path to an image file, or its encoded bytes. Should be
                a face crop - the server does no face detection.

        Returns:
            The server's prediction.

        Raises:
            FerClientError: If the server is unreachable or rejects the image.
        """
        image_bytes = image if isinstance(image, bytes) else Path(image).read_bytes()
        files = {UPLOAD_FIELD_NAME: (UPLOAD_FILE_NAME, image_bytes)}
        response = self._request(requests.post, ROUTE_PREDICT, files=files)
        body = response.json()
        return Prediction(
            emotion=body[RESPONSE_KEY_EMOTION],
            confidence=body[RESPONSE_KEY_CONFIDENCE],
            probabilities=body[RESPONSE_KEY_PROBABILITIES],
        )

    def health(self) -> bool:
        """Check whether the server is up.

        Returns:
            True if the server answered its health check, False otherwise.
        """
        try:
            self._request(requests.get, ROUTE_HEALTH)
        except FerClientError:
            return False
        return True

    def _request(self, send, route: str, **kwargs) -> requests.Response:
        """Send one request and translate transport/HTTP failures into FerClientError.

        Args:
            send: requests.get or requests.post.
            route: Server route path, e.g. "/predict".
            **kwargs: Extra arguments forwarded to `send`.

        Returns:
            The successful (2xx) response.

        Raises:
            FerClientError: On connection failure, timeout, or a non-2xx status.
        """
        try:
            response = send(f"{self.base_url}{route}", timeout=self.timeout, **kwargs)
        except requests.RequestException as e:
            raise FerClientError(MSG_SERVER_UNREACHABLE_TEMPLATE.format(url=self.base_url)) from e

        if not response.ok:
            raise FerClientError(
                MSG_SERVER_ERROR_TEMPLATE.format(status_code=response.status_code, detail=_error_detail(response))
            )
        return response


def _error_detail(response: requests.Response) -> str:
    """Extract FastAPI's "detail" message from an error response, falling back to the raw body."""
    try:
        return str(response.json()[RESPONSE_KEY_ERROR_DETAIL])
    except (ValueError, KeyError, TypeError):
        return response.text
