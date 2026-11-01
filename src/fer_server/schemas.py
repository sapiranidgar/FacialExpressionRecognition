"""Pydantic response models returned by the FER server's routes."""

from __future__ import annotations

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    """The model's prediction for one uploaded face image.

    Attributes:
        emotion: Name of the most probable emotion class.
        confidence: Softmax probability of `emotion`.
        probabilities: Softmax probability of every emotion class, keyed by
            class name. Exposed in full (not just the top label) so
            score-based black-box attacks can query the server.
    """

    emotion: str
    confidence: float
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    """Liveness check result.

    Attributes:
        status: Always "ok" when the server is up.
        model: Name of the model being served.
    """

    status: str
    model: str
