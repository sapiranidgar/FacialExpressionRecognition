"""FastAPI entry point for the FER inference server.

Loads the served model once at startup (failing fast if its checkpoint is
missing) and exposes a prediction route that classifies the facial
expression in an uploaded face image.

Run with the repo root as the working directory and `src` on PYTHONPATH:
    PYTHONPATH=src uvicorn fer_server.main:app --reload
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status

from fer_server.constants import (
    HEALTH_STATUS_OK,
    MODEL_NAME,
    ROUTE_HEALTH,
    ROUTE_PREDICT,
    UPLOAD_FIELD_NAME,
)
from fer_server.predictor import InvalidImageError, Predictor
from fer_server.schemas import HealthResponse, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the served model once before the server starts accepting requests."""
    app.state.predictor = Predictor.from_checkpoint()
    yield


app = FastAPI(title="FER Server", lifespan=lifespan)


def get_predictor(request: Request) -> Predictor:
    """Return the Predictor loaded at startup (overridden in tests with a stub model)."""
    return request.app.state.predictor


@app.post(ROUTE_PREDICT, response_model=PredictionResponse)
def predict(
    file: UploadFile = File(..., alias=UPLOAD_FIELD_NAME),
    predictor: Predictor = Depends(get_predictor),
) -> PredictionResponse:
    """Classify the facial expression in an uploaded face-crop image.

    A plain (non-async) def, so FastAPI runs the CPU-bound forward pass in
    its threadpool instead of blocking the event loop.
    """
    image_bytes = file.file.read()
    try:
        return predictor.predict(image_bytes)
    except InvalidImageError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@app.get(ROUTE_HEALTH, response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report that the server is up and which model it serves."""
    return HealthResponse(status=HEALTH_STATUS_OK, model=MODEL_NAME)
