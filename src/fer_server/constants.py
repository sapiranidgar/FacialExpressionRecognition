"""Constants specific to fer_server: which model to serve, its input size, and route paths.

Constants shared with other packages (class names, model-type identifiers,
normalization stats) live in common.constants instead - this module is
strictly for the server's own wiring.
"""

from __future__ import annotations

from artifacts.checkpoints import CHECKPOINTS_DIR
from common.constants import MODEL_TYPE_RESNET

# --- Served model: the clean-accuracy winner recorded in artifacts/results/winner.json ---
MODEL_NAME = "resnet18_finetuned"
MODEL_TYPE = MODEL_TYPE_RESNET
MODEL_CHECKPOINT_PATH = CHECKPOINTS_DIR / "resnet18_finetuned.pt"
MODEL_IMAGE_SIZE = 224

# --- Routes ---
ROUTE_PREDICT = "/predict"
ROUTE_HEALTH = "/health"
UPLOAD_FIELD_NAME = "file"

# --- Responses ---
HEALTH_STATUS_OK = "ok"
MSG_CHECKPOINT_MISSING_TEMPLATE = "Model checkpoint not found at {path}. Train the model first (see fer_training)."
MSG_INVALID_IMAGE = "Uploaded file could not be decoded as an image."
