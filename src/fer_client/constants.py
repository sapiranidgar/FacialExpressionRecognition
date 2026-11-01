"""Constants for fer_client: where the server lives, its API contract, and UI text.

fer_client deliberately doesn't import fer_server or common, so the server's
route paths and response keys are mirrored here rather than imported.
"""

from __future__ import annotations

# --- Server location ---
SERVER_URL = "http://localhost:8000"
REQUEST_TIMEOUT_SECONDS = 30.0

# --- Server API contract (mirrors fer_server.constants / fer_server.schemas) ---
ROUTE_PREDICT = "/predict"
ROUTE_HEALTH = "/health"
UPLOAD_FIELD_NAME = "file"
UPLOAD_FILE_NAME = "image"
RESPONSE_KEY_EMOTION = "emotion"
RESPONSE_KEY_CONFIDENCE = "confidence"
RESPONSE_KEY_PROBABILITIES = "probabilities"
RESPONSE_KEY_ERROR_DETAIL = "detail"

# --- Client error messages ---
MSG_SERVER_UNREACHABLE_TEMPLATE = "FER server unreachable at {url}. Is it running?"
MSG_SERVER_ERROR_TEMPLATE = "FER server returned HTTP {status_code}: {detail}"

# --- Web UI (fer_client.web.app) ---
PAGE_TITLE = "Facial Expression Recognition"
PAGE_ICON = "🙂"
UPLOAD_LABEL = "Upload a face image"
UPLOAD_HELP = "Upload a cropped face (like FER2013's images) - the server does no face detection."
UPLOAD_ACCEPTED_TYPES = ["png", "jpg", "jpeg", "bmp", "gif", "webp"]
PREVIEW_CAPTION = "Uploaded image"
SPINNER_TEXT = "Asking the FER server..."
RESULT_METRIC_LABEL = "Predicted expression"
RESULT_CONFIDENCE_TEMPLATE = "{confidence:.1%} confidence"
CHART_TITLE = "Probability per expression"
CHART_COLUMN_EMOTION = "expression"
CHART_COLUMN_PROBABILITY = "probability"
