# fer_server

`fer_server` is a FastAPI server that exposes the trained victim model over HTTP, so it can be queried like a real deployed model rather than called as a Python function. It is the thing `fer_client` talks to, and the thing black-box attacks/defenses in `attacks`/`defenses` are meant to query without ever touching the model directly.

## The Task

Accept a face image over HTTP, run it through the exact same preprocessing pipeline training used for evaluation, forward it through the victim model, and return the predicted emotion plus every class's probability. This gives the project a genuine query-only access point — critical for the black-box attack category (transfer-based + Square Attack), which simulates an attacker who can only observe inputs/outputs, not gradients. The full probability map is returned (not just the top label) because Square Attack is score-based.

## Package Structure

```
fer_server/
├── constants.py        # served model name/type, MODEL_CHECKPOINT_PATH, input size, routes, messages
├── schemas.py          # Pydantic PredictionResponse / HealthResponse
├── predictor.py        # Predictor: builds the model, loads the checkpoint, image bytes -> prediction (no FastAPI)
├── main.py             # FastAPI app: lifespan startup hook + routes
└── requirements.txt    # -r ../common/requirements.txt + fastapi, uvicorn, python-multipart, pillow
```

## Flow of the Code

1. **Startup** (`main.lifespan`): `Predictor.from_checkpoint()` builds ResNet18 via `common.models` (with `pretrained=False` — no ImageNet download, since the checkpoint overwrites every weight) and loads the state dict from the constant `MODEL_CHECKPOINT_PATH` (`artifacts/checkpoints/resnet18_finetuned.pt`). If the file is missing, startup fails immediately instead of on the first request.
2. **Each request** (`POST /predict`): the uploaded bytes are decoded with PIL, preprocessed with `common.data.dataset.get_transforms(train=False)` (grayscale → 3 channels, 224×224, ImageNet normalization — no augmentation), and run through the model under `torch.inference_mode()`. Softmax probabilities are returned.
3. `fer_server` imports from `common` and `artifacts` only — never from `fer_training` (no `config.yaml`), `fer_client`, `attacks`, or `defenses`. Class names come from `common.constants.FER_CLASS_NAMES`.

The input image is assumed to already be a **face crop** (like FER2013's images) — there is no face detection.

## API

### `POST /predict`

Multipart upload with the image in the `file` field (any format PIL can decode).

```bash
curl -F "file=@face.png" http://localhost:8000/predict
```

```json
{
  "emotion": "happy",
  "confidence": 0.991,
  "probabilities": {"angry": 0.001, "disgust": 0.0, "fear": 0.002, "happy": 0.991, "neutral": 0.004, "sad": 0.001, "surprise": 0.001}
}
```

Errors: `400` if the upload can't be decoded as an image; `422` if the `file` field is missing.

### `GET /health`

```json
{"status": "ok", "model": "resnet18_finetuned"}
```

Interactive docs are served at `/docs`.

## Configuration

Everything is a constant in `constants.py`. The served model is fixed to the clean-accuracy winner recorded in `artifacts/results/winner.json` (`resnet18_finetuned`); if a retrain changes the winner, update `MODEL_NAME`, `MODEL_TYPE`, `MODEL_CHECKPOINT_PATH`, and `MODEL_IMAGE_SIZE` together.

## Execution Pipeline

Train the model first (see `fer_training`) so the checkpoint exists, then run from the repo root with `src` on the Python path:

```bash
pip install -r src/fer_server/requirements.txt
PYTHONPATH=src uvicorn fer_server.main:app --reload
```

## Tests

`tests/fer_server/test_server.py` exercises the routes with FastAPI's `TestClient`, overriding the `get_predictor` dependency with a tiny random model so the tests run without the (gitignored) trained checkpoint.

## Where Results Are Saved

Not applicable — a running server doesn't produce files; it answers requests.
