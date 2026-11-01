# fer_server

## Status

**Not implemented yet.** This is scaffolding only — `__init__.py`, a stub `main.py`, and `requirements.txt`. No server logic exists. This README documents the agreed design so the eventual implementation has somewhere to start from; see [`PLAN.md`](../../PLAN.md) for the full repo-reorganization decisions.

`fer_server` is a FastAPI server that exposes the trained victim model over HTTP, so it can be queried like a real deployed model rather than called as a Python function. It is the thing `fer_client` talks to, and the thing black-box attacks/defenses in `attacks`/`defenses` are meant to query without ever touching the model directly.

## The Task (Planned)

Accept an image over HTTP, run it through the exact same preprocessing pipeline training used, forward it through the victim model (whichever architecture `artifacts/results/winner.json` names as the winner), and return the predicted emotion probabilities. This gives the project a genuine query-only access point — critical for the black-box attack category (transfer-based + Square Attack), which is supposed to simulate an attacker who can only observe inputs/outputs, not gradients.

## Package Structure

```
fer_server/
├── main.py             # FastAPI app entry point (currently a docstring stub)
└── requirements.txt    # -r ../common/requirements.txt + fastapi, uvicorn, python-multipart, pillow
```

## Planned Flow of the Code

1. On startup: read `artifacts/results/winner.json` to find the victim's architecture name, build that model via `common.models`, and load its weights from `artifacts.checkpoints.CHECKPOINTS_DIR`.
2. On each request: accept a raw image upload, preprocess it with `common.data.dataset.get_transforms` (the same eval-mode pipeline training's validation/test splits use — no augmentation), run it through the loaded model, and return class probabilities.
3. `fer_server` imports from `common` only — never from `fer_training` (no `config.yaml`, no `Trainer`) and never from `fer_client`/`attacks`/`defenses`. It is purely a consumer of `common` + `artifacts`.

## Configuration

Not yet defined. Likely candidates once implemented: host/port, which checkpoint to serve (default: read from `winner.json`, override via env var or CLI flag).

## Execution Pipeline (Planned)

Once implemented, run from the repo root with `src` on the Python path:

```bash
PYTHONPATH=src uvicorn fer_server.main:app --reload
```

## Where Results Are Saved

Not applicable — a running server doesn't produce files; it answers requests.
