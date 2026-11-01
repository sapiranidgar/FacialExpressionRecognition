"""FastAPI entry point for the FER inference server.

Not implemented yet. Planned: a single prediction endpoint that accepts a
raw image upload, builds the victim model via `common.models` + loads its
trained weights from `artifacts.checkpoints`, preprocesses the image with
`common.data.dataset.get_transforms`, and returns predicted emotion
probabilities.

Run (once implemented) with the repo root as the working directory and
`src` on PYTHONPATH:
    PYTHONPATH=src uvicorn fer_server.main:app --reload
"""
