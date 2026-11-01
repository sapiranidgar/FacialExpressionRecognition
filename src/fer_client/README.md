# fer_client

`fer_client` queries a running `fer_server` over HTTP. It has two parts:

- **`client.py`** — a thin library (`FerClient`) that sends an image and parses the prediction. It depends only on `requests` and knows nothing about models, PyTorch, or `common`, so anything that just wants to query the server (the web UI, a demo script, a black-box attack) can use it without the ML stack.
- **`web/app.py`** — a Streamlit website where a user uploads a face image and sees the predicted facial expression. It goes through `FerClient`; it never touches a model.

```
browser ──> Streamlit app (fer_client.web) ──FerClient──> fer_server /predict
```

## Package Structure

```
fer_client/
├── constants.py        # SERVER_URL, timeout, mirrored server API contract, UI text
├── client.py           # FerClient, Prediction, FerClientError
├── web/
│   └── app.py          # Streamlit website
└── requirements.txt    # requests, streamlit
```

## Library Usage

```python
from fer_client.client import FerClient, FerClientError

client = FerClient()                       # defaults to constants.SERVER_URL (http://localhost:8000)
if client.health():
    prediction = client.predict("face.png")  # path or raw bytes
    print(prediction.emotion, prediction.confidence, prediction.probabilities)
```

`predict` raises `FerClientError` if the server is unreachable/times out, or rejects the image (the message includes the server's `detail`, e.g. "Uploaded file could not be decoded as an image."). `health()` returns `False` instead of raising.

The server's route paths and response keys are mirrored in `constants.py` rather than imported from `fer_server`, keeping `fer_client` dependency-free of the server package.

## The Website

Upload (file picker or drag-and-drop) a **cropped face** image — the server does no face detection. As soon as it's chosen, the page shows a preview, the predicted expression with its confidence, and a bar chart of every expression's probability (highest first). If the server rejects the file or isn't running, an error message is shown inline.

## Configuration

Everything is a constant in `constants.py`: `SERVER_URL` (default `http://localhost:8000`) and `REQUEST_TIMEOUT_SECONDS`. `FerClient(base_url=..., timeout=...)` overrides them per instance. The website runs on Streamlit's default port, 8501.

## Execution Pipeline

From the repo root, start the server first, then the website:

```bash
pip install -r src/fer_server/requirements.txt -r src/fer_client/requirements.txt
PYTHONPATH=src uvicorn fer_server.main:app            # terminal 1, http://localhost:8000
PYTHONPATH=src streamlit run src/fer_client/web/app.py # terminal 2, http://localhost:8501
```

## Tests

`tests/fer_client/test_client.py` unit-tests `FerClient` with `requests.get`/`post` monkeypatched to canned responses (success parsing, 400 → `FerClientError` with detail, unreachable server). The website itself is checked manually.

## Where Results Are Saved

Not applicable — `fer_client` doesn't persist anything; whoever calls it decides what to do with the response.
