# fer_client

## Status

**Not implemented yet.** This is scaffolding only — `__init__.py`, a stub `client.py`, and `requirements.txt`. No client logic exists. This README documents the agreed design so the eventual implementation has somewhere to start from; see [`PLAN.md`](../../PLAN.md) for the full repo-reorganization decisions.

`fer_client` is a thin HTTP client for `fer_server`'s prediction endpoint. It deliberately knows nothing about models, PyTorch, or `common` — it only knows how to send an image and parse the response. That's what keeps it usable by anything that just wants to query a running server (a grader, a demo script, a black-box attack) without installing the full ML stack.

## The Task (Planned)

Send an image (bytes or a file path) to `fer_server`'s prediction route over HTTP via `requests`, and return the parsed prediction. Nothing more — no preprocessing, no model logic, no retries/caching beyond what's strictly needed. All of that genuinely belongs to the server, not the client.

## Package Structure

```
fer_client/
├── client.py            # the HTTP call (currently a docstring stub)
└── requirements.txt     # just `requests`
```

## Planned Flow of the Code

1. Caller provides an image (path or bytes) and the server's base URL.
2. `client.py` sends it as a file upload to the server's prediction route.
3. The parsed JSON response (predicted emotion probabilities) is returned to the caller.

`fer_client` has no dependency on `common`, `fer_training`, or `artifacts` — by design, so it never needs `torch`/`torchvision` installed. It's used *by* `attacks` (black-box evasion attacks query the server only, never the model directly) and `defenses` (inference-time defenses sit in front of the server and forward through this client), not the other way around.

## Configuration

Not yet defined. Likely candidates once implemented: the server's base URL (default `http://localhost:8000` or similar), request timeout.

## Execution Pipeline (Planned)

`fer_client` is a library, not a CLI entry point — it's meant to be imported by whatever script needs to query the server (a demo script, an attack, a defense), which will itself provide the execution entry point once those are implemented.

## Where Results Are Saved

Not applicable — `fer_client` doesn't persist anything; whoever calls it decides what to do with the response.
