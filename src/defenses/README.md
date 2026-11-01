# defenses

## Status

**Not implemented yet.** Scaffolding only — `__init__.py` and `requirements.txt`, no defense logic.

Defenses against the attacks in `attacks`, and the scripts to run them.

See [`PLAN.md`](../../PLAN.md) for the planned defense categories (a training-time defense such as adversarial training, retraining via `common`; and an inference-time defense that wraps calls to `fer_server` via `fer_client`, without modifying the server itself).

Everything else — package structure, code flow, execution, configuration, results — will be documented here once implemented.
