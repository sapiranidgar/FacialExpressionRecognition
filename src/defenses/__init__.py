"""Module 5: defenses against the attacks in `attacks`, and their execution.

Not implemented yet - scaffolding only. Planned, per PLAN.md:
  - Training-time defense (adversarial training): retrains the victim
    architecture on adversarial examples via `common`'s Trainer/dataset-
    building, producing a hardened checkpoint.
  - Inference-time defense (e.g. input preprocessing or an adversarial-
    input detector): wraps calls to the server via `fer_client`, the same
    way black-box attacks reach the server - defenses never modify the
    server's own code, they sit in front of it.
"""
