"""Module 4: adversarial attack implementations and their execution.

Not implemented yet - scaffolding only. Planned, per PLAN.md:
  - White-box evasion (FGSM, PGD): direct gradient access via `common`
    (build the victim model, load its weights from `artifacts.checkpoints`,
    compute perturbations against it directly).
  - Black-box evasion (transfer-based + Square Attack): query-only access
    via `fer_client`, treating the server as the only access point.
  - Backdoor/poisoning attack: retrains the victim architecture on
    triggered data via `common`'s Trainer/dataset-building, the same way a
    retraining-based defense would.
"""
