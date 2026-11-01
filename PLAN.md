# Facial Expression Recognition + Adversarial Attacks — Project Plan

## Context

This is a class assignment with two phases: (1) build a facial expression
classifier, (2) attack it. No GPU is available (CPU-only local training).
Design decisions below were reached through an extended requirements
discussion before any code was written.

- **Compute**: CPU-only, local.
- **Framework**: PyTorch (richest adversarial-attack tooling for phase 2).
- **Dataset**: FER2013 (35,887 grayscale 48x48 images, 7 emotions), pulled
  from Kaggle (`msambare/fer2013`, folder-per-class layout — plugs directly
  into `torchvision.datasets.ImageFolder`). Chosen over RAF-DB/AffectNet/CK+
  because it needs no access-request/approval wait.
- **Models**: build **both**, compare clean accuracy, and attack whichever
  wins:
  1. A simple CNN trained from scratch on 48x48 grayscale.
  2. A ResNet18 (ImageNet-pretrained) fine-tuned — images upsampled to
     224x224 and channel-replicated to 3ch; freeze most of the backbone,
     fine-tune `layer4` + the classifier head only (forced by CPU budget).
  - The **losing** model becomes the substitute for the transfer-based
    black-box attack later — no third model needed.
- **Attacks** (phase 2, not implemented yet) — one implementation per
  category, all via the Adversarial Robustness Toolbox (ART):
  - White-box evasion: **FGSM** and **PGD**, both, direct gradient access
    to the victim.
  - Black-box evasion: **both** transfer-based (craft on the substitute,
    test transfer to the victim) and query-based (**Square Attack**,
    no gradients at all against the victim).
  - Poisoning: **backdoor trigger attack** (BadNets-style) — small fixed
    patch stamped on a subset of training images, relabeled to a target
    class ("happy" by default), victim architecture retrained on the
    poisoned set, then evaluated for backdoor success.
- Hyperparameters (poisoning rate, epsilon budgets, trigger size/position,
  query budgets, frozen-layer split) are implementation defaults, not
  architectural decisions — set in `configs/config.yaml`.

## Project Structure

```
FacialExpressionRecognition/
├── README.md
├── PLAN.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml                 # paths + all hyperparameters, single source of truth
├── data/
│   └── fer2013/                    # downloaded dataset (gitignored)
│       ├── train/<emotion>/*.jpg
│       └── test/<emotion>/*.jpg
├── src/
│   └── fer/
│       ├── __init__.py
│       ├── config.py                # load/validate config.yaml into a dataclass
│       ├── data/
│       │   ├── __init__.py
│       │   ├── download.py          # Kaggle API download + extraction
│       │   └── dataset.py           # FER2013 Dataset(s) + transforms; poisoning hook (stubbed for now)
│       ├── models/
│       │   ├── __init__.py
│       │   ├── simple_cnn.py        # from-scratch CNN
│       │   └── resnet_finetune.py   # ResNet18 wrapper, layer freezing
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py           # shared train/val loop, checkpointing, early stop
│       │   └── metrics.py           # accuracy / F1 / confusion matrix helpers
│       ├── attacks/                 # NOT implemented yet (phase 2)
│       └── utils/
│           ├── __init__.py
│           ├── seed.py              # global reproducibility
│           └── visualize.py         # confusion matrices, training curves
├── scripts/                         # thin CLI entry points, no logic of their own
│   ├── download_dataset.py
│   ├── train_cnn.py
│   ├── train_resnet.py
│   └── evaluate_clean.py            # compares both models, writes results/winner.json
├── checkpoints/                     # saved weights (gitignored)
├── results/                         # metrics json + plots (gitignored)
│   └── clean_eval/
└── tests/
    ├── test_dataset.py              # shapes
    └── test_models.py               # forward-pass shapes for both architectures
```

Attack-related files (`src/fer/attacks/`, `scripts/run_*_attack.py`,
`results/whitebox_eval/`, `results/blackbox_eval/`, `results/backdoor_eval/`)
are deliberately **not** part of this build pass — phase 2, later.

## Implementation Steps (this pass — model phase only)

**1. Scaffolding**
- `requirements.txt`: `torch`, `torchvision` (CPU wheels), `numpy`, `pandas`,
  `scikit-learn`, `matplotlib`, `kaggle`, `pyyaml`, `tqdm`, `pytest`.
  (`adversarial-robustness-toolbox` added later, in the attacks pass.)
- `configs/config.yaml`: data paths, batch size, per-model LR/epochs,
  frozen-layer spec for ResNet18. Attack-related keys added later.
- `src/fer/config.py` loads this into a typed object used everywhere else.

**2. Dataset**
- `src/fer/data/download.py`: wraps the Kaggle CLI/API to pull
  `msambare/fer2013` into `data/fer2013/`. Requires the user to have a
  Kaggle account + API token in `~/.kaggle/kaggle.json` — the script checks
  for it and prints setup instructions if missing.
- `src/fer/data/dataset.py`: two transform pipelines (48x48 grayscale
  tensor for the CNN; resize-224 + 3-channel replicate + ImageNet
  normalization for ResNet18), built on `torchvision.datasets.ImageFolder`.

**3. Models**
- `simple_cnn.py`: ~4 conv blocks (BatchNorm + ReLU + MaxPool, channels
  32->64->128->256), global average pool, dropout, FC to 7 classes.
- `resnet_finetune.py`: loads `torchvision.models.resnet18(weights=IMAGENET1K_V1)`,
  freezes all parameters except `layer4` and a replaced `fc` (512->7).

**4. Training**
- `training/trainer.py`: one `Trainer` class (optimizer, CrossEntropyLoss,
  epoch loop, validation, early stopping, checkpoint saving) used by both
  `scripts/train_cnn.py` and `scripts/train_resnet.py`.

**5. Clean evaluation & model selection**
- `scripts/evaluate_clean.py`: loads both checkpoints, runs accuracy/F1/
  confusion matrix on the FER2013 test split, saves plots to
  `results/clean_eval/`, and writes `results/winner.json` recording which
  architecture is the **victim** and which is the **substitute** for the
  later attack phase.

**6. Tests**
- Lightweight `pytest` smoke tests: dataset item shapes for both
  transform pipelines, forward-pass output shape for both models.

## Verification

- `pytest tests/test_models.py` confirms both architectures produce
  `(batch, 7)` logits from a synthetic batch.
- A short training run (1-2 epochs) manually inspected for decreasing loss
  confirms the training loop works end-to-end on CPU before committing to
  full training runs.
- `results/clean_eval/` contains a confusion matrix and metrics table for
  both models, and `results/winner.json` names the victim.

## Phase 2 (later, not in this pass)

Attack implementation (white-box FGSM/PGD, black-box transfer + Square
Attack, backdoor poisoning) via ART, plus `src/fer/attacks/`,
`scripts/run_*_attack.py`, and the corresponding `results/*_eval/` folders.
