# FacialExpressionRecognition

This repository represents implementation of a facial recognition model and its attack attempts.

## The Task
Build a CNN (or fine-tune a pretrained model) to classify emotions from human face images.
In the next step, implement an attack on this model.

See [PLAN.md](PLAN.md) for the full design and project structure.

## Setup

```
pip install -r requirements.txt
pip install -e .
```

The second command installs this repo's `fer` package in editable mode, so
`scripts/*.py` and `tests/*.py` can `import fer` regardless of where they're
run from.

### Kaggle API credentials

Downloading FER2013 uses the `kaggle` Python package, which authenticates
with a **username + key pair** - not a single token. Kaggle's newer
"Generate New Token" button on https://www.kaggle.com/settings/api gives you
just the key (looking like `KGAT_...`); you still need to pair it with your
Kaggle username yourself.

Create a file named `.env` in the repo root (copy `.env.example`) - it's
already covered by `.gitignore`, so it's never committed:

```
KAGGLE_USERNAME=your-kaggle-username
KAGGLE_KEY=your-key-from-kaggle.com/settings/api
```

`scripts/download_dataset.py` loads this file automatically. If you'd rather
use the older `kaggle.json` file instead, that still works too - the
`kaggle` package accepts either. Never commit your key to the repo or paste
it into a chat/issue; if it ever leaks, rotate it from the same settings
page.

## Running the Full Pipeline

Each step reads its settings from [`configs/config.yaml`](configs/config.yaml)
(paths, batch size, per-model hyperparameters) - edit that file rather than
the scripts if you want to change something.

```
python scripts/download_dataset.py   # 1. download + unzip FER2013 into data/fer2013/
python scripts/train_cnn.py          # 2. train the from-scratch SimpleCNN
python scripts/train_resnet.py       # 3. fine-tune ResNet18
python scripts/evaluate_clean.py     # 4. compare both on the test set, pick the attack victim
pytest tests/                        # (optional) run the test suite
```

Steps 2 and 3 are independent of each other and can run in either order.
Step 4 requires both checkpoints to already exist. All training runs on CPU
and prints a live progress bar per epoch plus an ETA for the epochs
remaining, so long runs are easy to monitor.

## The Models

Both models are trained and evaluated the same way (Adam optimizer,
cross-entropy loss, early stopping on validation loss, best-epoch
checkpointing) via the shared `fer.training.Trainer` - only the
architecture and hyperparameters differ.

### Simple CNN (from scratch) - `fer.models.simple_cnn.SimpleCNN`

A small convolutional network trained from scratch, sized to train quickly
on CPU:

- Input: 48x48 grayscale images (FER2013's native resolution).
- Four convolutional blocks with increasing channel widths (32 -> 64 -> 128
  -> 256), each block = `Conv2d(3x3, padding=1) -> BatchNorm -> ReLU ->
  MaxPool(2)`, halving spatial resolution each time.
- Global average pooling, then `Dropout(0.3) -> Linear` to the 7 emotion
  classes.
- Hyperparameters (from `config.yaml`'s `simple_cnn` section): 30 epochs max,
  learning rate `1e-3`, weight decay `1e-4`, early stopping patience 5.

### Fine-tuned ResNet18 - `fer.models.resnet_finetune.build_resnet18_finetune`

An ImageNet-pretrained `torchvision.models.resnet18`, adapted for this task:

- Input: images upsampled to 224x224 and replicated from 1 to 3 channels,
  normalized with ImageNet statistics (mean/std), matching what the
  pretrained backbone expects.
- The classifier head (`fc`) is replaced with a new `Linear(512, 7)` layer.
- Only `layer4` and `fc` are left trainable; every earlier layer is frozen.
  This is a deliberate CPU-only tradeoff (see [PLAN.md](PLAN.md)) - fine-tuning
  the full network on CPU would be far slower for little accuracy benefit.
  Which layers are unfrozen is configurable via `resnet_finetune.unfrozen_layers`
  in `config.yaml`.
- Hyperparameters: 10 epochs max, learning rate `5e-4`, weight decay `1e-4`,
  early stopping patience 3 (fewer epochs than the CNN since it starts from
  pretrained weights rather than from scratch).

## Evaluation & Model Selection

`scripts/evaluate_clean.py` is what decides which model becomes the phase-2
attack target. For **each** trained model, it runs inference on the FER2013
test split and computes (via `fer.training.metrics.compute_metrics`):

- **Overall accuracy** and **macro-averaged F1**.
- A **confusion matrix**.
- A **per-class breakdown**, for every emotion: precision, recall, a
  one-vs-rest accuracy (is "this image is/isn't class X" correct - distinct
  from recall, which only looks at images actually belonging to class X),
  and support (how many test images belong to that class).

It then prints a console summary and saves, per model:

| File | Contents |
|---|---|
| `results/clean_eval/<name>_confusion_matrix.png` | Confusion matrix heatmap |
| `results/clean_eval/<name>_per_class_metrics.csv` | Precision/recall/accuracy/support table |
| `results/clean_eval/<name>_per_class_metrics.png` | Same table as a grouped bar chart |

where `<name>` is `simple_cnn` or `resnet18_finetuned`. Each training script
also saves its own loss/accuracy-per-epoch plot directly (not via
`evaluate_clean.py`): `results/clean_eval/simple_cnn_training_curves.png`
and `results/clean_eval/resnet18_training_curves.png`.

Finally, it compares the two models' overall accuracy: the higher-accuracy
model becomes the **victim** (the model phase 2's attacks will target); the
other becomes the **substitute** (used for the transfer-based black-box
attack, since it's a model the "attacker" has full access to but the victim
is not). This decision, along with the full metrics for both models, is
written to `results/winner.json`.

## Where Models Are Saved

Trained weights (the best epoch by validation loss, not necessarily the
last one) are saved to the `checkpoints/` directory:

- `checkpoints/simple_cnn.pt`
- `checkpoints/resnet18_finetuned.pt`

Both `checkpoints/` and `results/` are gitignored - they're regenerated by
running the scripts above, not committed to the repo. Filenames are
configurable via each model's `checkpoint_name` entry in `config.yaml`.

### The Attacks

- Not implemented yet. Planned: white-box evasion (FGSM, PGD), black-box
  evasion (transfer-based and query-based via Square Attack), and a
  BadNets-style backdoor poisoning attack. See [PLAN.md](PLAN.md).
