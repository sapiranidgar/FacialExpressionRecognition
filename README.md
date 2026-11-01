# FacialExpressionRecognition

This repository represents implementation of a facial recognition model and its attack attempts.

## The Task
Build a CNN (or fine-tune a pretrained model) to classify emotions from human face images.
In the next step, implement an attack on this model.

See [PLAN.md](PLAN.md) for the original design discussion and the phase-2 (attacks) plan.

## Project Structure

```
configs/
  config.yaml              # single source of truth: paths + every hyperparameter
src/fer/
  config/                  # typed dataclasses (one per section) + the YAML loader
  data/
    download.py            # pulls FER2013 from Kaggle
    dataset.py             # transforms, datasets, dataloaders - the hub everything else plugs into
  preprocessing/            # train-only augmentation + class-imbalance handling (see below)
  models/
    simple_cnn.py           # SimpleCNN, trained from scratch
    resnet_finetune.py      # ResNet18, fine-tuned from ImageNet weights
  training/
    trainer.py               # Trainer: shared train/val loop, LR scheduling, early stopping
    metrics.py                # accuracy/F1/confusion-matrix/per-class metrics
  tuning/                    # grid search over lr/weight_decay (see below)
  utils/                      # seeding, plotting
scripts/                       # thin CLI entry points over the fer package (this is what you run)
tests/                          # pytest suite, one file per fer/ subpackage roughly
checkpoints/                     # trained weights (gitignored, created by running the scripts)
results/                          # metrics, plots, tuning reports (gitignored)
```

## Code Flow

Everything is driven by **`configs/config.yaml`**, loaded once via `fer.config.load_config()`
into a typed `Config` object that gets passed down through every layer below - there's no
hidden global state or hardcoded path/hyperparameter anywhere else in `src/fer`.

1. **`fer.data.download`** pulls FER2013 into `data/fer2013/{train,test}/<emotion>/*.jpg`.
2. **`fer.data.dataset`** builds the transform pipeline per model type (resize/normalize,
   plus train-only augmentation from `fer.preprocessing`), splits the train folder into
   train/val, and returns dataloaders - optionally also a class-balanced sampler and/or a
   class-weights tensor from `fer.preprocessing`, depending on `config.yaml`.
3. **`fer.models`** defines the two architectures (`SimpleCNN`, `build_resnet18_finetune`) as
   plain `nn.Module`s - no training logic lives here.
4. **`fer.training.Trainer`** is the one training loop both models go through: it owns the
   optimizer, the LR scheduler, early stopping, checkpointing, and evaluation. A model only
   differs from another by which `nn.Module` and `TrainConfig` you hand it.
5. **`scripts/train_cnn.py`** / **`scripts/train_resnet.py`** wire steps 2-4 together for one
   model and save a checkpoint + training-curve plot.
6. **`scripts/tune_hyperparameters.py`** (optional, see below) searches for better `lr`/
   `weight_decay` values *before* a full training run, using the same `Trainer` underneath.
7. **`scripts/evaluate_clean.py`** loads both checkpoints, scores them on the untouched test
   split, and decides which one is the stronger model.

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

Every step reads its settings from [`configs/config.yaml`](configs/config.yaml) - edit that
file (not the scripts) to change paths, batch size, epochs, or any hyperparameter. All
training runs on CPU and prints a live per-batch progress bar plus a per-epoch summary
(loss/accuracy/current learning rate) with an ETA, so long runs are easy to monitor.

```
# 1. Download the dataset
python scripts/download_dataset.py

# 2. (Optional) search for better hyperparameters before a full training run
python scripts/tune_hyperparameters.py --model simple_cnn
python scripts/tune_hyperparameters.py --model resnet

# 3. Train both models (independent of each other and of step 2 - run in either order)
python scripts/train_cnn.py
python scripts/train_resnet.py

# 4. Compare both on the test set and pick the stronger one
python scripts/evaluate_clean.py

# (optional) run the test suite
pytest tests/
```

Step 2 is optional but recommended the first time you set this up on new data or after
changing the architecture - it updates `config.yaml`'s `lr`/`weight_decay` for you (see
**Hyperparameter Tuning** below), so step 3 trains with better settings than the committed
defaults. Step 4 requires both checkpoints from step 3 to already exist.

To find the *best* FER model end-to-end from a clean checkout, run steps 1-4 in order
exactly as listed above.

## The Models

Both models are trained and evaluated the same way (Adam optimizer, cross-entropy loss -
optionally class-weighted, `ReduceLROnPlateau` learning-rate scheduling, early stopping on
validation loss, best-epoch checkpointing) via the shared `fer.training.Trainer` - only the
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
- Hyperparameters (from `config.yaml`'s `simple_cnn` section): up to 50 epochs,
  learning rate `1e-3`, weight decay `1e-4`, early stopping patience 8, LR
  halved after 3 epochs without validation-loss improvement.

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
- Hyperparameters: up to 20 epochs, learning rate `5e-4`, weight decay `1e-4`,
  early stopping patience 6, LR halved after 2 epochs without improvement
  (shorter patience than the CNN since it starts from pretrained weights
  rather than from scratch).

## Preprocessing

`fer.preprocessing` (wired in by `fer.data.dataset.build_dataloaders`) implements two
independent improvements over the raw dataset, both controlled by `config.yaml`'s
`preprocessing` section and both applied **only to the training split** - never to
validation or test. Augmentation exists to stop the model from memorizing exact training
pixels; the validation and test splits have to stay fixed, real, unmodified data, since
that's what early stopping and the final reported accuracy are supposed to measure. Mixing
augmented images into evaluation would make results non-deterministic and incomparable
between models, and would also undermine phase 2's attacks, where "clean accuracy" needs to
mean accuracy on real, untampered images.

**Data augmentation** (`use_augmentation: true` by default): horizontal flip (p=0.5),
random rotation (±15°), and random erasing (p=0.25) - cheap, label-preserving
transformations for a face (flipping or slightly rotating a face doesn't change its
expression) that meaningfully reduce overfitting on FER2013's relatively small training set.

**Class-imbalance handling**: FER2013 is skewed - "disgust" has roughly 16x fewer images
than "happy" - so without correction a model can score well overall while almost never
predicting the rare classes. Two alternative fixes are implemented (enable at most one, via
`use_class_weights` / `use_weighted_sampler` - combining them double-corrects):
- **Class-weighted loss** (the default): `CrossEntropyLoss` penalizes missing a rare class
  more than a common one.
- **Weighted sampler**: oversamples rare classes in the training `DataLoader` instead, so
  every batch is already more balanced.

Both derive their weights from inverse class frequency, softened by `class_weight_power`
(`0.5` by default - the square root of the raw ratio). Raw inverse-frequency weighting
(`1.0`) was tried first and fixed "disgust"'s recall, but at the cost of tanking its
precision by over-predicting it; `0.5` is a gentler correction.

## Hyperparameter Tuning

`scripts/tune_hyperparameters.py` (backed by `fer.tuning`) grid-searches `lr` and
`weight_decay` for one model at a time:

```
python scripts/tune_hyperparameters.py --model simple_cnn   # or --model resnet
python scripts/tune_hyperparameters.py --model simple_cnn --trial-epochs 12   # override the default
```

- **Search space**: 3 values each for `lr` and `weight_decay` (defined in
  `fer.tuning.search_space.DEFAULT_GRIDS`, centered on each model's current `config.yaml`
  value) → 9 trials per model.
- **Per trial**: a fresh model is trained for a reduced epoch budget (8 by default, override
  with `--trial-epochs`) with a shortened early-stopping/LR-scheduler patience, then scored
  on **validation macro F1** - chosen over accuracy because it reflects per-class balance
  (the exact thing the preprocessing step above is trying to fix), and over validation loss
  because it directly measures the metric the project cares about. The test split is never
  used for tuning, for the same reason it's never augmented.
- **Scratch checkpoints**: each trial briefly writes to `results/tuning/<model>/`, deleted
  immediately after scoring - this never touches the real `checkpoints/*.pt` files.
- **Output**: a ranked CSV at `results/tuning/<model>_grid_search_results.csv`, a printed
  results table, and an **auto-write** of the winning `lr`/`weight_decay` back into
  `configs/config.yaml`'s corresponding section (`simple_cnn:` or `resnet_finetune:`), using
  `ruamel.yaml`'s round-trip mode so every comment and every other setting in the file is
  left untouched. Since `config.yaml` is tracked by git, any auto-written change is visible
  in `git diff` and trivially revertible.

Run this before `train_cnn.py` / `train_resnet.py` to train the full model with
better-than-default hyperparameters, or skip it entirely and train with the committed
defaults - the two training scripts don't require it.

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

## Where Models and Results Are Saved

Trained weights (the best epoch by validation loss, not necessarily the
last one) are saved to the `checkpoints/` directory:

- `checkpoints/simple_cnn.pt`
- `checkpoints/resnet18_finetuned.pt`

`checkpoints/` and `results/` are both gitignored - they're regenerated by running the
scripts above, not committed to the repo. Checkpoint filenames are configurable via each
model's `checkpoint_name` entry in `config.yaml`. `results/` additionally holds
`results/clean_eval/` (plots and metrics from training and `evaluate_clean.py`),
`results/winner.json` (the victim/substitute decision), and `results/tuning/` (grid-search
reports, if you've run `tune_hyperparameters.py`).

### The Attacks

- Not implemented yet. Planned: white-box evasion (FGSM, PGD), black-box
  evasion (transfer-based and query-based via Square Attack), and a
  BadNets-style backdoor poisoning attack. See [PLAN.md](PLAN.md).
