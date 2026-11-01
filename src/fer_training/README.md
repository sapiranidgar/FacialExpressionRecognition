# fer_training

`fer_training` is a package that trains, tunes, and evaluates the two baseline facial-expression-recognition models (a from-scratch `SimpleCNN` and a fine-tuned `ResNet18`) on FER2013. It builds on top of the shared `common` library for every actual ML operation, and adds everything specific to running that pipeline: a YAML config, downloading the dataset, hyperparameter search, and the CLI scripts you actually run.

## The Task

Build two different facial-expression classifiers, compare them honestly on a held-out test set, and settle on which one is the stronger model — because phase 2 of this project (`attacks`/`defenses`) needs one trained model to attack (the "victim") and, for the transfer-based black-box attack, a second model the attacker has full access to (the "substitute"). `fer_training`'s job ends at that decision: it writes out two trained checkpoints, a full metrics report for both, and a `winner.json` recording which is which. See [`PLAN.md`](../../PLAN.md) at the repo root for the original design rationale (why two models, why these architectures, why these preprocessing choices).

## Package Structure

```
fer_training/
├── constants.py              # Kaggle settings, result-file naming, tuning constants
├── config/
│   ├── loader.py               # load_config() - parses config.yaml, derives paths from `artifacts`
│   ├── full_config.py           # Config - the top-level object
│   ├── paths_config.py           # PathsConfig
│   ├── data_config.py             # DataConfig
│   ├── simple_cnn_config.py        # SimpleCnnConfig
│   └── resnet_finetune_config.py    # ResnetFinetuneConfig
├── data/
│   └── download.py             # downloads FER2013 from Kaggle into artifacts.data.DATA_DIR
├── tuning/
│   ├── search_space.py          # default (lr, weight_decay) grids per model
│   ├── model_factory.py          # builds a fresh model for one trial
│   ├── trial.py                   # runs one trial (train for a reduced budget, score on validation)
│   ├── grid_search.py              # runs every trial in the grid, ranks by validation macro F1
│   ├── results_io.py                # saves/formats trial results
│   ├── config_writer.py              # writes the winning hyperparameters back into config.yaml
│   └── trial_result.py                # TrialResult
├── scripts/                    # the actual CLI entry points - see Execution Pipeline below
│   ├── download_dataset.py
│   ├── train_cnn.py
│   ├── train_resnet.py
│   ├── evaluate_clean.py
│   ├── tune_hyperparameters.py
│   └── consts.py                # print templates / plot titles used only by these scripts
└── requirements.txt            # -r ../common/requirements.txt + pyyaml, ruamel.yaml, kaggle, python-dotenv
```

## Flow of the Code

1. **`fer_training.config.load_config()`** reads `configs/config.yaml` (at the repo root) into a typed `Config`. Paths are *not* read from the YAML — `PathsConfig` is built from `artifacts`'s shared location constants (`artifacts.data.DATA_DIR`, `artifacts.checkpoints.CHECKPOINTS_DIR`, `artifacts.results.RESULTS_DIR`) plus the `"fer2013/train"` / `"fer2013/test"` convention, so this is the one place that translation happens.
2. Each script in `scripts/` has a small `build_dataset_config(config)` and `build_train_config(config)` helper that **translates** the loaded `Config` into `common`'s plain `DatasetConfig`/`TrainConfig` objects — this is the adapter layer that lets `common` stay config-agnostic (see `common/README.md`).
3. From there, the script hands off entirely to `common`: `common.data.dataset.build_dataloaders(...)` → build a model from `common.models` → `common.training.Trainer(...).fit(...)` → `.evaluate(...)`.
4. `tune_hyperparameters.py` follows the same config-loading and translation steps, but instead of a single training run it calls `fer_training.tuning.run_grid_search`, which runs many short `common.training.Trainer` runs (one per hyperparameter combination) and keeps only the best.
5. `evaluate_clean.py` is the only script that loads *both* trained checkpoints, runs inference on the untouched test split for each, and picks the victim/substitute.

## Execution Pipeline

Every command below is run from the **repo root**, with `src` on the Python path:

```bash
# 1. Download the dataset (no-ops if already present)
PYTHONPATH=src python -m fer_training.scripts.download_dataset

# 2. (Optional) search for better lr/weight_decay before a full run - auto-writes config.yaml
PYTHONPATH=src python -m fer_training.scripts.tune_hyperparameters --model simple_cnn
PYTHONPATH=src python -m fer_training.scripts.tune_hyperparameters --model resnet
# override the per-trial epoch budget (default 8):
#   ... --model simple_cnn --trial-epochs 12

# 3. Train both models (independent of each other and of step 2 - either order)
PYTHONPATH=src python -m fer_training.scripts.train_cnn
PYTHONPATH=src python -m fer_training.scripts.train_resnet

# 4. Compare both on the test set, decide the victim/substitute
PYTHONPATH=src python -m fer_training.scripts.evaluate_clean
```

Step 4 requires both checkpoints from step 3 to exist. Training prints a live per-batch progress bar plus a per-epoch summary (loss/accuracy/current learning rate) with an ETA, so long runs are easy to monitor on CPU.

## Configuration

Everything in `configs/config.yaml` (repo root) except credentials and paths:

| Section | Controls |
|---|---|
| `seed` | Global reproducibility seed. |
| `data` | `num_classes`, `class_names` (must match the downloaded dataset's folder names), `val_fraction`, `batch_size`, `num_workers`. |
| `simple_cnn` | `image_size`, `epochs`, `lr`, `weight_decay`, `early_stopping_patience`, `lr_scheduler_patience`/`lr_scheduler_factor`, `checkpoint_name`. Currently: up to 50 epochs, lr `1e-3`, patience 8 (scheduler patience 3). |
| `resnet_finetune` | Same shape, plus `unfrozen_layers` (which top-level ResNet18 submodules stay trainable — everything else is frozen; see `PLAN.md` for why this is CPU-forced). Currently: up to 20 epochs, lr `5e-4`, patience 6 (scheduler patience 2). |
| `preprocessing` | `use_augmentation` + its three probabilities; `use_class_weights`/`use_weighted_sampler` (don't enable both) + `class_weight_power` (`0.5` by default — square-root-softened inverse frequency; `1.0` fixed FER2013's "disgust" recall but hurt its precision). |

Kaggle credentials are **not** in `config.yaml` — see the repo-root `README.md`'s Kaggle section (`.env` file, never committed). Paths are **not** in `config.yaml` either — they're derived from `artifacts`, so every package agrees on where data/checkpoints/results live; there's nothing to configure there.

`tune_hyperparameters.py`'s own search grid (which `lr`/`weight_decay` values to try) is *not* in `config.yaml` — it's a hardcoded default in `fer_training/tuning/search_space.py`, by design: grid search isn't part of the normal "one concrete training run" config surface that `config.yaml` represents.

## Where Results Are Saved

Everything lands inside the shared `artifacts` package (see `artifacts/README.md`):

| What | Where |
|---|---|
| Trained weights | `artifacts/checkpoints/simple_cnn.pt`, `artifacts/checkpoints/resnet18_finetuned.pt` |
| Per-model training curves | `artifacts/results/clean_eval/{simple_cnn,resnet18}_training_curves.png` |
| Confusion matrix, per-class CSV + plot | `artifacts/results/clean_eval/<name>_*.{png,csv}` (from `evaluate_clean.py`) |
| Victim/substitute decision + full metrics | `artifacts/results/winner.json` |
| Grid-search results | `artifacts/results/tuning/<model>_grid_search_results.csv` |
