# common

`common` is the shared library underneath every other package in this repo. It holds the two model architectures, the training loop, data augmentation and class-imbalance handling, dataset/dataloader construction, metrics, and plotting utilities. It is never run directly — it has no CLI entry point and no config file of its own; every other package imports from it.

## The Task

`common` exists so that "build a model," "train it," "load a batch of FER2013 images," and "compute accuracy/F1/a confusion matrix" are implemented exactly once, instead of once per package. Right now that means `fer_training`'s scripts, but the same code is meant to be reused later by `fer_server` (to build the victim model and preprocess an incoming image the same way training did), by `attacks` (white-box attacks need a loaded model with working gradients; the backdoor attack needs to retrain), and by `defenses` (adversarial training needs to retrain too).

To make that reuse possible without creating a dependency in the wrong direction, `common` has one hard rule: **it never reads `fer_training`'s `config.yaml` or imports anything from `fer_training`.** Every public function and class takes plain arguments — paths, floats, ints, or a small settings dataclass defined inside `common` itself (`TrainConfig`, `PreprocessingConfig`, `DatasetConfig`). Callers translate their own configuration into these plain objects before calling in.

## Package Structure

```
common/
├── constants.py              # identifiers, normalization stats, architecture constants, metrics-dict keys, plotting constants
├── models/
│   ├── simple_cnn.py          # SimpleCNN - from-scratch CNN
│   └── resnet_finetune.py     # build_resnet18_finetune - ImageNet-pretrained ResNet18, adapted
├── training/
│   ├── trainer.py              # Trainer - the train/val loop, LR scheduling, early stopping, checkpointing
│   ├── train_config.py          # TrainConfig - plain settings for one Trainer run
│   ├── train_history.py          # TrainHistory - per-epoch loss/accuracy log
│   └── metrics.py                 # compute_metrics, predict_all
├── preprocessing/
│   ├── preprocessing_config.py     # PreprocessingConfig - plain augmentation/imbalance settings
│   ├── augmentation_pipeline.py     # composes the individual augmentations below
│   ├── horizontal_flip.py            # }
│   ├── random_rotation.py             # } individual augmentation techniques
│   ├── random_erasing.py               # }
│   ├── class_frequency.py               # shared inverse-frequency weight math
│   ├── class_weights.py                  # per-class loss weights
│   ├── class_balanced_sampler.py          # alternative: oversampling sampler
│   └── labels.py                           # extracts labels from a dataset without decoding images
├── data/
│   ├── dataset_config.py        # DatasetConfig - plain settings for building datasets/dataloaders
│   └── dataset.py                # get_transforms, build_datasets, build_dataloaders
├── utils/
│   ├── seed.py                    # set_seed - reproducibility
│   └── visualize.py                # confusion matrix / per-class / training-curve plots
└── requirements.txt
```

## Flow of the Code

A caller (today, `fer_training`'s scripts) does the following, in order:

1. Build a **`PreprocessingConfig`** and a **`DatasetConfig`** (the latter embeds the former) from its own settings.
2. Call **`common.data.dataset.build_dataloaders(model_type, dataset_config)`**. Internally this builds the train/val/test `ImageFolder` datasets (loading the train folder twice — once augmented, once plain — so validation images never see augmentation even though they're drawn from the same files as training), wraps them in `DataLoader`s, and optionally returns a class-weights tensor or swaps in a class-balanced sampler, depending on `dataset_config.preprocessing`.
3. Instantiate a model from **`common.models`** (`SimpleCNN` or `build_resnet18_finetune`) — this step has no dependency on steps 1–2 at all.
4. Build a **`TrainConfig`** and construct a **`common.training.Trainer`** with the model, the `TrainConfig`, and (optionally) the class-weights tensor from step 2.
5. Call **`trainer.fit(train_loader, val_loader)`** — runs the epoch loop (`Trainer._run_epoch`), stepping a `ReduceLROnPlateau` scheduler and tracking early stopping on validation loss each epoch, restoring the best-validation-loss weights and saving them to `train_config.checkpoint_path` at the end. Returns a `TrainHistory`.
6. Call **`trainer.evaluate(loader)`** — runs `predict_all` + `compute_metrics` (from `common.training.metrics`) over any loader, returning overall accuracy/macro-F1/confusion-matrix/per-class precision-recall-accuracy-support.
7. Optionally pass the returned history/metrics into **`common.utils.visualize`**'s plotting functions, and always through **`common.utils.seed.set_seed`** at the very start for reproducibility.

`constants.py` sits underneath nearly every other file (model-type identifiers, normalization stats, the `compute_metrics` dict-key contract); nothing in `common` imports from outside the package.

## Configuration

`common` has no YAML config and no `load_config()` of its own. Instead there are three plain dataclasses a caller constructs directly:

- **`PreprocessingConfig`** (`common/preprocessing/preprocessing_config.py`): `use_augmentation`, `horizontal_flip_probability`, `rotation_degrees`, `random_erasing_probability`, `use_class_weights`, `use_weighted_sampler`, `class_weight_power`.
- **`DatasetConfig`** (`common/data/dataset_config.py`): `image_size`, `train_dir`, `test_dir`, `class_names`, `val_fraction`, `seed`, `batch_size`, `num_workers`, `num_classes`, `preprocessing` (a `PreprocessingConfig`).
- **`TrainConfig`** (`common/training/train_config.py`): `epochs`, `lr`, `weight_decay`, `early_stopping_patience`, `lr_scheduler_patience`, `lr_scheduler_factor`, `checkpoint_path`, `num_classes`.

If you want to change an actual hyperparameter value (learning rate, augmentation probabilities, etc.) for the training pipeline that exists today, edit `configs/config.yaml` in the repo root — that's `fer_training`'s file, not `common`'s. See `fer_training/README.md`.

## Execution Pipeline

Not applicable — `common` has no entry point. It is imported, not run. See `fer_training/README.md` for the actual commands that exercise this code end to end.

## Where Results Are Saved

`common` doesn't decide *where* anything is saved — `Trainer.fit()` writes a checkpoint to whatever `train_config.checkpoint_path` the caller gave it, and the `visualize` functions write a PNG to whatever `out_path` is passed in. The actual locations (today, inside `artifacts/checkpoints/` and `artifacts/results/`) are a decision made by the caller (`fer_training`), not by `common`.
