# artifacts

`artifacts` is pure storage — it contains no logic beyond three one-line path constants. Its only job is to give every other package one reliable, importable place to find the downloaded dataset, trained model checkpoints, and generated results, instead of each package independently guessing a relative path.

## The Task

Multiple packages need to agree on the same physical files: `fer_training` downloads the dataset and writes checkpoints; `fer_server` and `attacks` will need to read those same checkpoints to load the victim model; `attacks`/`defenses` will need to read the same dataset to retrain (the backdoor attack, adversarial training). Rather than each package computing its own relative path and hoping they stay in sync, they all `import` from `artifacts` directly, so there is exactly one source of truth.

## Package Structure

```
artifacts/
├── data/
│   └── __init__.py        # DATA_DIR = Path(__file__).resolve().parent
├── checkpoints/
│   └── __init__.py        # CHECKPOINTS_DIR = Path(__file__).resolve().parent
└── results/
    └── __init__.py        # RESULTS_DIR = Path(__file__).resolve().parent
```

Each sub-package's `__init__.py` is the entire implementation: it resolves its own containing directory and exposes that as a constant. There is deliberately nothing else here.

## Flow of the Code

A consumer imports the constant it needs and uses it directly:

```python
from artifacts.data import DATA_DIR
from artifacts.checkpoints import CHECKPOINTS_DIR
from artifacts.results import RESULTS_DIR
```

Today, `fer_training.config.loader.load_config()` is what actually wires these in: it builds `fer_training`'s `PathsConfig` from these three constants (plus the `"fer2013/train"` / `"fer2013/test"` convention) rather than reading paths from `config.yaml`. `fer_training.data.download.download_fer2013` writes into `DATA_DIR`; `common.training.Trainer.fit()` writes a checkpoint to whatever path it's given, which `fer_training`'s scripts point at `CHECKPOINTS_DIR`; `evaluate_clean.py` reads from `CHECKPOINTS_DIR` and writes into `RESULTS_DIR`.

## Configuration

None — nothing here is configurable. Each path is always "wherever this sub-package lives on disk," computed from `__file__`, not read from any settings file.

## Execution Pipeline

Not applicable — `artifacts` is never run, only imported.

## Where Results Are Saved

This package *is* the "where." Current contents (all gitignored except the three `__init__.py` files — see the repo-root `.gitignore`):

| Directory | Contents |
|---|---|
| `artifacts/data/fer2013/{train,test}/<emotion>/` | The downloaded FER2013 images. |
| `artifacts/checkpoints/*.pt` | Trained model weights. |
| `artifacts/results/clean_eval/` | Training-curve plots, confusion matrices, per-class metric tables/plots. |
| `artifacts/results/winner.json` | The victim/substitute decision and full metrics for both models. |
| `artifacts/results/tuning/` | Grid-search result CSVs. |
