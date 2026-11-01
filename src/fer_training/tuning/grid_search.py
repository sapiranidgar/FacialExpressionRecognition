"""Runs a grid search over a model's default hyperparameter grid, ranked by validation macro F1."""

from __future__ import annotations

from pathlib import Path

import torch
from common.data.dataset import ModelType
from torch.utils.data import DataLoader

from fer_training.config import Config
from fer_training.tuning.search_space import build_param_combinations, get_default_grid
from fer_training.tuning.trial import run_single_trial
from fer_training.tuning.trial_result import TrialResult


def run_grid_search(
    model_type: ModelType,
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor | None,
    trial_epochs: int,
    scratch_dir: Path,
) -> list[TrialResult]:
    """Train and score one trial per (lr, weight_decay) combination in the default grid.

    Only the training and validation splits are used - the test split is
    never touched during hyperparameter search, the same way it's never
    augmented: search decisions must not leak information from the data
    used later to report final, reproducible performance.

    Args:
        model_type: Either "simple_cnn" or "resnet".
        config: Loaded project configuration.
        train_loader: Dataloader for the training split, built once by the
            caller and reused across every trial - only each trial's
            hyperparameters change, not the data.
        val_loader: Dataloader for the validation split.
        class_weights: Optional per-class loss weights, as returned by
            common.data.dataset.build_dataloaders.
        trial_epochs: Maximum epochs per trial.
        scratch_dir: Directory scratch checkpoints are briefly written to
            during each trial.

    Returns:
        Every trial's TrialResult, sorted best-first by val_f1_macro.
    """
    grid = get_default_grid(model_type)
    combinations = build_param_combinations(grid)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    results: list[TrialResult] = []
    for trial_number, (lr, weight_decay) in enumerate(combinations, start=1):
        print(f"\n--- Trial {trial_number}/{len(combinations)}: lr={lr} weight_decay={weight_decay} ---")
        checkpoint_path = scratch_dir / f"trial_{trial_number}.pt"
        result = run_single_trial(
            model_type,
            config,
            train_loader,
            val_loader,
            class_weights,
            lr,
            weight_decay,
            trial_epochs,
            checkpoint_path,
        )
        print(
            f"--- Trial {trial_number} result: "
            f"val_f1_macro={result.val_f1_macro:.4f} val_accuracy={result.val_accuracy:.4f} ---"
        )
        results.append(result)

    results.sort(key=lambda r: r.val_f1_macro, reverse=True)
    return results
