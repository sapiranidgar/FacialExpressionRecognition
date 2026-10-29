"""Runs one grid-search trial: train a fresh model with given hyperparameters, score it on validation."""

from __future__ import annotations

from pathlib import Path

import torch
from common.constants import DEVICE_CPU, METRICS_KEY_ACCURACY, METRICS_KEY_F1_MACRO
from common.data.dataset import ModelType
from common.training.train_config import TrainConfig
from common.training.trainer import Trainer
from torch.utils.data import DataLoader

from fer_training.config import Config
from fer_training.constants import (
    TRIAL_EARLY_STOPPING_PATIENCE,
    TRIAL_LR_SCHEDULER_FACTOR,
    TRIAL_LR_SCHEDULER_PATIENCE,
)
from fer_training.tuning.model_factory import build_model_for_tuning
from fer_training.tuning.trial_result import TrialResult


def run_single_trial(
    model_type: ModelType,
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor | None,
    lr: float,
    weight_decay: float,
    trial_epochs: int,
    checkpoint_path: Path,
) -> TrialResult:
    """Train one fresh model for a reduced epoch budget and score it on validation.

    Uses a shortened early-stopping/LR-scheduler patience relative to a full
    training run (TRIAL_EARLY_STOPPING_PATIENCE, TRIAL_LR_SCHEDULER_PATIENCE)
    so trials stay fast. The checkpoint this trial writes is deleted before
    returning - grid-search trials exist only to rank hyperparameters, so
    their weights are never kept, and this never touches the real
    checkpoints used elsewhere in the pipeline.

    Args:
        model_type: Either "simple_cnn" or "resnet".
        config: Loaded project configuration.
        train_loader: Dataloader for the training split.
        val_loader: Dataloader for the validation split.
        class_weights: Optional per-class loss weights, as returned by
            common.data.dataset.build_dataloaders.
        lr: Learning rate for this trial.
        weight_decay: Weight decay for this trial.
        trial_epochs: Maximum epochs for this trial (reduced from a full run).
        checkpoint_path: Scratch path this trial's weights are briefly
            written to, then deleted.

    Returns:
        A TrialResult with this trial's hyperparameters and validation metrics.
    """
    model = build_model_for_tuning(model_type, config)
    train_cfg = TrainConfig(
        epochs=trial_epochs,
        lr=lr,
        weight_decay=weight_decay,
        early_stopping_patience=TRIAL_EARLY_STOPPING_PATIENCE,
        lr_scheduler_patience=TRIAL_LR_SCHEDULER_PATIENCE,
        lr_scheduler_factor=TRIAL_LR_SCHEDULER_FACTOR,
        checkpoint_path=checkpoint_path,
        num_classes=config.data.num_classes,
    )
    trainer = Trainer(model, train_cfg, device=DEVICE_CPU, class_weights=class_weights)
    history = trainer.fit(train_loader, val_loader)
    metrics = trainer.evaluate(val_loader)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return TrialResult(
        lr=lr,
        weight_decay=weight_decay,
        val_f1_macro=metrics[METRICS_KEY_F1_MACRO],
        val_accuracy=metrics[METRICS_KEY_ACCURACY],
        epochs_trained=len(history.train_loss),
    )
