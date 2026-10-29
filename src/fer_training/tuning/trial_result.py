"""Defines TrialResult, the outcome of one grid-search trial."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class TrialResult:
    """The hyperparameters and validation metrics from one grid-search trial.

    Attributes:
        lr: The learning rate used for this trial.
        weight_decay: The weight decay used for this trial.
        val_f1_macro: Macro-averaged F1 on the validation split - the metric
            trials are ranked by.
        val_accuracy: Overall accuracy on the validation split, reported
            alongside val_f1_macro for context.
        epochs_trained: Number of epochs actually run before early stopping
            (or the trial's full epoch budget, if it never triggered).
    """

    lr: float
    weight_decay: float
    val_f1_macro: float
    val_accuracy: float
    epochs_trained: int
