"""Defines TrainHistory, the per-epoch metric log recorded during training."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainHistory:
    """Per-epoch loss and accuracy recorded across a training run.

    Attributes:
        train_loss: Training loss at the end of each epoch.
        val_loss: Validation loss at the end of each epoch.
        train_acc: Training accuracy at the end of each epoch.
        val_acc: Validation accuracy at the end of each epoch.
    """

    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    train_acc: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)

    def as_dict(self) -> dict:
        """Return the recorded history as a plain dict keyed by metric name.

        Returns:
            A dict with keys "train_loss", "val_loss", "train_acc", "val_acc",
            each mapping to a list of per-epoch values.
        """
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
        }
