"""Defines Trainer, the train/validation loop shared by every model."""

from __future__ import annotations

import copy
import time
from datetime import timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fer.constants import DEVICE_CPU, TQDM_BATCH_UNIT
from fer.training.metrics import compute_metrics, predict_all
from fer.training.train_config import TrainConfig
from fer.training.train_history import TrainHistory


class Trainer:
    """Runs the train/validation loop, early stopping, and checkpointing.

    Works with any nn.Module classifier; architecture-specific choices (which
    parameters are trainable, learning rate, epoch count) are supplied via the
    model's own requires_grad flags and a TrainConfig instance.
    """

    def __init__(self, model: nn.Module, train_config: TrainConfig, device: str = DEVICE_CPU):
        """Initialize the trainer with a model, run settings, and target device.

        Args:
            model: The model to train. Only parameters with
                requires_grad=True are passed to the optimizer.
            train_config: Hyperparameters and paths for this run.
            device: Torch device string to move the model and batches to.
        """
        self.model = model.to(device)
        self.cfg = train_config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            trainable_params, lr=train_config.lr, weight_decay=train_config.weight_decay
        )
        self.history = TrainHistory()

    def _run_epoch(self, loader: DataLoader, train: bool, epoch: int) -> tuple[float, float]:
        """Run one pass over `loader`, updating weights if `train` is True.

        Displays a per-batch progress bar with running loss/accuracy.

        Args:
            loader: Dataloader to iterate over.
            train: If True, run in training mode with gradient updates;
                otherwise run in eval mode with no gradient tracking.
            epoch: Current epoch number, used only to label the progress bar.

        Returns:
            An (average_loss, accuracy) tuple for this pass over the loader.
        """
        self.model.train(train)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        phase = "train" if train else "val"
        progress = tqdm(
            loader,
            desc=f"epoch {epoch}/{self.cfg.epochs} [{phase}]",
            leave=False,
            unit=TQDM_BATCH_UNIT,
        )

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for images, labels in progress:
                images, labels = images.to(self.device), labels.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                if train:
                    loss.backward()
                    self.optimizer.step()

                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (outputs.argmax(dim=1) == labels).sum().item()
                total_samples += batch_size

                progress.set_postfix(loss=total_loss / total_samples, acc=total_correct / total_samples)

        return total_loss / total_samples, total_correct / total_samples

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainHistory:
        """Train for up to `cfg.epochs` epochs with early stopping.

        After each epoch, prints a summary line with loss/accuracy, time
        taken, and an ETA for the remaining epochs (based on the running
        average epoch duration). Restores the best validation-loss weights
        and saves them to `cfg.checkpoint_path` before returning.

        Args:
            train_loader: Dataloader for the training split.
            val_loader: Dataloader for the validation split.

        Returns:
            The TrainHistory recorded across the run.
        """
        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        epoch_durations: list[float] = []

        for epoch in range(1, self.cfg.epochs + 1):
            start_time = time.time()

            train_loss, train_acc = self._run_epoch(train_loader, train=True, epoch=epoch)
            val_loss, val_acc = self._run_epoch(val_loader, train=False, epoch=epoch)

            duration = time.time() - start_time
            epoch_durations.append(duration)
            avg_duration = sum(epoch_durations) / len(epoch_durations)
            eta = timedelta(seconds=round(avg_duration * (self.cfg.epochs - epoch)))

            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.train_acc.append(train_acc)
            self.history.val_acc.append(val_acc)

            print(
                f"epoch {epoch}/{self.cfg.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"| {duration:.1f}s/epoch, ~{eta} remaining"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.cfg.early_stopping_patience:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(no val improvement for {self.cfg.early_stopping_patience} epochs)"
                    )
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.cfg.checkpoint_path)
        print(f"Saved best checkpoint to {self.cfg.checkpoint_path}")

        return self.history

    def evaluate(self, loader: DataLoader) -> dict:
        """Run the model over `loader` and compute classification metrics.

        Args:
            loader: Dataloader to evaluate on.

        Returns:
            The metrics dict produced by fer.training.metrics.compute_metrics
            (overall accuracy, macro F1, confusion matrix, and per-class
            precision/recall/accuracy/support).
        """
        y_true, y_pred = predict_all(self.model, loader, self.device)
        return compute_metrics(y_true, y_pred, self.cfg.num_classes)
