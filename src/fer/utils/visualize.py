"""Plotting helpers for confusion matrices, per-class metrics, and training curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from fer.constants import (
    CONFUSION_MATRIX_FIGSIZE,
    PER_CLASS_BAR_WIDTH,
    PER_CLASS_METRIC_NAMES,
    PER_CLASS_METRICS_FIGSIZE,
    SCORE_AXIS_MAX,
    TICK_LABEL_ROTATION_DEGREES,
    TRAINING_CURVES_FIGSIZE,
)


def plot_confusion_matrix(
    cm: np.ndarray, class_names: list[str], out_path: Path, title: str = "Confusion Matrix"
) -> None:
    """Render and save a confusion matrix heatmap.

    Args:
        cm: (num_classes, num_classes) confusion matrix, rows are true
            labels and columns are predicted labels.
        class_names: Class names in label-index order, used as axis ticks.
        out_path: File path the PNG is saved to; parent directories are
            created if needed.
        title: Plot title.
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=CONFUSION_MATRIX_FIGSIZE)
    disp.plot(ax=ax, xticks_rotation=TICK_LABEL_ROTATION_DEGREES, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_class_metrics(
    per_class: dict[int, dict[str, float]],
    class_names: list[str],
    out_path: Path,
    title: str = "Per-Class Metrics",
) -> None:
    """Render and save a grouped bar chart of per-class precision/recall/accuracy.

    Args:
        per_class: Dict keyed by class index, as returned by
            fer.training.metrics.compute_metrics's "per_class" entry.
        class_names: Class names in label-index order, used as the x-axis.
        out_path: File path the PNG is saved to; parent directories are
            created if needed.
        title: Plot title.
    """
    x = np.arange(len(class_names))

    fig, ax = plt.subplots(figsize=PER_CLASS_METRICS_FIGSIZE)
    for i, metric in enumerate(PER_CLASS_METRIC_NAMES):
        values = [per_class[idx][metric] for idx in range(len(class_names))]
        offset = (i - (len(PER_CLASS_METRIC_NAMES) - 1) / 2) * PER_CLASS_BAR_WIDTH
        ax.bar(x + offset, values, PER_CLASS_BAR_WIDTH, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=TICK_LABEL_ROTATION_DEGREES, ha="right")
    ax.set_ylim(0, SCORE_AXIS_MAX)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_training_curves(history: dict[str, list[float]], out_path: Path, title: str = "Training Curves") -> None:
    """Render and save side-by-side loss and accuracy curves for a training run.

    Args:
        history: Dict with keys "train_loss", "val_loss", "train_acc",
            "val_acc" (e.g. from TrainHistory.as_dict()), each a per-epoch list.
        out_path: File path the PNG is saved to; parent directories are
            created if needed.
        title: Figure title.
    """
    fig, axes = plt.subplots(1, 2, figsize=TRAINING_CURVES_FIGSIZE)

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
