"""Computes classification metrics and runs inference over a dataloader."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm

from fer.constants import (
    METRICS_KEY_ACCURACY,
    METRICS_KEY_CONFUSION_MATRIX,
    METRICS_KEY_F1_MACRO,
    METRICS_KEY_PER_CLASS,
    PER_CLASS_KEY_ACCURACY,
    PER_CLASS_KEY_PRECISION,
    PER_CLASS_KEY_RECALL,
    PER_CLASS_KEY_SUPPORT,
    TQDM_BATCH_UNIT,
    TQDM_EVAL_DESCRIPTION,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    """Compute overall and per-class classification metrics.

    Args:
        y_true: Ground-truth class indices, shape (n_samples,).
        y_pred: Predicted class indices, shape (n_samples,).
        num_classes: Total number of classes.

    Returns:
        A dict with keys:
            "accuracy": overall accuracy (float).
            "f1_macro": macro-averaged F1 score (float).
            "confusion_matrix": (num_classes, num_classes) numpy array.
            "per_class": dict keyed by class index, each value a dict with
                "precision", "recall", one-vs-rest "accuracy", and "support"
                (number of ground-truth samples of that class).
    """
    labels = list(range(num_classes))
    accuracy = float((y_true == y_pred).mean())
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    precision_per_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    per_class = {}
    for i in labels:
        # One-vs-rest accuracy: fraction of all samples correctly identified
        # as "class i" or "not class i" - distinct from recall, which only
        # looks at samples whose true label is i.
        class_accuracy = float(((y_true == i) == (y_pred == i)).mean())
        per_class[i] = {
            PER_CLASS_KEY_PRECISION: float(precision_per_class[i]),
            PER_CLASS_KEY_RECALL: float(recall_per_class[i]),
            PER_CLASS_KEY_ACCURACY: class_accuracy,
            PER_CLASS_KEY_SUPPORT: int((y_true == i).sum()),
        }

    return {
        METRICS_KEY_ACCURACY: accuracy,
        METRICS_KEY_F1_MACRO: f1_macro,
        METRICS_KEY_CONFUSION_MATRIX: cm,
        METRICS_KEY_PER_CLASS: per_class,
    }


@torch.no_grad()
def predict_all(model: torch.nn.Module, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Run `model` over every batch in `loader` and collect predictions.

    Args:
        model: A trained classifier in any mode (set to eval() internally).
        loader: Dataloader yielding (images, labels) batches.
        device: Torch device string the model's inputs should be moved to.

    Returns:
        A (y_true, y_pred) tuple of concatenated numpy arrays of class indices.
    """
    model.eval()
    all_preds = []
    all_labels = []
    for images, labels in tqdm(loader, desc=TQDM_EVAL_DESCRIPTION, leave=False, unit=TQDM_BATCH_UNIT):
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)
