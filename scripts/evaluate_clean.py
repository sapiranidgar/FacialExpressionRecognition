"""Compares the two trained models on clean test accuracy and picks the phase-2 attack victim."""

import json
from pathlib import Path

import pandas as pd
import torch
from consts import (
    CONFUSION_MATRIX_TITLE_TEMPLATE,
    MSG_MODEL_SUMMARY_TEMPLATE,
    MSG_SUBSTITUTE_TEMPLATE,
    MSG_VICTIM_TEMPLATE,
    MSG_WROTE_FILE_TEMPLATE,
    PER_CLASS_METRICS_TITLE_TEMPLATE,
    PER_CLASS_TABLE_KEY_CLASS,
    WINNER_KEY_METRICS,
    WINNER_KEY_SUBSTITUTE,
    WINNER_KEY_VICTIM,
)

from fer.config import Config, load_config
from fer.constants import (
    CLEAN_EVAL_SUBDIR,
    CONFUSION_MATRIX_FILE_SUFFIX,
    DEVICE_CPU,
    METRICS_KEY_ACCURACY,
    METRICS_KEY_CONFUSION_MATRIX,
    METRICS_KEY_F1_MACRO,
    METRICS_KEY_PER_CLASS,
    MODEL_TYPE_RESNET,
    MODEL_TYPE_SIMPLE_CNN,
    PER_CLASS_KEY_ACCURACY,
    PER_CLASS_KEY_PRECISION,
    PER_CLASS_KEY_RECALL,
    PER_CLASS_KEY_SUPPORT,
    PER_CLASS_METRICS_CSV_SUFFIX,
    PER_CLASS_METRICS_PLOT_SUFFIX,
    RESNET_RESULT_NAME,
    SIMPLE_CNN_RESULT_NAME,
    WINNER_FILE_NAME,
)
from fer.data.dataset import ModelType, build_dataloaders
from fer.models.resnet_finetune import build_resnet18_finetune
from fer.models.simple_cnn import SimpleCNN
from fer.training.metrics import compute_metrics, predict_all
from fer.utils.visualize import plot_confusion_matrix, plot_per_class_metrics

MODELS: list[tuple[ModelType, str]] = [
    (MODEL_TYPE_SIMPLE_CNN, SIMPLE_CNN_RESULT_NAME),
    (MODEL_TYPE_RESNET, RESNET_RESULT_NAME),
]


def load_model(model_type: ModelType, config: Config) -> torch.nn.Module:
    """Build the given model architecture and load its trained checkpoint.

    Args:
        model_type: Either "simple_cnn" or "resnet".
        config: Loaded project configuration.

    Returns:
        The model in eval() mode with trained weights loaded.
    """
    if model_type == MODEL_TYPE_SIMPLE_CNN:
        model = SimpleCNN(num_classes=config.data.num_classes, in_channels=1)
        ckpt_path = config.paths.checkpoints_dir / config.simple_cnn.checkpoint_name
    else:
        model = build_resnet18_finetune(
            num_classes=config.data.num_classes,
            unfrozen_layers=config.resnet_finetune.unfrozen_layers,
        )
        ckpt_path = config.paths.checkpoints_dir / config.resnet_finetune.checkpoint_name

    state_dict = torch.load(ckpt_path, map_location=DEVICE_CPU)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def per_class_dataframe(per_class: dict[int, dict[str, float]], class_names: list[str]) -> pd.DataFrame:
    """Convert a per-class metrics dict into a table for printing/saving.

    Args:
        per_class: Dict keyed by class index, as returned by
            fer.training.metrics.compute_metrics's "per_class" entry.
        class_names: Class names in label-index order.

    Returns:
        A DataFrame with one row per class: class, precision, recall,
        accuracy, support.
    """
    rows = [
        {
            PER_CLASS_TABLE_KEY_CLASS: name,
            PER_CLASS_KEY_PRECISION: per_class[idx][PER_CLASS_KEY_PRECISION],
            PER_CLASS_KEY_RECALL: per_class[idx][PER_CLASS_KEY_RECALL],
            PER_CLASS_KEY_ACCURACY: per_class[idx][PER_CLASS_KEY_ACCURACY],
            PER_CLASS_KEY_SUPPORT: per_class[idx][PER_CLASS_KEY_SUPPORT],
        }
        for idx, name in enumerate(class_names)
    ]
    return pd.DataFrame(rows)


def evaluate_model(model_type: ModelType, config: Config) -> dict:
    """Load a trained model and compute its metrics on the test split.

    Args:
        model_type: Either "simple_cnn" or "resnet".
        config: Loaded project configuration.

    Returns:
        The metrics dict produced by fer.training.metrics.compute_metrics.
    """
    _, _, test_loader = build_dataloaders(model_type, config)
    model = load_model(model_type, config)
    y_true, y_pred = predict_all(model, test_loader, DEVICE_CPU)
    return compute_metrics(y_true, y_pred, config.data.num_classes)


def save_evaluation_artifacts(name: str, metrics: dict, class_names: list[str], results_dir: Path) -> pd.DataFrame:
    """Save the confusion matrix and per-class metrics (CSV + plot) for one model.

    Args:
        name: Result name used as the file prefix (e.g. "simple_cnn").
        metrics: The metrics dict produced by compute_metrics for this model.
        class_names: Class names in label-index order.
        results_dir: Directory the artifacts are written into.

    Returns:
        The per-class metrics table, for reuse when printing the console report.
    """
    plot_confusion_matrix(
        metrics[METRICS_KEY_CONFUSION_MATRIX],
        class_names,
        results_dir / f"{name}{CONFUSION_MATRIX_FILE_SUFFIX}",
        title=CONFUSION_MATRIX_TITLE_TEMPLATE.format(name=name),
    )

    per_class_df = per_class_dataframe(metrics[METRICS_KEY_PER_CLASS], class_names)
    per_class_df.to_csv(results_dir / f"{name}{PER_CLASS_METRICS_CSV_SUFFIX}", index=False)
    plot_per_class_metrics(
        metrics[METRICS_KEY_PER_CLASS],
        class_names,
        results_dir / f"{name}{PER_CLASS_METRICS_PLOT_SUFFIX}",
        title=PER_CLASS_METRICS_TITLE_TEMPLATE.format(name=name),
    )
    return per_class_df


def summarize_model(metrics: dict, class_names: list[str]) -> dict:
    """Build the JSON-serializable summary entry for one model.

    Args:
        metrics: The metrics dict produced by compute_metrics for this model.
        class_names: Class names in label-index order.

    Returns:
        A dict with "accuracy", "f1_macro", and "per_class" (keyed by class
        name rather than index, so it serializes to readable JSON).
    """
    return {
        METRICS_KEY_ACCURACY: metrics[METRICS_KEY_ACCURACY],
        METRICS_KEY_F1_MACRO: metrics[METRICS_KEY_F1_MACRO],
        METRICS_KEY_PER_CLASS: {
            class_names[idx]: values for idx, values in metrics[METRICS_KEY_PER_CLASS].items()
        },
    }


def print_model_report(name: str, metrics: dict, per_class_df: pd.DataFrame) -> None:
    """Print the console summary for one model: overall metrics plus the per-class table.

    Args:
        name: Result name identifying the model in the printed output.
        metrics: The metrics dict produced by compute_metrics for this model.
        per_class_df: The per-class metrics table from save_evaluation_artifacts.
    """
    print(
        MSG_MODEL_SUMMARY_TEMPLATE.format(
            name=name, accuracy=metrics[METRICS_KEY_ACCURACY], f1_macro=metrics[METRICS_KEY_F1_MACRO]
        )
    )
    print(per_class_df.to_string(index=False))


def select_victim_and_substitute(summary: dict[str, dict]) -> tuple[str, str]:
    """Pick the phase-2 attack victim and substitute from the model summaries.

    The model with the higher clean accuracy becomes the victim (the attack
    target); the other becomes the substitute used for the transfer-based
    black-box attack.

    Args:
        summary: Dict keyed by result name, each value from summarize_model.

    Returns:
        A (victim, substitute) tuple of result names.
    """
    victim = max(summary, key=lambda k: summary[k][METRICS_KEY_ACCURACY])
    substitute = next(k for k in summary if k != victim)
    return victim, substitute


def write_winner_file(victim: str, substitute: str, summary: dict[str, dict], config: Config) -> Path:
    """Write results/winner.json recording the victim, substitute, and full metrics.

    Args:
        victim: Result name of the attack victim.
        substitute: Result name of the substitute model.
        summary: Dict keyed by result name, each value from summarize_model.
        config: Loaded project configuration.

    Returns:
        The path winner.json was written to.
    """
    winner_info = {WINNER_KEY_VICTIM: victim, WINNER_KEY_SUBSTITUTE: substitute, WINNER_KEY_METRICS: summary}
    winner_path = config.paths.results_dir / WINNER_FILE_NAME
    with open(winner_path, "w") as f:
        json.dump(winner_info, f, indent=2)
    return winner_path


def print_winner_report(victim: str, substitute: str, winner_path: Path) -> None:
    """Print the final victim/substitute decision and where it was saved.

    Args:
        victim: Result name of the attack victim.
        substitute: Result name of the substitute model.
        winner_path: Path winner.json was written to.
    """
    print(MSG_VICTIM_TEMPLATE.format(victim=victim))
    print(MSG_SUBSTITUTE_TEMPLATE.format(substitute=substitute))
    print(MSG_WROTE_FILE_TEMPLATE.format(path=winner_path))


def main() -> None:
    """Evaluate both trained models on the test split and record the attack victim/substitute.

    For each model, saves a confusion matrix, a per-class metrics table
    (CSV + bar chart), and prints a summary. Writes results/winner.json
    naming whichever model has the higher clean accuracy as the "victim"
    (phase-2 attack target) and the other as the "substitute" (for the
    transfer-based black-box attack).
    """
    config = load_config()
    results_dir = config.paths.results_dir / CLEAN_EVAL_SUBDIR
    results_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    for model_type, name in MODELS:
        metrics = evaluate_model(model_type, config)
        per_class_df = save_evaluation_artifacts(name, metrics, config.data.class_names, results_dir)
        summary[name] = summarize_model(metrics, config.data.class_names)
        print_model_report(name, metrics, per_class_df)

    victim, substitute = select_victim_and_substitute(summary)
    winner_path = write_winner_file(victim, substitute, summary, config)
    print_winner_report(victim, substitute, winner_path)


if __name__ == "__main__":
    main()
