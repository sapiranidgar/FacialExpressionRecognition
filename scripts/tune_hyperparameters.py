"""Grid-searches lr and weight_decay for one model, reporting and auto-writing the winning combo."""

import argparse

from consts import (
    MSG_GRID_SEARCH_BEST_TEMPLATE,
    MSG_GRID_SEARCH_START_TEMPLATE,
    MSG_WROTE_CONFIG_TEMPLATE,
    MSG_WROTE_FILE_TEMPLATE,
)

from fer.config import Config, DEFAULT_CONFIG_PATH, load_config
from fer.constants import (
    DEFAULT_TRIAL_EPOCHS,
    MODEL_TYPE_RESNET,
    MODEL_TYPE_SIMPLE_CNN,
    MODEL_TYPE_TO_CONFIG_SECTION,
    TUNING_RESULTS_CSV_SUFFIX,
    TUNING_SUBDIR,
)
from fer.data.dataset import ModelType, build_dataloaders
from fer.tuning.config_writer import write_best_hyperparameters
from fer.tuning.grid_search import run_grid_search
from fer.tuning.results_io import format_results_table, save_results_csv
from fer.tuning.search_space import build_param_combinations, get_default_grid
from fer.tuning.trial_result import TrialResult
from fer.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse --model and --trial-epochs CLI arguments.

    Returns:
        Parsed arguments with `model` (either "simple_cnn" or "resnet") and
        `trial_epochs` (int).
    """
    parser = argparse.ArgumentParser(description="Grid-search lr and weight_decay for one model.")
    parser.add_argument("--model", required=True, choices=[MODEL_TYPE_SIMPLE_CNN, MODEL_TYPE_RESNET])
    parser.add_argument("--trial-epochs", type=int, default=DEFAULT_TRIAL_EPOCHS)
    return parser.parse_args()


def run_search(model_type: ModelType, config: Config, trial_epochs: int) -> list[TrialResult]:
    """Build dataloaders once and run the grid search for one model.

    Args:
        model_type: Either "simple_cnn" or "resnet".
        config: Loaded project configuration.
        trial_epochs: Maximum epochs per trial.

    Returns:
        Every trial's result, sorted best-first by validation macro F1.
    """
    num_trials = len(build_param_combinations(get_default_grid(model_type)))
    print(MSG_GRID_SEARCH_START_TEMPLATE.format(model=model_type, num_trials=num_trials, trial_epochs=trial_epochs))

    train_loader, val_loader, _, class_weights = build_dataloaders(model_type, config)
    scratch_dir = config.paths.results_dir / TUNING_SUBDIR / model_type
    return run_grid_search(model_type, config, train_loader, val_loader, class_weights, trial_epochs, scratch_dir)


def save_and_report(results: list[TrialResult], model_type: ModelType, config: Config) -> None:
    """Save results to CSV, print the results table, and print the winning trial.

    Args:
        results: Every trial's result, sorted best-first.
        model_type: Either "simple_cnn" or "resnet".
        config: Loaded project configuration.
    """
    results_path = config.paths.results_dir / TUNING_SUBDIR / f"{model_type}{TUNING_RESULTS_CSV_SUFFIX}"
    save_results_csv(results, results_path)
    print(MSG_WROTE_FILE_TEMPLATE.format(path=results_path))

    print(format_results_table(results))

    best = results[0]
    print(
        MSG_GRID_SEARCH_BEST_TEMPLATE.format(
            lr=best.lr,
            weight_decay=best.weight_decay,
            val_f1_macro=best.val_f1_macro,
            val_accuracy=best.val_accuracy,
        )
    )


def apply_best_hyperparameters(results: list[TrialResult], model_type: ModelType) -> None:
    """Write the winning trial's lr and weight_decay back into config.yaml.

    Args:
        results: Every trial's result, sorted best-first.
        model_type: Either "simple_cnn" or "resnet".
    """
    best = results[0]
    write_best_hyperparameters(model_type, best.lr, best.weight_decay, DEFAULT_CONFIG_PATH)
    section = MODEL_TYPE_TO_CONFIG_SECTION[model_type]
    print(MSG_WROTE_CONFIG_TEMPLATE.format(path=DEFAULT_CONFIG_PATH, section=section))


def main() -> None:
    """Load config, run the grid search for the chosen model, report, and auto-write the winner."""
    args = parse_args()
    config = load_config()
    set_seed(config.seed)

    results = run_search(args.model, config, args.trial_epochs)
    save_and_report(results, args.model, config)
    apply_best_hyperparameters(results, args.model)


if __name__ == "__main__":
    main()
