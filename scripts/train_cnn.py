"""Trains the from-scratch SimpleCNN on FER2013 and evaluates it on the test split."""

from consts import MSG_SIMPLE_CNN_TEST_METRICS_TEMPLATE, SIMPLE_CNN_TRAINING_TITLE
from torch.utils.data import DataLoader

from fer.config import Config, load_config
from fer.constants import (
    CLEAN_EVAL_SUBDIR,
    DEVICE_CPU,
    METRICS_KEY_ACCURACY,
    METRICS_KEY_F1_MACRO,
    MODEL_TYPE_SIMPLE_CNN,
    SIMPLE_CNN_RESULT_NAME,
    TRAINING_CURVES_FILE_SUFFIX,
)
from fer.data.dataset import build_dataloaders
from fer.models.simple_cnn import SimpleCNN
from fer.training.train_config import TrainConfig
from fer.training.train_history import TrainHistory
from fer.training.trainer import Trainer
from fer.utils.seed import set_seed
from fer.utils.visualize import plot_training_curves


def build_model(config: Config) -> SimpleCNN:
    """Instantiate the SimpleCNN architecture for the given config.

    Args:
        config: Loaded project configuration.

    Returns:
        An untrained SimpleCNN.
    """
    return SimpleCNN(num_classes=config.data.num_classes, in_channels=1)


def build_train_config(config: Config) -> TrainConfig:
    """Build the TrainConfig for the SimpleCNN run from the project config.

    Args:
        config: Loaded project configuration.

    Returns:
        A TrainConfig populated from config.simple_cnn.
    """
    return TrainConfig(
        epochs=config.simple_cnn.epochs,
        lr=config.simple_cnn.lr,
        weight_decay=config.simple_cnn.weight_decay,
        early_stopping_patience=config.simple_cnn.early_stopping_patience,
        checkpoint_path=config.paths.checkpoints_dir / config.simple_cnn.checkpoint_name,
        num_classes=config.data.num_classes,
    )


def save_training_curves(history: TrainHistory, config: Config) -> None:
    """Plot and save the SimpleCNN's training/validation loss and accuracy curves.

    Args:
        history: The TrainHistory returned by Trainer.fit().
        config: Loaded project configuration.
    """
    plot_training_curves(
        history.as_dict(),
        config.paths.results_dir / CLEAN_EVAL_SUBDIR / f"{SIMPLE_CNN_RESULT_NAME}{TRAINING_CURVES_FILE_SUFFIX}",
        title=SIMPLE_CNN_TRAINING_TITLE,
    )


def evaluate_and_report(trainer: Trainer, test_loader: DataLoader) -> None:
    """Evaluate the trained model on the test split and print the result.

    Args:
        trainer: The Trainer holding the trained SimpleCNN.
        test_loader: Dataloader for the test split.
    """
    test_metrics = trainer.evaluate(test_loader)
    print(
        MSG_SIMPLE_CNN_TEST_METRICS_TEMPLATE.format(
            accuracy=test_metrics[METRICS_KEY_ACCURACY], f1_macro=test_metrics[METRICS_KEY_F1_MACRO]
        )
    )


def main() -> None:
    """Load config, train the SimpleCNN end-to-end, and print test-set metrics."""
    config = load_config()
    set_seed(config.seed)

    train_loader, val_loader, test_loader = build_dataloaders(MODEL_TYPE_SIMPLE_CNN, config)

    model = build_model(config)
    train_cfg = build_train_config(config)

    trainer = Trainer(model, train_cfg, device=DEVICE_CPU)
    history = trainer.fit(train_loader, val_loader)

    save_training_curves(history, config)
    evaluate_and_report(trainer, test_loader)


if __name__ == "__main__":
    main()
