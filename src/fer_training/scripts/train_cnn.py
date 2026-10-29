"""Trains the from-scratch SimpleCNN on FER2013 and evaluates it on the test split."""

from common.constants import DEVICE_CPU, METRICS_KEY_ACCURACY, METRICS_KEY_F1_MACRO, MODEL_TYPE_SIMPLE_CNN
from common.data.dataset import build_dataloaders
from common.data.dataset_config import DatasetConfig
from common.models.simple_cnn import SimpleCNN
from common.training.train_config import TrainConfig
from common.training.train_history import TrainHistory
from common.training.trainer import Trainer
from common.utils.seed import set_seed
from common.utils.visualize import plot_training_curves
from torch.utils.data import DataLoader

from fer_training.config import Config, load_config
from fer_training.constants import CLEAN_EVAL_SUBDIR, SIMPLE_CNN_RESULT_NAME, TRAINING_CURVES_FILE_SUFFIX
from fer_training.scripts.consts import MSG_SIMPLE_CNN_TEST_METRICS_TEMPLATE, SIMPLE_CNN_TRAINING_TITLE


def build_dataset_config(config: Config) -> DatasetConfig:
    """Translate fer_training's Config into common's plain DatasetConfig for the SimpleCNN.

    Args:
        config: Loaded project configuration.

    Returns:
        A DatasetConfig populated from config.simple_cnn/data/paths/preprocessing.
    """
    return DatasetConfig(
        image_size=config.simple_cnn.image_size,
        train_dir=config.paths.train_dir,
        test_dir=config.paths.test_dir,
        class_names=config.data.class_names,
        val_fraction=config.data.val_fraction,
        seed=config.seed,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        num_classes=config.data.num_classes,
        preprocessing=config.preprocessing,
    )


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
        lr_scheduler_patience=config.simple_cnn.lr_scheduler_patience,
        lr_scheduler_factor=config.simple_cnn.lr_scheduler_factor,
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

    dataset_config = build_dataset_config(config)
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(MODEL_TYPE_SIMPLE_CNN, dataset_config)

    model = build_model(config)
    train_cfg = build_train_config(config)

    trainer = Trainer(model, train_cfg, device=DEVICE_CPU, class_weights=class_weights)
    history = trainer.fit(train_loader, val_loader)

    save_training_curves(history, config)
    evaluate_and_report(trainer, test_loader)


if __name__ == "__main__":
    main()
