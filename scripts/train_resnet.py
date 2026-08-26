"""Fine-tunes ResNet18 on FER2013 and evaluates it on the test split."""

from consts import (
    MSG_BUILT_LOADERS,
    MSG_BUILT_MODEL,
    MSG_BUILT_TRAINER,
    MSG_EVALUATING_MODEL,
    MSG_FINISHED_TRAINING,
    MSG_RESNET_TEST_METRICS_TEMPLATE,
    MSG_STARTED_TRAINING,
    RESNET_TRAINING_CURVES_PREFIX,
    RESNET_TRAINING_TITLE,
)
from torch import nn
from torch.utils.data import DataLoader

from fer.config import Config, load_config
from fer.constants import (
    CLEAN_EVAL_SUBDIR,
    DEVICE_CPU,
    METRICS_KEY_ACCURACY,
    METRICS_KEY_F1_MACRO,
    MODEL_TYPE_RESNET,
    TRAINING_CURVES_FILE_SUFFIX,
)
from fer.data.dataset import build_dataloaders
from fer.models.resnet_finetune import build_resnet18_finetune
from fer.training.train_config import TrainConfig
from fer.training.train_history import TrainHistory
from fer.training.trainer import Trainer
from fer.utils.seed import set_seed
from fer.utils.visualize import plot_training_curves


def build_model(config: Config) -> nn.Module:
    """Instantiate the ResNet18 fine-tune architecture for the given config.

    Args:
        config: Loaded project configuration.

    Returns:
        ResNet18 with a replaced classifier head and the configured layers unfrozen.
    """
    return build_resnet18_finetune(
        num_classes=config.data.num_classes,
        unfrozen_layers=config.resnet_finetune.unfrozen_layers,
    )


def build_train_config(config: Config) -> TrainConfig:
    """Build the TrainConfig for the ResNet18 fine-tune run from the project config.

    Args:
        config: Loaded project configuration.

    Returns:
        A TrainConfig populated from config.resnet_finetune.
    """
    return TrainConfig(
        epochs=config.resnet_finetune.epochs,
        lr=config.resnet_finetune.lr,
        weight_decay=config.resnet_finetune.weight_decay,
        early_stopping_patience=config.resnet_finetune.early_stopping_patience,
        checkpoint_path=config.paths.checkpoints_dir / config.resnet_finetune.checkpoint_name,
        num_classes=config.data.num_classes,
    )


def save_training_curves(history: TrainHistory, config: Config) -> None:
    """Plot and save the ResNet18's training/validation loss and accuracy curves.

    Args:
        history: The TrainHistory returned by Trainer.fit().
        config: Loaded project configuration.
    """
    plot_training_curves(
        history.as_dict(),
        config.paths.results_dir
        / CLEAN_EVAL_SUBDIR
        / f"{RESNET_TRAINING_CURVES_PREFIX}{TRAINING_CURVES_FILE_SUFFIX}",
        title=RESNET_TRAINING_TITLE,
    )


def evaluate_and_report(trainer: Trainer, test_loader: DataLoader) -> None:
    """Evaluate the trained model on the test split and print the result.

    Args:
        trainer: The Trainer holding the fine-tuned ResNet18.
        test_loader: Dataloader for the test split.
    """
    test_metrics = trainer.evaluate(test_loader)
    print(
        MSG_RESNET_TEST_METRICS_TEMPLATE.format(
            accuracy=test_metrics[METRICS_KEY_ACCURACY], f1_macro=test_metrics[METRICS_KEY_F1_MACRO]
        )
    )


def main() -> None:
    """Load config, fine-tune ResNet18 end-to-end, and print test-set metrics."""
    config = load_config()
    set_seed(config.seed)

    train_loader, val_loader, test_loader = build_dataloaders(MODEL_TYPE_RESNET, config)
    print(MSG_BUILT_LOADERS)

    model = build_model(config)
    print(MSG_BUILT_MODEL)

    train_cfg = build_train_config(config)
    print(MSG_BUILT_TRAINER)

    trainer = Trainer(model, train_cfg, device=DEVICE_CPU)
    print(MSG_STARTED_TRAINING)
    history = trainer.fit(train_loader, val_loader)
    print(MSG_FINISHED_TRAINING)

    save_training_curves(history, config)

    print(MSG_EVALUATING_MODEL)
    evaluate_and_report(trainer, test_loader)


if __name__ == "__main__":
    main()
