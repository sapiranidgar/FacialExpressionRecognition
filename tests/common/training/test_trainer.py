import torch
from torch.utils.data import DataLoader, TensorDataset

from common.models.simple_cnn import SimpleCNN
from common.training.train_config import TrainConfig
from common.training.trainer import Trainer


def _make_loader(n=16, num_classes=3, batch_size=4):
    # SimpleCNN pools 4 times (halving each time), so inputs need >= 16px
    # per side to survive without collapsing to a 0-sized feature map.
    x = torch.randn(n, 1, 16, 16)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_trainer_applies_class_weights_to_criterion(tmp_path):
    model = SimpleCNN(num_classes=3, in_channels=1)
    class_weights = torch.tensor([1.0, 2.0, 3.0])
    cfg = TrainConfig(
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        early_stopping_patience=5,
        lr_scheduler_patience=2,
        lr_scheduler_factor=0.5,
        checkpoint_path=tmp_path / "model.pt",
        num_classes=3,
    )
    trainer = Trainer(model, cfg, device="cpu", class_weights=class_weights)
    assert torch.equal(trainer.criterion.weight, class_weights)


def test_trainer_with_no_class_weights_uses_unweighted_loss(tmp_path):
    model = SimpleCNN(num_classes=3, in_channels=1)
    cfg = TrainConfig(
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        early_stopping_patience=5,
        lr_scheduler_patience=2,
        lr_scheduler_factor=0.5,
        checkpoint_path=tmp_path / "model.pt",
        num_classes=3,
    )
    trainer = Trainer(model, cfg, device="cpu")
    assert trainer.criterion.weight is None


def test_trainer_reduces_lr_after_plateau(tmp_path):
    torch.manual_seed(0)
    model = SimpleCNN(num_classes=3, in_channels=1)
    cfg = TrainConfig(
        epochs=6,
        lr=0.1,
        weight_decay=0.0,
        early_stopping_patience=10,
        lr_scheduler_patience=1,
        lr_scheduler_factor=0.5,
        checkpoint_path=tmp_path / "model.pt",
        num_classes=3,
    )
    trainer = Trainer(model, cfg, device="cpu")
    train_loader = _make_loader()
    val_loader = _make_loader()

    initial_lr = trainer.optimizer.param_groups[0]["lr"]
    trainer.fit(train_loader, val_loader)
    final_lr = trainer.optimizer.param_groups[0]["lr"]

    # Train/val loaders are independent random tensors, so validation loss
    # cannot consistently improve - with lr_scheduler_patience=1, the LR
    # should have been reduced at least once over 6 epochs.
    assert final_lr < initial_lr
