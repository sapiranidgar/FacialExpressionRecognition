import pandas as pd
import pytest
import torch
from ruamel.yaml import YAML
from torch.utils.data import DataLoader, TensorDataset

from fer_training.config import load_config
from fer_training.tuning.config_writer import write_best_hyperparameters
from fer_training.tuning.grid_search import run_grid_search
from fer_training.tuning.model_factory import build_model_for_tuning
from fer_training.tuning.results_io import format_results_table, results_to_dataframe, save_results_csv
from fer_training.tuning.search_space import build_param_combinations, get_default_grid
from fer_training.tuning.trial import run_single_trial
from fer_training.tuning.trial_result import TrialResult

config = load_config()


def _make_loader(n=16, batch_size=4, size=16):
    # Synthetic size independent of config.simple_cnn.image_size - just
    # needs to survive SimpleCNN's 4 pooling layers (>= 16px per side).
    x = torch.randn(n, 1, size, size)
    y = torch.randint(0, config.data.num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_get_default_grid_has_lr_and_weight_decay_for_each_model():
    for model_type in ("simple_cnn", "resnet"):
        grid = get_default_grid(model_type)
        assert "lr" in grid
        assert "weight_decay" in grid
        assert len(grid["lr"]) > 0
        assert len(grid["weight_decay"]) > 0


def test_build_param_combinations_is_cartesian_product():
    grid = {"lr": [0.1, 0.2], "weight_decay": [0.001, 0.0001, 0.00001]}
    combos = build_param_combinations(grid)
    assert len(combos) == 6
    assert (0.1, 0.001) in combos
    assert (0.2, 0.00001) in combos


def test_build_model_for_tuning_returns_fresh_instances():
    model_a = build_model_for_tuning("simple_cnn", config)
    model_b = build_model_for_tuning("simple_cnn", config)
    assert model_a is not model_b


def test_run_single_trial_returns_result_and_cleans_up_checkpoint(tmp_path):
    train_loader = _make_loader()
    val_loader = _make_loader()
    checkpoint_path = tmp_path / "trial_1.pt"

    result = run_single_trial(
        "simple_cnn",
        config,
        train_loader,
        val_loader,
        None,
        lr=0.001,
        weight_decay=0.0001,
        trial_epochs=1,
        checkpoint_path=checkpoint_path,
    )

    assert isinstance(result, TrialResult)
    assert result.lr == 0.001
    assert result.weight_decay == 0.0001
    assert 0.0 <= result.val_f1_macro <= 1.0
    assert 0.0 <= result.val_accuracy <= 1.0
    assert result.epochs_trained == 1
    assert not checkpoint_path.exists()


def test_run_grid_search_sorts_results_best_first_and_cleans_scratch_dir(tmp_path, monkeypatch):
    train_loader = _make_loader()
    val_loader = _make_loader()

    import fer_training.tuning.grid_search as grid_search_module

    # Tiny grid patched in so this test's runtime doesn't track the real default grid's size.
    monkeypatch.setattr(
        grid_search_module,
        "get_default_grid",
        lambda model_type: {"lr": [0.001, 0.01], "weight_decay": [0.0001]},
    )

    results = run_grid_search(
        "simple_cnn", config, train_loader, val_loader, None, trial_epochs=1, scratch_dir=tmp_path
    )

    assert len(results) == 2
    assert results[0].val_f1_macro >= results[1].val_f1_macro
    assert list(tmp_path.glob("*.pt")) == []


def test_results_to_dataframe_preserves_order():
    results = [
        TrialResult(lr=0.001, weight_decay=0.0001, val_f1_macro=0.5, val_accuracy=0.6, epochs_trained=3),
        TrialResult(lr=0.002, weight_decay=0.0001, val_f1_macro=0.7, val_accuracy=0.65, epochs_trained=2),
    ]
    df = results_to_dataframe(results)
    assert list(df["val_f1_macro"]) == [0.5, 0.7]


def test_save_results_csv_roundtrip(tmp_path):
    results = [
        TrialResult(lr=0.001, weight_decay=0.0001, val_f1_macro=0.5, val_accuracy=0.6, epochs_trained=3),
    ]
    csv_path = tmp_path / "results.csv"
    save_results_csv(results, csv_path)

    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert float(df["lr"][0]) == pytest.approx(0.001)
    assert float(df["val_f1_macro"][0]) == pytest.approx(0.5)


def test_format_results_table_is_nonempty_string():
    results = [TrialResult(lr=0.001, weight_decay=0.0001, val_f1_macro=0.5, val_accuracy=0.6, epochs_trained=3)]
    table = format_results_table(results)
    assert isinstance(table, str)
    assert "val_f1_macro" in table


def test_write_best_hyperparameters_preserves_comments_and_other_sections(tmp_path):
    original = """\
seed: 42

simple_cnn:
  epochs: 50               # a helpful comment
  lr: 0.001
  weight_decay: 0.0001
  checkpoint_name: simple_cnn.pt

resnet_finetune:
  lr: 0.0005
  weight_decay: 0.0001
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(original)

    write_best_hyperparameters("simple_cnn", lr=0.002, weight_decay=0.00001, config_path=config_path)

    raw = config_path.read_text()
    assert "# a helpful comment" in raw
    assert "checkpoint_name: simple_cnn.pt" in raw

    yaml = YAML()
    with open(config_path) as f:
        data = yaml.load(f)

    assert data["simple_cnn"]["lr"] == pytest.approx(0.002)
    assert data["simple_cnn"]["weight_decay"] == pytest.approx(0.00001)
    # untouched fields/sections preserved
    assert data["simple_cnn"]["epochs"] == 50
    assert data["resnet_finetune"]["lr"] == pytest.approx(0.0005)
    assert data["resnet_finetune"]["weight_decay"] == pytest.approx(0.0001)
