import torch

from fer.models.resnet_finetune import build_resnet18_finetune
from fer.models.simple_cnn import SimpleCNN


def test_simple_cnn_output_shape():
    model = SimpleCNN(num_classes=7, in_channels=1)
    x = torch.randn(4, 1, 48, 48)
    out = model(x)
    assert out.shape == (4, 7)


def test_resnet_finetune_output_shape():
    model = build_resnet18_finetune(num_classes=7, unfrozen_layers=["layer4", "fc"])
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 7)


def test_resnet_finetune_freezes_expected_layers():
    model = build_resnet18_finetune(num_classes=7, unfrozen_layers=["layer4", "fc"])
    for name, param in model.named_parameters():
        expected_trainable = name.startswith("layer4") or name.startswith("fc")
        assert param.requires_grad == expected_trainable, name
