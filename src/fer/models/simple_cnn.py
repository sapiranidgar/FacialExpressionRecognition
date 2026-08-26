"""Defines SimpleCNN, the from-scratch convolutional classifier."""

from __future__ import annotations

import torch
import torch.nn as nn

from fer.constants import (
    SIMPLE_CNN_CHANNELS,
    SIMPLE_CNN_CONV_KERNEL_SIZE,
    SIMPLE_CNN_CONV_PADDING,
    SIMPLE_CNN_DEFAULT_DROPOUT,
    SIMPLE_CNN_POOL_SIZE,
)


class SimpleCNN(nn.Module):
    """A small from-scratch CNN for emotion classification.

    Stacks one conv block per entry in SIMPLE_CNN_CHANNELS (each block halving
    spatial resolution via max pooling), then global-average-pools and applies
    a dropout + linear classifier head. Designed to train quickly on CPU for
    low-resolution grayscale input.
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 1,
        dropout: float = SIMPLE_CNN_DEFAULT_DROPOUT,
    ):
        """Initialize the network.

        Args:
            num_classes: Number of output emotion classes.
            in_channels: Number of input image channels (1 for grayscale).
            dropout: Dropout probability applied before the final linear layer.
        """
        super().__init__()
        channels = [in_channels, *SIMPLE_CNN_CHANNELS]
        self.features = nn.Sequential(
            *[self._conv_block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(SIMPLE_CNN_CHANNELS[-1], num_classes),
        )

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        """Build one conv -> batchnorm -> relu -> maxpool block.

        Args:
            in_ch: Number of input channels.
            out_ch: Number of output channels.

        Returns:
            The assembled block as an nn.Sequential.
        """
        return nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=SIMPLE_CNN_CONV_KERNEL_SIZE,
                padding=SIMPLE_CNN_CONV_PADDING,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(SIMPLE_CNN_POOL_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input batch of shape (batch, in_channels, H, W).

        Returns:
            Class logits of shape (batch, num_classes).
        """
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
