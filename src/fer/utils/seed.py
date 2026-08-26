"""Seeds every source of randomness used during training for reproducibility."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python's random module, numpy, and torch with the same value.

    Args:
        seed: The seed value to apply to all three random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
