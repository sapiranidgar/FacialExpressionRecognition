"""Location of trained model checkpoints, shared by every package that needs them."""

from pathlib import Path

CHECKPOINTS_DIR = Path(__file__).resolve().parent
