"""Location of the downloaded dataset, shared by every package that needs it."""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
