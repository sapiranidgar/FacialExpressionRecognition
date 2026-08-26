"""Downloads the FER2013 dataset from Kaggle into the local data directory."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from fer.constants import (
    ENV_FILE_NAME,
    KAGGLE_CONFIG_DIR_NAME,
    KAGGLE_CONFIG_FILE_NAME,
    KAGGLE_DATASET,
    KAGGLE_KEY_ENV_VAR,
    KAGGLE_USERNAME_ENV_VAR,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
KAGGLE_CREDENTIALS_PATH = Path.home() / KAGGLE_CONFIG_DIR_NAME / KAGGLE_CONFIG_FILE_NAME

load_dotenv(REPO_ROOT / ENV_FILE_NAME)


def _has_kaggle_credentials() -> bool:
    """Check whether Kaggle credentials are available via env vars or kaggle.json.

    Returns:
        True if both KAGGLE_USERNAME and KAGGLE_KEY are set in the environment
        (including via a loaded .env file), or a kaggle.json file exists at
        the standard location.
    """
    has_env_vars = bool(os.environ.get(KAGGLE_USERNAME_ENV_VAR)) and bool(os.environ.get(KAGGLE_KEY_ENV_VAR))
    return has_env_vars or KAGGLE_CREDENTIALS_PATH.exists()


def _check_kaggle_credentials() -> None:
    """Raise a SystemExit with setup instructions if Kaggle credentials are missing.

    Raises:
        SystemExit: If no credentials are found, with instructions for
            setting up a .env file or kaggle.json.
    """
    if not _has_kaggle_credentials():
        raise SystemExit(
            "Kaggle API credentials not found.\n\n"
            "The kaggle package authenticates with a username + key pair, "
            "read from (in order): environment variables, a .env file, or a "
            "kaggle.json file.\n\n"
            f"Create a file named {ENV_FILE_NAME} in the repo root ({REPO_ROOT}) - "
            "it's already covered by .gitignore - containing:\n\n"
            f"    {KAGGLE_USERNAME_ENV_VAR}=your-kaggle-username\n"
            f"    {KAGGLE_KEY_ENV_VAR}=your-key-from-kaggle.com/settings/api\n\n"
            "Then re-run this script. See .env.example for the template.\n"
        )


def download_fer2013(data_dir: Path) -> None:
    """Download and unzip the FER2013 dataset from Kaggle into `data_dir`.

    No-ops if `data_dir` already contains files, so re-running this is safe.

    Args:
        data_dir: Destination directory. Created if it doesn't exist.

    Raises:
        SystemExit: If Kaggle credentials are not configured.
    """
    _check_kaggle_credentials()
    data_dir.mkdir(parents=True, exist_ok=True)

    if any(data_dir.iterdir()):
        print(f"{data_dir} is not empty, skipping download. Delete it to re-download.")
        return

    # Imported lazily: the kaggle package authenticates at import time and
    # raises if credentials are missing, so this must happen after the check
    # above (which has already loaded .env into the environment for it to find).
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(data_dir), unzip=True, quiet=False)
    print(f"Downloaded {KAGGLE_DATASET} into {data_dir}")
