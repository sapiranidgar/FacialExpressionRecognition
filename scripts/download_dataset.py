"""Downloads the FER2013 dataset into the path configured in configs/config.yaml."""

from fer.config import load_config
from fer.data.download import download_fer2013


def main() -> None:
    """Load config and download FER2013 into config.paths.data_dir."""
    config = load_config()
    download_fer2013(config.paths.data_dir)


if __name__ == "__main__":
    main()
