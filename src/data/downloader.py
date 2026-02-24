"""Kaggle dataset downloader — idempotent wrapper around the Kaggle CLI.

Usage::

    python -m src.data.downloader
"""

import os
import subprocess
import sys
from pathlib import Path

from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

_DATASET = "ninadmehendale/multimodal-iris-fingerprint-biometric-data"
_DEST = "data/raw"
_MARKER = "IRIS and FINGERPRINT DATASET"


def download_dataset(dest: str = _DEST) -> None:
    """Download and unzip the Kaggle dataset if not already present.

    Args:
        dest: Local directory to unzip the dataset into.

    Raises:
        EnvironmentError: If ``~/.kaggle/kaggle.json`` is missing.
        subprocess.CalledProcessError: If the Kaggle CLI fails.
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise EnvironmentError(
            f"Kaggle credentials not found at {kaggle_json}.\n"
            "Steps to fix:\n"
            "  1. Go to https://www.kaggle.com/settings/account\n"
            "  2. API section -> 'Create New Token'\n"
            "  3. Place the downloaded kaggle.json at "
            f"{kaggle_json}"
        )

    marker = Path(dest) / _MARKER
    if marker.exists():
        logger.info(
            "Dataset already present at %s — skipping download.",
            marker,
        )
        return

    Path(dest).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "kaggle",
        "datasets", "download",
        "-d", _DATASET,
        "--unzip",
        "-p", dest,
    ]
    logger.info("Downloading dataset: %s", _DATASET)
    logger.info("Command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("Download complete -> %s", dest)


if __name__ == "__main__":
    setup_logging()
    download_dataset()
