"""Lightweight experiment metrics tracker.

Appends one JSON line per epoch to ``metrics.jsonl`` so results are
human-readable and trivially parseable without any external tracking
dependency.
"""

import json
from pathlib import Path
from typing import Any, Dict

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MetricsTracker:
    """Append per-epoch metrics to a ``.jsonl`` file.

    Each call to :meth:`log` writes one JSON-serialisable dict as
    a newline-terminated record.

    Args:
        output_path: Destination ``.jsonl`` file.  Parent directories
            are created automatically.
    """

    def __init__(self, output_path: str) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("MetricsTracker writing to %s", self._path)

    def log(self, record: Dict[str, Any]) -> None:
        """Append *record* as a JSON line.

        Args:
            record: Arbitrary serialisable dict
                (e.g. ``{"epoch": 1, "train_loss": 0.5}``).
        """
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def read_all(self) -> list:
        """Return all logged records.

        Returns:
            List of dicts in insertion order.
        """
        if not self._path.exists():
            return []
        with open(self._path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
