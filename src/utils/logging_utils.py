"""Logging utilities: configure a single root logger for the project."""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """Configure and return the root logger.

    Sets up a stream handler (stdout) and, optionally, a
    rotating file handler. Safe to call multiple times — handlers
    are only added once.

    Args:
        level: Logging level (default: logging.INFO).
        log_file: Optional path to a log file. Parent dirs are
            created automatically.

    Returns:
        Configured root logger.
    """
    logger = logging.getLogger()
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    # File handler (optional)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger.

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        Named logger instance.
    """
    return logging.getLogger(name)
