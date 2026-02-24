"""Reproducibility utilities: set global random seeds."""

import random

import numpy as np
import torch

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for full reproducibility.

    Covers Python's ``random``, NumPy, PyTorch (CPU + CUDA) and
    enables cuDNN determinism mode.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)
