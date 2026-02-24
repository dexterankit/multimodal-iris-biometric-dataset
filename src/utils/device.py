"""Device resolution utility.

Priority order: CUDA (NVIDIA/AMD GPU) -> MPS (Apple Silicon) -> CPU.
"""

from __future__ import annotations

import torch

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def resolve_device() -> torch.device:
    """Select the best available compute device.

    Priority:
    1. **CUDA** -- any NVIDIA (or ROCm) GPU visible to PyTorch.
    2. **MPS** -- Apple Silicon GPU (macOS 12.3+, PyTorch 2.0+).
    3. **CPU** -- universal fallback.

    Device info (name, total VRAM) is logged at INFO level so it
    appears in all training and inference runs.

    Returns:
        A :class:`torch.device` pointing to the selected backend.

    Example::

        device = resolve_device()
        model.to(device)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        vram = torch.cuda.get_device_properties(idx).total_memory
        vram_gb = vram / (1024 ** 3)
        logger.info(
            "GPU detected -- using CUDA device %d: %s "
            "(%.1f GB VRAM)",
            idx, name, vram_gb,
        )
        return device

    # MPS is available on Apple Silicon (torch >= 2.0)
    if getattr(torch.backends, "mps", None) and \
            torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Apple Silicon GPU detected -- using MPS device")
        return device

    device = torch.device("cpu")
    logger.info("No GPU found -- falling back to CPU")
    return device
