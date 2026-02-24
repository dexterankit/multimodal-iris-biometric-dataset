"""Inference pipeline: load a checkpoint and predict person identity."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms

from src.models.multimodal_net import MultimodalNet
from src.utils.device import resolve_device
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_FP_MEAN = [0.485, 0.456, 0.406]
_FP_STD = [0.229, 0.224, 0.225]
_IRIS_MEAN = [0.5]
_IRIS_STD = [0.5]


class Predictor:
    """Load a trained checkpoint and run single-sample inference.

    Args:
        ckpt_path: Path to a ``.pt`` checkpoint produced by
            :class:`~src.training.trainer.Trainer`.
    """

    def __init__(self, ckpt_path: str) -> None:
        self.device = resolve_device()
        ckpt = torch.load(
            ckpt_path, map_location=self.device
        )

        # Reconstruct model from saved training config
        cfg = ckpt.get("config", {})
        self.model = MultimodalNet(
            num_classes=cfg.get("num_classes", 45),
            embedding_dim=cfg.get("embedding_dim", 128),
            freeze_backbone=False,  # not needed at inference
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

        # Store image sizes from config (fall back to defaults)
        self._fp_size: Tuple[int, int] = tuple(
            cfg.get("fp_size", [128, 128])
        )
        self._iris_size: Tuple[int, int] = tuple(
            cfg.get("iris_size", [64, 64])
        )

        self._fp_tf = transforms.Compose(
            [
                transforms.Resize(self._fp_size),
                transforms.ToTensor(),
                transforms.Normalize(_FP_MEAN, _FP_STD),
            ]
        )
        self._iris_tf = transforms.Compose(
            [
                transforms.Resize(self._iris_size),
                transforms.ToTensor(),
                transforms.Normalize(_IRIS_MEAN, _IRIS_STD),
            ]
        )

        logger.info(
            "Predictor loaded from %s (epoch %d, val_acc=%.4f)",
            ckpt_path,
            ckpt.get("epoch", -1),
            ckpt.get("val_acc", 0.0),
        )

    @torch.no_grad()
    def predict(
        self,
        fp_path: str,
        left_iris_path: str,
        right_iris_path: str,
    ) -> int:
        """Predict person ID (1-indexed) from image paths.

        Args:
            fp_path: Path to a fingerprint image.
            left_iris_path: Path to a left iris image.
            right_iris_path: Path to a right iris image.

        Returns:
            Predicted person ID in range ``[1, num_classes]``.
        """
        for p in (fp_path, left_iris_path, right_iris_path):
            if not Path(p).exists():
                raise FileNotFoundError(f"Image not found: {p}")

        fp = self._fp_tf(
            Image.open(fp_path).convert("RGB")
        ).unsqueeze(0).to(self.device)

        left = self._iris_tf(
            Image.open(left_iris_path).convert("L")
        ).unsqueeze(0).to(self.device)

        right = self._iris_tf(
            Image.open(right_iris_path).convert("L")
        ).unsqueeze(0).to(self.device)

        logits = self.model(fp, left, right)
        person_id: int = logits.argmax(dim=1).item() + 1
        logger.info("Predicted person_id = %d", person_id)
        return person_id
