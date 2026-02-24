"""Training loop with checkpoint save/load and metrics tracking."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.models.multimodal_net import MultimodalNet
from src.training.metrics import MetricsTracker
from src.utils.device import resolve_device
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Trainer:
    """Encapsulates the full training and validation loop.

    Args:
        model: :class:`~src.models.multimodal_net.MultimodalNet`.
        train_loader: Training :class:`~torch.utils.data.DataLoader`.
        val_loader: Validation :class:`~torch.utils.data.DataLoader`.
        cfg: Hydra training config (``cfg.training``).
        metrics_path: Path to ``metrics.jsonl`` output file.
    """

    def __init__(
        self,
        model: MultimodalNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        metrics_path: str = "outputs/metrics.jsonl",
    ) -> None:
        self.device = resolve_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(cfg.lr),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.tracker = MetricsTracker(metrics_path)

        self._best_val_acc: float = -float("inf")
        checkpoint_dir = Path(cfg.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._best_ckpt = checkpoint_dir / "best.pt"

        logger.info(
            "Trainer initialised — device=%s, lr=%s, epochs=%d",
            self.device, cfg.lr, cfg.epochs,
        )

    # ------------------------------------------------------------------
    # Core loop helpers
    # ------------------------------------------------------------------

    def _forward_batch(
        self, batch: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Move batch to device and run forward pass.

        Args:
            batch: ``(fp, left, right, labels)`` from DataLoader.

        Returns:
            Tuple ``(logits, labels)`` on device.
        """
        fp, left, right, labels = batch
        fp = fp.to(self.device)
        left = left.to(self.device)
        right = right.to(self.device)
        labels = labels.to(self.device)
        logits = self.model(fp, left, right)
        return logits, labels

    @staticmethod
    def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute top-1 accuracy.

        Args:
            logits: Raw model output ``(B, C)``.
            labels: Ground-truth integer labels ``(B,)``.

        Returns:
            Accuracy in ``[0, 1]``.
        """
        preds = logits.argmax(dim=1)
        return (preds == labels).float().mean().item()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        """Run one full training epoch.

        Returns:
            Dict with keys ``train_loss`` and ``train_acc``.
        """
        self.model.train()
        total_loss, total_acc, n_batches = 0.0, 0.0, 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()
            logits, labels = self._forward_batch(batch)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_acc += self._accuracy(logits, labels)
            n_batches += 1

        return {
            "train_loss": total_loss / max(n_batches, 1),
            "train_acc": total_acc / max(n_batches, 1),
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run one full validation pass.

        Returns:
            Dict with keys ``val_loss`` and ``val_acc``.
        """
        self.model.eval()
        total_loss, total_acc, n_batches = 0.0, 0.0, 0

        for batch in self.val_loader:
            logits, labels = self._forward_batch(batch)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            total_acc += self._accuracy(logits, labels)
            n_batches += 1

        return {
            "val_loss": total_loss / max(n_batches, 1),
            "val_acc": total_acc / max(n_batches, 1),
        }

    def fit(self) -> None:
        """Train for ``cfg.epochs`` epochs, saving the best checkpoint."""
        epochs = int(self.cfg.epochs)
        logger.info("Starting training for %d epochs", epochs)

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            elapsed = time.perf_counter() - t0

            record = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "epoch_time_s": round(elapsed, 3),
            }
            self.tracker.log(record)

            logger.info(
                "Epoch %3d/%d — loss: %.4f | acc: %.4f | "
                "val_loss: %.4f | val_acc: %.4f  (%.1fs)",
                epoch, epochs,
                record["train_loss"], record["train_acc"],
                record["val_loss"], record["val_acc"],
                elapsed,
            )

            if record["val_acc"] > self._best_val_acc:
                self._best_val_acc = record["val_acc"]
                self.save_checkpoint(str(self._best_ckpt), epoch)
                logger.info(
                    "  -> New best val_acc=%.4f, checkpoint saved",
                    self._best_val_acc,
                )

        logger.info(
            "Training complete. Best val_acc=%.4f, "
            "checkpoint at %s",
            self._best_val_acc, self._best_ckpt,
        )

    def save_checkpoint(self, path: str, epoch: int) -> None:
        """Persist model and optimiser states plus config.

        Args:
            path: Destination ``.pt`` file path.
            epoch: Current epoch number (stored in checkpoint).
        """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_acc": self._best_val_acc,
                "config": OmegaConf.to_container(
                    self.cfg, resolve=True
                ),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> int:
        """Load model and optimiser states from a checkpoint.

        Args:
            path: Path to a ``.pt`` checkpoint file.

        Returns:
            Epoch number stored in the checkpoint.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self._best_val_acc = ckpt.get("val_acc", 0.0)
        logger.info(
            "Checkpoint loaded from %s (epoch %d, val_acc=%.4f)",
            path, ckpt["epoch"], self._best_val_acc,
        )
        return ckpt["epoch"]
