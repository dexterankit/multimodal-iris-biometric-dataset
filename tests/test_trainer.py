"""Tests for the Trainer class."""

import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.data.dataset import build_dataloaders
from src.models.multimodal_net import MultimodalNet
from src.training.trainer import Trainer


@pytest.fixture()
def minimal_cfg(tmp_path: Path):
    """Minimal OmegaConf training config."""
    return OmegaConf.create(
        {
            "lr": 0.001,
            "epochs": 2,
            "batch_size": 2,
            "seed": 42,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        }
    )


@pytest.fixture()
def trainer_and_loaders(manifest_path: str, minimal_cfg, tmp_path):
    """Construct a Trainer with synthetic data."""
    train_loader, val_loader = build_dataloaders(
        manifest_path=manifest_path,
        fp_size=(128, 128),
        iris_size=(64, 64),
        batch_size=2,
        num_workers=0,
    )
    model = MultimodalNet(
        num_classes=45, embedding_dim=64, freeze_backbone=False
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=minimal_cfg,
        metrics_path=str(tmp_path / "metrics.jsonl"),
    )
    return trainer, train_loader, val_loader


def test_train_epoch_returns_finite_loss(
    trainer_and_loaders,
) -> None:
    """train_epoch() must return a finite loss."""
    trainer, _, _ = trainer_and_loaders
    metrics = trainer.train_epoch()
    assert "train_loss" in metrics
    assert math.isfinite(metrics["train_loss"])


def test_validate_returns_finite_loss(
    trainer_and_loaders,
) -> None:
    """validate() must return finite val_loss and val_acc."""
    trainer, _, _ = trainer_and_loaders
    metrics = trainer.validate()
    assert math.isfinite(metrics["val_loss"])
    assert 0.0 <= metrics["val_acc"] <= 1.0


def test_checkpoint_save_and_load(
    trainer_and_loaders, tmp_path: Path
) -> None:
    """Checkpoint round-trip preserves model weights and epoch."""
    trainer, _, _ = trainer_and_loaders
    ckpt_path = str(tmp_path / "test.pt")

    # Save
    trainer.save_checkpoint(ckpt_path, epoch=1)
    assert Path(ckpt_path).exists()

    # Mutate model weights
    for p in trainer.model.parameters():
        p.data.fill_(0.0)

    # Load and verify restoration
    epoch = trainer.load_checkpoint(ckpt_path)
    assert epoch == 1
    # At least one parameter should be non-zero now
    any_nonzero = any(
        p.abs().sum().item() > 0
        for p in trainer.model.parameters()
    )
    assert any_nonzero, "Model weights not restored from checkpoint"


def test_metrics_jsonl_written(
    trainer_and_loaders, tmp_path: Path
) -> None:
    """fit() must append JSON lines to metrics.jsonl."""
    trainer, _, _ = trainer_and_loaders
    trainer.fit()

    metrics_file = tmp_path / "metrics.jsonl"
    assert metrics_file.exists()
    records = trainer.tracker.read_all()
    assert len(records) == 2  # cfg.epochs == 2
    assert "train_loss" in records[0]
    assert "val_acc" in records[0]
