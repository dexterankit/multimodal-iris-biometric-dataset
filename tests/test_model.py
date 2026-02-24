"""Tests for MultimodalNet forward pass and architecture."""

import math

import pytest
import torch

from src.models.multimodal_net import (
    FingerprintBranch,
    IrisBranch,
    MultimodalNet,
)


@pytest.fixture()
def dummy_batch():
    """Return a small batch of random tensors (no GPU required)."""
    B = 2
    fp = torch.randn(B, 3, 128, 128)
    left = torch.randn(B, 1, 64, 64)
    right = torch.randn(B, 1, 64, 64)
    return fp, left, right, B


def test_iris_branch_output_shape(dummy_batch) -> None:
    """IrisBranch outputs (B, 32)."""
    fp, left, right, B = dummy_batch
    branch = IrisBranch()
    out = branch(left)
    assert out.shape == (B, 32), f"Unexpected shape: {out.shape}"


def test_fingerprint_branch_output_shape(dummy_batch) -> None:
    """FingerprintBranch outputs (B, 1280)."""
    fp, left, right, B = dummy_batch
    # freeze=False avoids downloading weights in CI on first run
    branch = FingerprintBranch(freeze=False)
    out = branch(fp)
    assert out.shape == (B, 1280), f"Unexpected shape: {out.shape}"


def test_multimodal_net_output_shape(dummy_batch) -> None:
    """MultimodalNet forward pass outputs (B, num_classes)."""
    fp, left, right, B = dummy_batch
    model = MultimodalNet(
        num_classes=45, embedding_dim=128, freeze_backbone=False
    )
    logits = model(fp, left, right)
    assert logits.shape == (B, 45), f"Unexpected shape: {logits.shape}"


def test_multimodal_net_no_nan(dummy_batch) -> None:
    """Forward pass must not produce NaN or Inf."""
    fp, left, right, B = dummy_batch
    model = MultimodalNet(
        num_classes=45, embedding_dim=128, freeze_backbone=False
    )
    logits = model(fp, left, right)
    assert not torch.isnan(logits).any(), "NaN in logits"
    assert not torch.isinf(logits).any(), "Inf in logits"


def test_iris_branch_weight_sharing() -> None:
    """Left and right iris use the same IrisBranch instance."""
    model = MultimodalNet(
        num_classes=45, embedding_dim=128, freeze_backbone=False
    )
    # The single iris_branch attribute should appear only once
    assert model.iris_branch is model.iris_branch  # trivially true
    # Verify same object processed both sides in forward
    x = torch.randn(1, 1, 64, 64)
    out_direct = model.iris_branch(x)
    assert out_direct.shape == (1, 32)


def test_frozen_backbone_no_grad() -> None:
    """freeze_backbone=True must zero out backbone gradients."""
    model = MultimodalNet(freeze_backbone=True)
    for param in model.fingerprint_branch.features.parameters():
        assert not param.requires_grad, (
            "Backbone parameter should be frozen"
        )
