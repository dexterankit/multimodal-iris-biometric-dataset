"""Tests for BiometricDataset and build_dataloaders."""

import pytest
import torch

from src.data.dataset import BiometricDataset, build_dataloaders


def test_dataset_len(manifest_path: str) -> None:
    """Dataset length equals number of rows in manifest."""
    import pyarrow.parquet as pq

    table = pq.read_table(manifest_path)
    rows = table.to_pylist()

    ds = BiometricDataset(rows, fp_size=(128, 128), iris_size=(64, 64))
    assert len(ds) == len(rows)


def test_dataset_item_shapes(manifest_path: str) -> None:
    """__getitem__ returns correctly shaped tensors."""
    import pyarrow.parquet as pq

    table = pq.read_table(manifest_path)
    rows = table.to_pylist()
    ds = BiometricDataset(rows, fp_size=(128, 128), iris_size=(64, 64))

    fp, left, right, label = ds[0]

    assert isinstance(fp, torch.Tensor)
    assert fp.shape == (3, 128, 128), f"fp shape: {fp.shape}"
    assert left.shape == (1, 64, 64), f"left iris shape: {left.shape}"
    assert right.shape == (1, 64, 64), f"right iris shape: {right.shape}"


def test_dataset_label_range(manifest_path: str) -> None:
    """Labels are 0-indexed integers within [0, 44]."""
    import pyarrow.parquet as pq

    table = pq.read_table(manifest_path)
    rows = table.to_pylist()
    ds = BiometricDataset(rows, fp_size=(128, 128), iris_size=(64, 64))

    for i in range(len(ds)):
        _, _, _, label = ds[i]
        assert isinstance(label, int)
        assert 0 <= label <= 44, f"label out of range: {label}"


def test_build_dataloaders(manifest_path: str) -> None:
    """build_dataloaders returns non-empty loaders."""
    train_loader, val_loader = build_dataloaders(
        manifest_path=manifest_path,
        fp_size=(128, 128),
        iris_size=(64, 64),
        batch_size=2,
        num_workers=0,
    )
    assert len(train_loader.dataset) > 0
    assert len(val_loader.dataset) > 0


def test_dataloader_batch_shape(manifest_path: str) -> None:
    """DataLoader yields correct batch tensor shapes."""
    train_loader, _ = build_dataloaders(
        manifest_path=manifest_path,
        fp_size=(128, 128),
        iris_size=(64, 64),
        batch_size=2,
        num_workers=0,
    )
    fp, left, right, labels = next(iter(train_loader))
    B = fp.shape[0]
    assert fp.shape == (B, 3, 128, 128)
    assert left.shape == (B, 1, 64, 64)
    assert right.shape == (B, 1, 64, 64)
    assert labels.shape == (B,)
