"""Tests for Ray-based parallel preprocessing."""

from pathlib import Path

import pytest
import pyarrow.parquet as pq
from PIL import Image

from src.data.preprocessing import run_preprocessing


def test_preprocessing_resizes_images(
    manifest_path: str, tmp_path: Path
) -> None:
    """Processed images must be written at the configured size."""
    processed_dir = str(tmp_path / "processed")

    updated = run_preprocessing(
        manifest_path=manifest_path,
        processed_dir=processed_dir,
        fp_size=[64, 64],    # use smaller sizes to keep test fast
        iris_size=[32, 32],
    )

    rows = updated.to_pylist()
    assert len(rows) > 0

    for row in rows:
        fp_path = Path(row["processed_fp_path"])
        left_path = Path(row["processed_left_path"])
        right_path = Path(row["processed_right_path"])

        assert fp_path.exists(), f"Missing: {fp_path}"
        assert left_path.exists(), f"Missing: {left_path}"
        assert right_path.exists(), f"Missing: {right_path}"

        fp_img = Image.open(str(fp_path))
        left_img = Image.open(str(left_path))
        right_img = Image.open(str(right_path))

        assert fp_img.size == (64, 64), (
            f"fp size mismatch: {fp_img.size}"
        )
        assert left_img.size == (32, 32), (
            f"left iris size mismatch: {left_img.size}"
        )
        assert right_img.size == (32, 32), (
            f"right iris size mismatch: {right_img.size}"
        )


def test_preprocessing_updates_manifest(
    manifest_path: str, tmp_path: Path
) -> None:
    """Manifest Parquet must contain processed path columns after run."""
    processed_dir = str(tmp_path / "processed2")

    run_preprocessing(
        manifest_path=manifest_path,
        processed_dir=processed_dir,
        fp_size=[64, 64],
        iris_size=[32, 32],
    )

    table = pq.read_table(manifest_path)
    assert "processed_fp_path" in table.schema.names
    assert "processed_left_path" in table.schema.names
    assert "processed_right_path" in table.schema.names
