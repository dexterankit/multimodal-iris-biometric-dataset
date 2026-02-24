"""Shared pytest fixtures for the test suite.

All fixtures generate synthetic data in temporary directories so no
real dataset is required during CI.
"""

import random
from pathlib import Path

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _make_bmp(path: Path, size: tuple, mode: str = "RGB") -> Path:
    """Create a single synthetic BMP image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new(mode, size, color=0)
    img.save(str(path))
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_dataset_dir(tmp_path: Path) -> Path:
    """Return a tiny dataset root with 5 persons (synthetic BMP images).

    Structure mirrors the real dataset:
        <root>/<person_id>/Fingerprint/*.BMP
        <root>/<person_id>/left/*.bmp
        <root>/<person_id>/right/*.bmp
    """
    rng = random.Random(0)
    for pid in range(1, 6):
        person_dir = tmp_path / str(pid)
        _make_bmp(person_dir / "Fingerprint" / "fp.BMP", (128, 128))
        _make_bmp(person_dir / "left" / "l.bmp", (64, 64), mode="L")
        _make_bmp(person_dir / "right" / "r.bmp", (64, 64), mode="L")
    return tmp_path


@pytest.fixture()
def manifest_path(synthetic_dataset_dir: Path, tmp_path: Path) -> str:
    """Build and return the path to a 5-row manifest Parquet."""
    from src.data.manifest import build_manifest

    out = str(tmp_path / "manifest.parquet")
    build_manifest(
        base_path=str(synthetic_dataset_dir),
        manifest_path=out,
        val_split=0.2,
        seed=42,
    )
    return out


@pytest.fixture()
def synthetic_images(tmp_path: Path):
    """Return paths to one synthetic fp / left / right BMP."""
    fp = _make_bmp(tmp_path / "fp.BMP", (128, 128))
    left = _make_bmp(tmp_path / "left.bmp", (64, 64), mode="L")
    right = _make_bmp(tmp_path / "right.bmp", (64, 64), mode="L")
    return str(fp), str(left), str(right)
