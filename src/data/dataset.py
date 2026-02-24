"""PyTorch Dataset and DataLoader factories for the biometric data."""

from pathlib import Path
from typing import Tuple

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ImageNet normalisation stats (fingerprint branch uses MobileNetV2)
_FP_MEAN = [0.485, 0.456, 0.406]
_FP_STD = [0.229, 0.224, 0.225]

# Standard grayscale normalisation for iris
_IRIS_MEAN = [0.5]
_IRIS_STD = [0.5]


class BiometricDataset(Dataset):
    """Multimodal biometric dataset.

    Each item is a triplet of pre-processed images belonging to one
    person: fingerprint (RGB), left iris (grayscale), right iris
    (grayscale), plus an integer class label (0-indexed).

    Args:
        manifest_rows: List of row dicts from the Parquet manifest,
            already filtered to the desired split.
        fp_size: ``(H, W)`` for fingerprint images.
        iris_size: ``(H, W)`` for iris images.
    """

    def __init__(
        self,
        manifest_rows: list,
        fp_size: Tuple[int, int] = (128, 128),
        iris_size: Tuple[int, int] = (64, 64),
    ) -> None:
        self._rows = manifest_rows
        self._fp_tf = transforms.Compose(
            [
                transforms.Resize(fp_size),
                transforms.ToTensor(),
                transforms.Normalize(_FP_MEAN, _FP_STD),
            ]
        )
        self._iris_tf = transforms.Compose(
            [
                transforms.Resize(iris_size),
                transforms.ToTensor(),
                transforms.Normalize(_IRIS_MEAN, _IRIS_STD),
            ]
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return (fp_tensor, left_tensor, right_tensor, label).

        The label is ``person_id - 1`` (zero-indexed, range 0–44).

        Args:
            idx: Sample index.

        Returns:
            Tuple of three tensors and an integer label.
        """
        row = self._rows[idx]

        # Prefer processed paths; fall back to raw paths
        fp_key = "processed_fp_path" if "processed_fp_path" in row \
            else "fp_path"
        left_key = "processed_left_path" \
            if "processed_left_path" in row else "left_iris_path"
        right_key = "processed_right_path" \
            if "processed_right_path" in row else "right_iris_path"

        fp_img = Image.open(row[fp_key]).convert("RGB")
        left_img = Image.open(row[left_key]).convert("L")
        right_img = Image.open(row[right_key]).convert("L")

        fp_tensor = self._fp_tf(fp_img)
        left_tensor = self._iris_tf(left_img)
        right_tensor = self._iris_tf(right_img)

        label = int(row["person_id"]) - 1  # 0-indexed
        return fp_tensor, left_tensor, right_tensor, label


def build_dataloaders(
    manifest_path: str,
    fp_size: Tuple[int, int] = (128, 128),
    iris_size: Tuple[int, int] = (64, 64),
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from the manifest.

    Args:
        manifest_path: Path to the Parquet manifest file.
        fp_size: Target ``(H, W)`` for fingerprint images.
        iris_size: Target ``(H, W)`` for iris images.
        batch_size: Samples per mini-batch.
        num_workers: DataLoader worker processes.
        pin_memory: Pin memory for faster GPU transfer.

    Returns:
        Tuple ``(train_loader, val_loader)``.
    """
    if not Path(manifest_path).exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. "
            "Run `python -m src.data.manifest` first."
        )

    table = pq.read_table(manifest_path)
    rows = table.to_pylist()

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]

    logger.info(
        "Dataset split — train: %d, val: %d",
        len(train_rows), len(val_rows),
    )

    train_ds = BiometricDataset(train_rows, fp_size, iris_size)
    val_ds = BiometricDataset(val_rows, fp_size, iris_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
