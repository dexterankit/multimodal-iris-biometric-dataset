"""Ray-based parallel image preprocessing.

Resizes and saves all images referenced in the manifest to
``data/processed/``.  Updates the manifest Parquet file with
three new columns: ``processed_fp_path``, ``processed_left_path``,
``processed_right_path``.

Usage::

    python -m src.data.preprocessing
"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
import ray
from PIL import Image

from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


@ray.remote
def _process_row(
    row: Dict[str, Any],
    fp_size: List[int],
    iris_size: List[int],
    processed_dir: str,
) -> Dict[str, str]:
    """Ray remote task: resize one person's three images.

    Args:
        row: Manifest row dict with keys ``person_id``,
            ``fp_path``, ``left_iris_path``, ``right_iris_path``.
        fp_size: ``[height, width]`` for fingerprint images.
        iris_size: ``[height, width]`` for iris images.
        processed_dir: Root output directory.

    Returns:
        Dict mapping column name → processed image path string.
    """
    pid = row["person_id"]
    out = Path(processed_dir)

    def _resize_save(
        src: str,
        sub: str,
        size: List[int],
        grayscale: bool,
    ) -> str:
        dest_dir = out / sub
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{pid}.bmp"
        mode = "L" if grayscale else "RGB"
        img = Image.open(src).convert(mode)
        img = img.resize(
            (size[1], size[0]), Image.BILINEAR
        )
        img.save(str(dest))
        return str(dest)

    return {
        "processed_fp_path": _resize_save(
            row["fp_path"], "fingerprints",
            fp_size, grayscale=False
        ),
        "processed_left_path": _resize_save(
            row["left_iris_path"], "left",
            iris_size, grayscale=True
        ),
        "processed_right_path": _resize_save(
            row["right_iris_path"], "right",
            iris_size, grayscale=True
        ),
    }


def run_preprocessing(
    manifest_path: str = "data/manifest.parquet",
    processed_dir: str = "data/processed",
    fp_size: List[int] | None = None,
    iris_size: List[int] | None = None,
) -> pa.Table:
    """Fan out preprocessing across all manifest rows via Ray.

    Args:
        manifest_path: Path to existing Parquet manifest.
        processed_dir: Destination root for processed images.
        fp_size: ``[h, w]`` target for fingerprint (default
            ``[128, 128]``).
        iris_size: ``[h, w]`` target for iris (default
            ``[64, 64]``).

    Returns:
        Updated PyArrow Table persisted back to *manifest_path*.
    """
    if fp_size is None:
        fp_size = [128, 128]
    if iris_size is None:
        iris_size = [64, 64]

    table = pq.read_table(manifest_path)
    rows = table.to_pylist()
    n = len(rows)
    logger.info(
        "Starting Ray preprocessing for %d persons…", n
    )

    ray.init(ignore_reinit_error=True)
    t0 = time.perf_counter()
    try:
        futures = [
            _process_row.remote(row, fp_size, iris_size, processed_dir)
            for row in rows
        ]
        results = ray.get(futures)
    finally:
        ray.shutdown()

    elapsed = time.perf_counter() - t0
    n_images = n * 3  # fp + left + right
    throughput = n_images / elapsed
    logger.info(
        "Preprocessing done: %d images in %.2fs "
        "(%.1f img/s)",
        n_images, elapsed, throughput,
    )

    # Append new columns to table
    updated = table.append_column(
        "processed_fp_path",
        pa.array([r["processed_fp_path"] for r in results]),
    ).append_column(
        "processed_left_path",
        pa.array([r["processed_left_path"] for r in results]),
    ).append_column(
        "processed_right_path",
        pa.array([r["processed_right_path"] for r in results]),
    )

    pq.write_table(updated, manifest_path)
    logger.info(
        "Manifest updated with processed paths: %s",
        manifest_path,
    )
    return updated


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Parallel image preprocessing via Ray."
    )
    parser.add_argument(
        "--manifest-path", default="data/manifest.parquet"
    )
    parser.add_argument(
        "--processed-dir", default="data/processed"
    )
    parser.add_argument(
        "--fp-size", type=int, nargs=2,
        default=[128, 128], metavar=("H", "W")
    )
    parser.add_argument(
        "--iris-size", type=int, nargs=2,
        default=[64, 64], metavar=("H", "W")
    )
    args = parser.parse_args()
    run_preprocessing(
        manifest_path=args.manifest_path,
        processed_dir=args.processed_dir,
        fp_size=args.fp_size,
        iris_size=args.iris_size,
    )
