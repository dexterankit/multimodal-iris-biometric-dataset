"""Build and persist a PyArrow Parquet manifest of the raw dataset.

Usage::

    python -m src.data.manifest                  # build (skip if exists)
    python -m src.data.manifest --rebuild         # force regeneration

The manifest is a 45-row Parquet table with columns:
    person_id, fp_path, left_iris_path, right_iris_path, split
"""

import argparse
import random
from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

# Column schema
SCHEMA = pa.schema(
    [
        pa.field("person_id", pa.int32()),
        pa.field("fp_path", pa.string()),
        pa.field("left_iris_path", pa.string()),
        pa.field("right_iris_path", pa.string()),
        pa.field("split", pa.string()),
    ]
)


def _find_bmp(folder: Path, pick: str, rng: random.Random) -> Path:
    """Return a .bmp/.BMP file from *folder*.

    Args:
        folder: Directory to search.
        pick: ``"first"`` for alphabetical first, ``"random"`` for
            a reproducibly random choice.
        rng: Seeded :class:`random.Random` instance.

    Returns:
        Path to selected image.

    Raises:
        FileNotFoundError: If no BMP files are found.
    """
    files: List[Path] = sorted(
        [f for f in folder.iterdir()
         if f.suffix.lower() == ".bmp"]
    )
    if not files:
        raise FileNotFoundError(
            f"No .bmp files found in {folder}"
        )
    if pick == "random":
        return rng.choice(files)
    return files[0]  # "first"


def build_manifest(
    base_path: str,
    manifest_path: str,
    val_split: float = 0.2,
    seed: int = 42,
    fp_pick: str = "first",
    iris_pick: str = "first",
) -> pa.Table:
    """Scan the raw dataset directory and write a Parquet manifest.

    Args:
        base_path: Root containing per-person subdirectories
            (e.g. ``data/raw/IRIS and FINGERPRINT DATASET``).
        manifest_path: Destination ``.parquet`` file path.
        val_split: Fraction of persons assigned to validation.
        seed: RNG seed for split assignment and file picking.
        fp_pick: How to select fingerprint image
            (``"first"`` or ``"random"``).
        iris_pick: How to select iris image per side
            (``"first"`` or ``"random"``).

    Returns:
        PyArrow Table written to *manifest_path*.
    """
    base = Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"Base path not found: {base}")

    rng = random.Random(seed)

    person_dirs = sorted(
        [d for d in base.iterdir() if d.is_dir()
         and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    logger.info(
        "Found %d person directories under %s",
        len(person_dirs), base
    )

    # Build per-person records
    records = []
    for person_dir in person_dirs:
        pid = int(person_dir.name)
        try:
            fp_path = _find_bmp(
                person_dir / "Fingerprint", fp_pick, rng
            )
            left_path = _find_bmp(
                person_dir / "left", iris_pick, rng
            )
            right_path = _find_bmp(
                person_dir / "right", iris_pick, rng
            )
        except FileNotFoundError as exc:
            logger.warning("Skipping person %d: %s", pid, exc)
            continue

        records.append(
            {
                "person_id": pid,
                "fp_path": str(fp_path),
                "left_iris_path": str(left_path),
                "right_iris_path": str(right_path),
            }
        )

    if not records:
        raise ValueError("No valid records found.")

    # Reproducible stratified split
    pids = [r["person_id"] for r in records]
    rng2 = random.Random(seed)
    shuffled = pids[:]
    rng2.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_split))
    val_set = set(shuffled[:n_val])
    for r in records:
        r["split"] = "val" if r["person_id"] in val_set else "train"

    n_val_actual = sum(1 for r in records if r["split"] == "val")
    logger.info(
        "Split: %d train, %d val",
        len(records) - n_val_actual, n_val_actual
    )

    table = pa.table(
        {
            "person_id": pa.array(
                [r["person_id"] for r in records], pa.int32()
            ),
            "fp_path": [r["fp_path"] for r in records],
            "left_iris_path": [r["left_iris_path"] for r in records],
            "right_iris_path": [r["right_iris_path"] for r in records],
            "split": [r["split"] for r in records],
        },
        schema=SCHEMA,
    )

    out = Path(manifest_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    logger.info("Manifest written to %s (%d rows)", out, len(table))
    return table


def load_manifest(manifest_path: str) -> pa.Table:
    """Load an existing Parquet manifest.

    Args:
        manifest_path: Path to the ``.parquet`` file.

    Returns:
        PyArrow Table.

    Raises:
        FileNotFoundError: If the manifest does not exist.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}. "
            "Run `python -m src.data.manifest` first."
        )
    return pq.read_table(path)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Build the dataset manifest."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force regeneration even if manifest already exists.",
    )
    parser.add_argument(
        "--base-path",
        default="data/raw/IRIS and FINGERPRINT DATASET",
    )
    parser.add_argument(
        "--manifest-path", default="data/manifest.parquet"
    )
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fp-pick", default="first",
        choices=["first", "random"]
    )
    parser.add_argument(
        "--iris-pick", default="first",
        choices=["first", "random"]
    )
    args = parser.parse_args()

    if not args.rebuild and Path(args.manifest_path).exists():
        logger.info(
            "Manifest already exists at %s. "
            "Use --rebuild to regenerate.",
            args.manifest_path,
        )
    else:
        build_manifest(
            base_path=args.base_path,
            manifest_path=args.manifest_path,
            val_split=args.val_split,
            seed=args.seed,
            fp_pick=args.fp_pick,
            iris_pick=args.iris_pick,
        )
