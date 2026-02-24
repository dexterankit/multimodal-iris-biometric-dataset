"""Inference entry point.

Example::

    python infer.py \\
        fp="data/processed/fingerprints/1.bmp" \\
        left="data/processed/left/1.bmp" \\
        right="data/processed/right/1.bmp" \\
        ckpt="outputs/checkpoints/best.pt"
"""

import hydra
from omegaconf import DictConfig

from src.inference.predictor import Predictor
from src.utils.logging_utils import setup_logging

import logging

log = logging.getLogger(__name__)


@hydra.main(
    config_path="configs",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """Run inference on a single sample.

    Expects three extra CLI keys injected via Hydra ``+key=value``
    syntax: ``fp``, ``left``, ``right``, ``ckpt``.

    Args:
        cfg: Hydra config, extended with inference-time overrides.
    """
    setup_logging()

    # Extra keys passed on CLI with `+` prefix
    fp_path: str = cfg.get("fp", "")
    left_path: str = cfg.get("left", "")
    right_path: str = cfg.get("right", "")
    ckpt_path: str = cfg.get("ckpt", "outputs/checkpoints/best.pt")

    if not all([fp_path, left_path, right_path]):
        raise ValueError(
            "Provide all three image paths:\n"
            "  python infer.py +fp=<path> +left=<path> "
            "+right=<path> +ckpt=<ckpt_path>"
        )

    predictor = Predictor(ckpt_path)
    person_id = predictor.predict(fp_path, left_path, right_path)
    log.info("Predicted person ID: %d", person_id)
    print(f"Predicted person ID: {person_id}")


if __name__ == "__main__":
    main()
