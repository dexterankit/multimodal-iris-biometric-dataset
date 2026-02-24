"""Training entry point — run with Hydra.

Example::

    python train.py                                    # defaults
    python train.py training.lr=0.001 training.epochs=20
    python train.py model.freeze_backbone=false
"""

import hydra
from omegaconf import DictConfig

from src.data.dataset import build_dataloaders
from src.data.manifest import build_manifest, load_manifest
from src.models.multimodal_net import MultimodalNet
from src.training.trainer import Trainer
from src.utils.logging_utils import setup_logging
from src.utils.seed import set_seed

import logging

log = logging.getLogger(__name__)


@hydra.main(
    config_path="configs",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point.

    Args:
        cfg: Composed Hydra config (data + model + training).
    """
    setup_logging()
    set_seed(cfg.training.seed)

    log.info("=== Multimodal Biometric Training ===")

    # Build manifest if it doesn't exist
    try:
        table = load_manifest(cfg.data.manifest_path)
        log.info(
            "Loaded existing manifest (%d rows)", len(table)
        )
    except FileNotFoundError:
        log.info("Manifest not found — building now…")
        table = build_manifest(
            base_path=cfg.data.base_path,
            manifest_path=cfg.data.manifest_path,
            val_split=cfg.data.val_split,
            seed=cfg.data.seed,
            fp_pick=cfg.data.fp_pick,
            iris_pick=cfg.data.iris_pick,
        )

    # Validate processed paths exist
    rows = table.to_pylist()
    if "processed_fp_path" not in rows[0]:
        log.warning(
            "Processed images not found in manifest. "
            "Run `python -m src.data.preprocessing` first. "
            "Falling back to raw paths."
        )

    # Build data loaders
    train_loader, val_loader = build_dataloaders(
        manifest_path=cfg.data.manifest_path,
        fp_size=tuple(cfg.data.fp_size),
        iris_size=tuple(cfg.data.iris_size),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=False,
    )

    # Build model
    model = MultimodalNet(
        num_classes=cfg.model.num_classes,
        embedding_dim=cfg.model.embedding_dim,
        freeze_backbone=cfg.model.freeze_backbone,
    )

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg.training,
        metrics_path="metrics.jsonl",
    )
    trainer.fit()


if __name__ == "__main__":
    main()
