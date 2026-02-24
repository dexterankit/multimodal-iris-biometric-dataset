"""Multimodal biometric recognition model.

PyTorch port of the Kaggle reference notebook:
https://www.kaggle.com/code/omidsakaki1370/multimodal-biometric-recognition-system

Architecture:
- FingerprintBranch  : MobileNetV2 (pretrained, optionally frozen)
- IrisBranch         : Lightweight CNN (shared weights for L/R iris)
- MultimodalNet      : Concatenation fusion + classification head
"""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class IrisBranch(nn.Module):
    """Small CNN for iris feature extraction.

    Mirrors the Keras ``create_iris_branch`` function from the
    reference notebook.  The same instance is reused for both left
    and right iris (shared weights).

    Input shape: ``(B, 1, 64, 64)``
    Output shape: ``(B, 32)``

    Args:
        in_channels: Input channels (1 for grayscale).
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Global pooling → (B, 32, 1, 1)
            nn.AdaptiveAvgPool2d(1),
        )
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract iris features.

        Args:
            x: Grayscale iris tensor ``(B, 1, H, W)``.

        Returns:
            Feature vector ``(B, 32)``.
        """
        return self.flatten(self.features(x))


class FingerprintBranch(nn.Module):
    """MobileNetV2-backed fingerprint feature extractor.

    Uses the ImageNet-pretrained MobileNetV2 as a backbone with the
    classification head removed.  Backbone weights are optionally
    frozen (``freeze=True``) to replicate the reference notebook's
    ``base_model.trainable = False``.

    Input shape: ``(B, 3, 128, 128)``
    Output shape: ``(B, 1280)``

    Args:
        freeze: If ``True``, freeze all backbone parameters.
    """

    def __init__(self, freeze: bool = True) -> None:
        super().__init__()
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        # Remove the classifier — keep only feature layers
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
            logger.info("FingerprintBranch: backbone weights frozen")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract fingerprint features.

        Args:
            x: RGB fingerprint tensor ``(B, 3, H, W)``.

        Returns:
            Feature vector ``(B, 1280)``.
        """
        x = self.features(x)
        x = self.pool(x)
        return self.flatten(x)


class MultimodalNet(nn.Module):
    """Full multimodal biometric recognition model.

    Fuses fingerprint (MobileNetV2) and iris (shared-weight CNN)
    features via concatenation, then classifies into *num_classes*
    person identities.

    Args:
        num_classes: Number of output classes (default 45).
        embedding_dim: Size of the fusion hidden layer.
        freeze_backbone: Whether to freeze MobileNetV2 weights.
    """

    # Feature vector sizes
    _FP_DIM: int = 1280
    _IRIS_DIM: int = 32

    def __init__(
        self,
        num_classes: int = 45,
        embedding_dim: int = 128,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.fingerprint_branch = FingerprintBranch(freeze=freeze_backbone)
        # Shared-weight iris branch: ONE instance, called twice
        self.iris_branch = IrisBranch(in_channels=1)

        fusion_dim = self._FP_DIM + self._IRIS_DIM * 2  # 1344

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim, num_classes),
        )

        logger.info(
            "MultimodalNet — fusion_dim=%d, embedding_dim=%d, "
            "num_classes=%d, freeze_backbone=%s",
            fusion_dim, embedding_dim, num_classes, freeze_backbone,
        )

    def forward(
        self,
        fp: torch.Tensor,
        left_iris: torch.Tensor,
        right_iris: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            fp: Fingerprint tensor ``(B, 3, 128, 128)``.
            left_iris: Left iris tensor ``(B, 1, 64, 64)``.
            right_iris: Right iris tensor ``(B, 1, 64, 64)``.

        Returns:
            Raw logits ``(B, num_classes)``.
        """
        fp_feat = self.fingerprint_branch(fp)           # (B, 1280)
        left_feat = self.iris_branch(left_iris)         # (B, 32)
        right_feat = self.iris_branch(right_iris)       # (B, 32)
        combined = torch.cat(
            [fp_feat, left_feat, right_feat], dim=1
        )                                               # (B, 1344)
        return self.classifier(combined)                # (B, 45)

    def output_shape(self) -> Tuple[int]:
        """Return classifier output dimension."""
        last = list(self.classifier.children())[-1]
        return (last.out_features,)
