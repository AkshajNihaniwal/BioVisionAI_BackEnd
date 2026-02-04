"""
Image backbone factory and wrapper for BIOVISION-AI.

Uses timm for ViT, EfficientNet, etc. Provides consistent interface
for embedding extraction.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


# Backbone name -> (timm name, default embed dim)
BACKBONE_REGISTRY = {
    "efficientnet_b0": ("tf_efficientnet_b0", 1280),
    "efficientnet_b1": ("tf_efficientnet_b1", 1280),
    "efficientnet_b2": ("tf_efficientnet_b2", 1408),
    "vit_tiny": ("vit_tiny_patch16_224", 192),
    "vit_small": ("vit_small_patch16_224", 384),
    "vit_base": ("vit_base_patch16_224", 768),
}


class ImageBackbone(nn.Module):
    """
    Wrapper around timm models for consistent embedding extraction.

    Removes classification head and returns feature vector.
    """

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool = True,
        embed_dim: Optional[int] = None,
        freeze_bn: bool = False,
    ) -> None:
        """
        Args:
            backbone_name: Key from BACKBONE_REGISTRY or timm model name.
            pretrained: Load ImageNet weights.
            embed_dim: Override output dim (default from registry).
            freeze_bn: Freeze batch norm for transfer learning.
        """
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required. Install with: pip install timm")

        if backbone_name in BACKBONE_REGISTRY:
            timm_name, default_embed = BACKBONE_REGISTRY[backbone_name]
            self.embed_dim = embed_dim or default_embed
        else:
            timm_name = backbone_name
            self.embed_dim = embed_dim or 768  # fallback

        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        # For models with global_pool="", we need to pool manually
        # timm with num_classes=0 typically returns (B, C, H, W) or (B, N, C)
        self._is_vit = "vit" in timm_name.lower()
        if self._is_vit:
            # ViT: (B, N+1, C) -> take [CLS] or mean
            self.pool = "cls"
        else:
            self.pool = "mean"

        if freeze_bn:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) image tensor.

        Returns:
            (B, embed_dim) feature vector.
        """
        out = self.backbone(x)
        if self._is_vit:
            # out: (B, N+1, C)
            if self.pool == "cls":
                return out[:, 0]
            return out[:, 1:].mean(dim=1)
        else:
            # CNN: (B, C, H, W) -> global average pool
            return out.mean(dim=[2, 3])


def create_image_backbone(
    backbone_name: str,
    pretrained: bool = True,
    embed_dim: Optional[int] = None,
    freeze_bn: bool = False,
) -> ImageBackbone:
    """Factory for ImageBackbone."""
    return ImageBackbone(
        backbone_name=backbone_name,
        pretrained=pretrained,
        embed_dim=embed_dim,
        freeze_bn=freeze_bn,
    )
