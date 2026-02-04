"""
Image backbones for dermoscopy and clinical photographs.

Supports ViT and EfficientNet via timm. Designed for edge deployment
(EfficientNet-B0/B1, ViT-Tiny/Small).
"""

from biovision_ai.models.backbones.image_backbone import (
    create_image_backbone,
    ImageBackbone,
)

__all__ = ["create_image_backbone", "ImageBackbone"]
