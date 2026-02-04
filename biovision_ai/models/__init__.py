"""
BIOVISION-AI multimodal model components.

Includes:
- Image backbones (ViT, EfficientNet)
- Segmentation module (UNet-style)
- Fusion layers
- Multi-task heads (diagnosis, risk, stage, trend)
- Full multimodal model
"""

from biovision_ai.models.multimodal_model import BioVisionModel

__all__ = ["BioVisionModel"]
