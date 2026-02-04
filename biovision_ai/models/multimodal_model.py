"""
Full BIOVISION-AI multimodal model.

Combines dermoscopy backbone, optional clinical backbone, clinical data encoder,
fusion layer, and multi-task heads. Supports optional segmentation preprocessing.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from biovision_ai.models.backbones.image_backbone import create_image_backbone
from biovision_ai.models.clinical_encoder import ClinicalDataEncoder
from biovision_ai.models.fusion import create_fusion_layer
from biovision_ai.models.heads import create_heads
from biovision_ai.models.segmentation.unet import LesionSegmentationUNet


class BioVisionModel(nn.Module):
    """
    BIOVISION-AI multimodal lesion analysis model.

    Forward pass:
    1. Optional: segment lesion, crop/mask dermoscopy
    2. Dermoscopy -> backbone -> embedding
    3. Clinical image (if present) -> backbone -> embedding
    4. Clinical tabular -> MLP -> embedding
    5. Fusion -> fused embedding
    6. Heads -> diagnosis, risk, stage, trend logits
    """

    def __init__(
        self,
        dermoscopy_backbone: str = "efficientnet_b0",
        clinical_backbone: str = "efficientnet_b0",
        dermoscopy_embed_dim: int = 1280,
        clinical_embed_dim: int = 1280,
        clinical_data_embed_dim: int = 64,
        num_clinical_features: int = 32,
        fusion_type: str = "concat",
        fusion_hidden_dim: int = 512,
        num_diagnosis_classes: int = 7,
        risk_levels: int = 3,
        num_stages: int = 3,
        num_trends: int = 3,
        heads_config: Optional[dict[str, bool]] = None,
        segmentation_enabled: bool = False,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dermoscopy_embed_dim = dermoscopy_embed_dim
        self.clinical_embed_dim = clinical_embed_dim
        self.clinical_data_embed_dim = clinical_data_embed_dim
        self.num_diagnosis_classes = num_diagnosis_classes
        self.risk_levels = risk_levels
        self.num_stages = num_stages
        self.num_trends = num_trends
        self.segmentation_enabled = segmentation_enabled

        # Segmentation (optional)
        if segmentation_enabled:
            self.segmentation = LesionSegmentationUNet(in_channels=3, base_channels=32)
        else:
            self.segmentation = None

        # Image backbones
        self.dermoscopy_backbone = create_image_backbone(
            backbone_name=dermoscopy_backbone,
            pretrained=pretrained,
            embed_dim=dermoscopy_embed_dim,
        )
        self.clinical_backbone = create_image_backbone(
            backbone_name=clinical_backbone,
            pretrained=pretrained,
            embed_dim=clinical_embed_dim,
        )

        # Clinical data encoder
        self.clinical_encoder = ClinicalDataEncoder(
            num_features=num_clinical_features,
            embed_dim=clinical_data_embed_dim,
            hidden_dims=(128, 128),
            dropout=dropout,
        )

        # Fusion
        self.fusion = create_fusion_layer(
            fusion_type=fusion_type,
            dermoscopy_dim=dermoscopy_embed_dim,
            clinical_dim=clinical_embed_dim,
            clinical_data_dim=clinical_data_embed_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
        )
        fusion_dim = self.fusion.output_dim

        # Heads
        self.heads = create_heads(
            fusion_dim=fusion_dim,
            num_diagnosis_classes=num_diagnosis_classes,
            num_risk_levels=risk_levels,
            num_stages=num_stages,
            num_trends=num_trends,
            heads_config=heads_config,
            dropout=dropout,
        )

    def _apply_segmentation(
        self, dermoscopy: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """Apply segmentation mask to dermoscopy (optional cropping/masking)."""
        if self.segmentation is None:
            return dermoscopy
        mask = self.segmentation(dermoscopy)
        mask_binary = (mask > threshold).float()
        return dermoscopy * mask_binary + dermoscopy * (1 - mask_binary) * 0.5
        # Simple masking: darken non-lesion region. Could also crop to bbox.

    def forward(
        self,
        dermoscopy: torch.Tensor,
        clinical: Optional[torch.Tensor] = None,
        clinical_features: Optional[torch.Tensor] = None,
        has_clinical: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            dermoscopy: (B, 3, H, W) dermoscopic image.
            clinical: (B, 3, H, W) optional clinical photo.
            clinical_features: (B, F) tabular clinical features.
            has_clinical: (B,) bool mask for valid clinical images.

        Returns:
            Dict with logits for each active head (diagnosis, risk, stage, trend).
        """
        B = dermoscopy.size(0)
        device = dermoscopy.device

        if self.segmentation_enabled and self.segmentation is not None:
            dermoscopy = self._apply_segmentation(dermoscopy)

        derm_emb = self.dermoscopy_backbone(dermoscopy)

        if clinical is not None:
            clin_emb = self.clinical_backbone(clinical)
        else:
            clin_emb = None

        if clinical_features is not None:
            data_emb = self.clinical_encoder(clinical_features)
        else:
            data_emb = torch.zeros(
                B, self.clinical_data_embed_dim, device=device, dtype=derm_emb.dtype
            )

        fused = self.fusion(
            derm_emb, clin_emb, data_emb, has_clinical=has_clinical
        )

        out: dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            out[name] = head(fused)
        return out
