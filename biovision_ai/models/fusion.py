"""
Multimodal fusion layers for BIOVISION-AI.

Combines dermoscopy embedding, clinical photo embedding (if present),
and clinical tabular embedding. Modular design for easy swapping.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """
    Simple concatenation + feed-forward fusion.

    Fuses: dermoscopy_emb, clinical_emb (or zero), clinical_data_emb.
    """

    def __init__(
        self,
        dermoscopy_dim: int,
        clinical_dim: int,
        clinical_data_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dermoscopy_dim = dermoscopy_dim
        self.clinical_dim = clinical_dim
        self.clinical_data_dim = clinical_data_dim
        total_dim = dermoscopy_dim + clinical_dim + clinical_data_dim
        self.fc = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.output_dim = hidden_dim

    def forward(
        self,
        dermoscopy_emb: torch.Tensor,
        clinical_emb: Optional[torch.Tensor],
        clinical_data_emb: torch.Tensor,
        has_clinical: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            dermoscopy_emb: (B, D)
            clinical_emb: (B, C) or None; use zeros if absent
            clinical_data_emb: (B, F)
            has_clinical: (B,) bool; mask for optional clinical

        Returns:
            (B, output_dim) fused embedding
        """
        B = dermoscopy_emb.size(0)
        if clinical_emb is None:
            clinical_emb = torch.zeros(
                B,
                self.clinical_dim,
                device=dermoscopy_emb.device,
                dtype=dermoscopy_emb.dtype,
            )
        elif has_clinical is not None:
            clinical_emb = clinical_emb.clone()
            clinical_emb[~has_clinical] = 0

        concat = torch.cat([dermoscopy_emb, clinical_emb, clinical_data_emb], dim=1)
        return self.fc(concat)


def create_concat_fusion(
    dermoscopy_dim: int,
    clinical_dim: int,
    clinical_data_dim: int,
    hidden_dim: int = 512,
    dropout: float = 0.2,
) -> ConcatFusion:
    """Factory for ConcatFusion."""
    return ConcatFusion(
        dermoscopy_dim=dermoscopy_dim,
        clinical_dim=clinical_dim,
        clinical_data_dim=clinical_data_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


class CrossAttentionFusion(nn.Module):
    """
    Transformer-style cross-attention fusion across modalities.

    More expressive but heavier. Optional advanced fusion.
    """

    def __init__(
        self,
        dermoscopy_dim: int,
        clinical_dim: int,
        clinical_data_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_derm = nn.Linear(dermoscopy_dim, hidden_dim)
        self.proj_clinical = nn.Linear(clinical_dim, hidden_dim)
        self.proj_data = nn.Linear(clinical_data_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = hidden_dim

    def forward(
        self,
        dermoscopy_emb: torch.Tensor,
        clinical_emb: Optional[torch.Tensor],
        clinical_data_emb: torch.Tensor,
        has_clinical: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = dermoscopy_emb.size(0)
        tokens = [
            self.proj_derm(dermoscopy_emb).unsqueeze(1),
            self.proj_data(clinical_data_emb).unsqueeze(1),
        ]
        if clinical_emb is not None and (has_clinical is None or has_clinical.any()):
            if has_clinical is not None:
                clinical_emb = clinical_emb.clone()
                clinical_emb[~has_clinical] = 0
            tokens.append(self.proj_clinical(clinical_emb).unsqueeze(1))
        else:
            zero_clinical = torch.zeros(
                B, self.proj_clinical.in_features,
                device=dermoscopy_emb.device,
                dtype=dermoscopy_emb.dtype,
            )
            tokens.append(self.proj_clinical(zero_clinical).unsqueeze(1))
        x = torch.cat(tokens, dim=1)
        x = self.transformer(x)
        return x.mean(dim=1)


def create_fusion_layer(
    fusion_type: str,
    dermoscopy_dim: int,
    clinical_dim: int,
    clinical_data_dim: int,
    hidden_dim: int = 512,
    **kwargs: object,
) -> nn.Module:
    """
    Factory for fusion layers.

    Args:
        fusion_type: "concat" or "cross_attention"
        dermoscopy_dim: Dermoscopy embedding size.
        clinical_dim: Clinical image embedding size.
        clinical_data_dim: Tabular features embedding size.
        hidden_dim: Fusion output dimension.
        **kwargs: Extra args for specific fusion types.

    Returns:
        Fusion module with .output_dim attribute.
    """
    if fusion_type == "concat":
        return create_concat_fusion(
            dermoscopy_dim=dermoscopy_dim,
            clinical_dim=clinical_dim,
            clinical_data_dim=clinical_data_dim,
            hidden_dim=hidden_dim,
            **kwargs,
        )
    elif fusion_type == "cross_attention":
        return CrossAttentionFusion(
            dermoscopy_dim=dermoscopy_dim,
            clinical_dim=clinical_dim,
            clinical_data_dim=clinical_data_dim,
            hidden_dim=hidden_dim,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
