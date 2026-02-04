"""
Sequence model for trajectory prediction (skeleton/placeholder).

Future: Transformer or LSTM over multiple past images of the same lesion
to predict future stage. Infrastructure ready; training logic TODO.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LesionSequenceModel(nn.Module):
    """
    Skeleton for sequence-based trajectory prediction.

    Takes multiple timepoint embeddings and predicts future stage/trend.
    TODO: Implement actual training. Structure supports:
    - Input: sequence of (dermoscopy_emb, clinical_emb, features) per timepoint
    - Output: stage, trend for next/future timepoint
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_stages: int = 3,
        num_trends: int = 3,
        max_sequence_length: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_sequence_length = max_sequence_length

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=False,
        )

        self.stage_head = nn.Linear(hidden_dim, num_stages)
        self.trend_head = nn.Linear(hidden_dim, num_trends)

    def forward(
        self,
        sequence_embeddings: torch.Tensor,
        sequence_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            sequence_embeddings: (B, T, embed_dim) per-timepoint embeddings.
            sequence_mask: (B, T) bool, True for valid timesteps.

        Returns:
            stage_logits (B, num_stages), trend_logits (B, num_trends).
        """
        lstm_out, (h_n, _) = self.lstm(sequence_embeddings)
        last_hidden = h_n[-1]

        return {
            "stage": self.stage_head(last_hidden),
            "trend": self.trend_head(last_hidden),
        }


# Placeholder for future Transformer-based sequence model
class LesionTransformerSequenceModel(nn.Module):
    """
    TODO: Transformer over time for trajectory prediction.

    Alternative to LSTM for modeling long-range temporal dependencies.
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, num_layers: int = 4) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Placeholder: returns input for now."""
        return self.transformer(x)
