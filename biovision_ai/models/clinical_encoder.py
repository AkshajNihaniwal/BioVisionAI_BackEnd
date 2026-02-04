"""
Clinical tabular data encoder for BIOVISION-AI.

MLP that encodes structured clinical features (age, sex, Fitzpatrick type,
anatomical site, symptom duration, flags, labs) into an embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClinicalDataEncoder(nn.Module):
    """
    MLP encoder for tabular clinical features.

    Input: (B, num_features) float tensor.
    Output: (B, embed_dim) embedding.
    """

    def __init__(
        self,
        num_features: int,
        embed_dim: int = 64,
        hidden_dims: tuple[int, ...] = (128, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers = []
        prev = num_features
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, embed_dim))
        self.mlp = nn.Sequential(*layers)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
