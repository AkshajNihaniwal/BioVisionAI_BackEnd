"""
RESEARCH-ONLY: Placeholder for generative trajectory module.

Future integration of diffusion/VAE to produce synthetic images
representing earlier/later stages of similar lesions.

IMPORTANT: These are educational/illustrative synthetic examples,
NOT true past images. Must be clearly labeled as such in UI/docs.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class GenerativeTrajectoryPlaceholder(nn.Module):
    """
    Placeholder for generative model producing synthetic trajectory images.

    TODO: Integrate diffusion or VAE. Output synthetic dermoscopy
    representing "typical" earlier/later appearance of similar lesions.
    For research/education only.
    """

    def __init__(self, latent_dim: int = 256, image_size: int = 224) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        # Stub: no actual generative layers yet
        self.stub = nn.Linear(latent_dim, 1)

    def forward(
        self,
        embedding: torch.Tensor,
        target_stage: str = "earlier",
    ) -> Optional[torch.Tensor]:
        """
        Placeholder: returns None. Future: return synthetic image tensor.

        Args:
            embedding: Fused lesion embedding.
            target_stage: "earlier" or "later" for trajectory direction.

        Returns:
            None (placeholder). Future: (B, 3, H, W) synthetic image.
        """
        _ = self.stub(embedding[:, : self.latent_dim])
        return None
