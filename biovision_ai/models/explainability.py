"""
Explainability utilities for BIOVISION-AI.

- Grad-CAM / Grad-CAM++ for CNN backbones
- Attention rollout for ViT backbones

Returns heatmaps overlayable on dermoscopic images for clinician review.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def grad_cam(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    target_output: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a CNN backbone.

    Args:
        model: Model containing target_layer (e.g., last conv of backbone).
        target_layer: Layer to compute gradients for (register hook).
        input_tensor: (1, 3, H, W) input image.
        target_class: Class index for gradient; if None, use max logit.
        target_output: Precomputed output logits (1, C); if None, run forward.

    Returns:
        (H, W) heatmap as numpy array, normalized [0, 1].
    """
    model.eval()
    activations = []
    gradients = []

    def save_activation(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        activations.append(output.detach())

    def save_gradient(module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor) -> None:
        gradients.append(grad_output[0].detach())

    h_forward = target_layer.register_forward_hook(save_activation)
    h_backward = target_layer.register_full_backward_hook(save_gradient)

    try:
        input_tensor.requires_grad_(True)
        output = model(input_tensor)
        if target_output is not None:
            out = target_output
        else:
            out = output if isinstance(output, torch.Tensor) else output.get("diagnosis", output[0])
        if target_class is None:
            target_class = out.argmax(dim=1).item()
        target_score = out[0, target_class]
        model.zero_grad()
        target_score.backward()

        act = activations[0][0]
        grad = gradients[0][0]
        weights = grad.mean(dim=(1, 2))
        cam = (weights[:, None, None] * act).sum(dim=0)
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = np.float32(cam)
        return cam
    finally:
        h_forward.remove()
        h_backward.remove()


def resize_heatmap(heatmap: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize heatmap to match input image size."""
    import torch
    h = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    h = F.interpolate(
        h, size=target_size, mode="bilinear", align_corners=False
    )
    return h.squeeze().numpy()


def attention_rollout(
    attentions: list[torch.Tensor],
    discard_ratio: float = 0.9,
    head_fusion: str = "mean",
) -> np.ndarray:
    """
    Compute attention rollout for ViT.

    Args:
        attentions: List of (B, num_heads, N+1, N+1) attention matrices.
        discard_ratio: Ratio of low-attention weights to discard.
        head_fusion: "mean" or "max" over heads.

    Returns:
        (N+1,) rollout weights (CLS + patch tokens).
    """
    result = torch.eye(attentions[0].size(-1), device=attentions[0].device)
    with torch.no_grad():
        for attn in attentions:
            attn = attn.mean(dim=1) if head_fusion == "mean" else attn.max(dim=1)[0]
            attn = attn + torch.eye(attn.size(-1), device=attn.device)
            attn = attn / attn.sum(dim=-1, keepdim=True)
            result = torch.matmul(attn, result)

    rollout = result[0, 0]
    mask = rollout > rollout.quantile(discard_ratio)
    rollout = rollout * mask
    rollout = rollout / rollout.sum()
    return rollout.cpu().numpy()


def rollout_to_spatial_heatmap(
    rollout: np.ndarray,
    patch_size: int = 16,
    num_patches_per_side: int = 14,
) -> np.ndarray:
    """
    Convert 1D rollout (excluding CLS) to 2D spatial heatmap.

    Args:
        rollout: (N+1,) with rollout[0] = CLS.
        patch_size: ViT patch size.
        num_patches_per_side: sqrt(num_patches).

    Returns:
        (H, W) heatmap.
    """
    patches = rollout[1 : 1 + num_patches_per_side**2]
    heatmap = patches.reshape(num_patches_per_side, num_patches_per_side)
    heatmap = np.kron(heatmap, np.ones((patch_size, patch_size)))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return np.float32(heatmap)


def get_cnn_target_layer(backbone: nn.Module) -> Optional[nn.Module]:
    """Find last conv layer in a CNN backbone for Grad-CAM."""
    last_conv = None
    for m in backbone.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv
