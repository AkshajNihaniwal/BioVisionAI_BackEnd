"""
Inference logic for BIOVISION-AI API.

Runs model forward, formats outputs, optionally generates heatmaps.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from biovision_ai.data.augmentations import get_dermoscopy_transforms, get_clinical_transforms

logger = logging.getLogger(__name__)


def _features_to_tensor(features: Optional[dict]) -> Optional[torch.Tensor]:
    """Convert clinical features dict to tensor (placeholder encoding)."""
    if not features:
        return None
    # Simple encoding: age/100, sex (M=1,F=0), fitzpatrick/6, etc.
    vals = []
    vals.append((features.get("age") or 50) / 100.0)
    vals.append(1.0 if (features.get("sex") or "").upper() == "M" else 0.0)
    vals.append((features.get("fitzpatrick_type") or 3) / 6.0)
    vals.append((features.get("symptom_duration_days") or 0) / 365.0)
    vals.append(1.0 if features.get("rapid_change") else 0.0)
    vals.append(1.0 if features.get("itching") else 0.0)
    vals.append(1.0 if features.get("bleeding") else 0.0)
    vals.append(1.0 if features.get("family_history") else 0.0)
    # Pad to 32 if needed
    while len(vals) < 32:
        vals.append(0.0)
    return torch.tensor([vals[:32]], dtype=torch.float32)


def run_inference(
    model_holder: Any,
    dermoscopy_image: Image.Image,
    clinical_image: Optional[Image.Image] = None,
    clinical_features: Optional[dict] = None,
    lesion_id: Optional[str] = None,
    generate_heatmap: bool = False,
    heatmap_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Run inference and return structured output.

    Args:
        model_holder: ModelHolder with loaded model.
        dermoscopy_image: PIL Image of dermoscopy.
        clinical_image: Optional clinical photo.
        clinical_features: Optional dict of clinical features.
        lesion_id: Optional ID for logging.
        generate_heatmap: Whether to generate Grad-CAM heatmap.
        heatmap_dir: Directory to save heatmaps.

    Returns:
        Dict with diagnosis_probabilities, risk_category, risk_score, etc.
    """
    device = model_holder.device
    model = model_holder.model

    transform_derm = get_dermoscopy_transforms(image_size=224, is_training=False)
    transform_clin = get_clinical_transforms(image_size=224, is_training=False)

    derm_tensor = transform_derm(dermoscopy_image).unsqueeze(0).to(device)
    clin_tensor = None
    if clinical_image is not None:
        clin_tensor = transform_clin(clinical_image).unsqueeze(0).to(device)
    cf_tensor = _features_to_tensor(clinical_features)
    if cf_tensor is not None:
        cf_tensor = cf_tensor.to(device)
    has_clinical = torch.tensor([clinical_image is not None], device=device, dtype=torch.bool)

    start = time.perf_counter()
    with torch.no_grad():
        out = model(
            dermoscopy=derm_tensor,
            clinical=clin_tensor,
            clinical_features=cf_tensor,
            has_clinical=has_clinical,
        )
    latency_ms = (time.perf_counter() - start) * 1000

    diagnosis_classes = model_holder.diagnosis_classes
    risk_labels = model_holder.risk_labels
    stage_labels = model_holder.stage_labels
    trend_labels = model_holder.trend_labels

    diagnosis_probs = torch.softmax(out["diagnosis"], dim=1)[0].cpu().numpy()
    diag_list = [
        {"class_name": c, "probability": float(p)}
        for c, p in zip(diagnosis_classes, diagnosis_probs)
    ]

    risk_probs = torch.softmax(out["risk"], dim=1)[0].cpu().numpy()
    risk_idx = int(np.argmax(risk_probs))
    risk_category = risk_labels[risk_idx] if risk_idx < len(risk_labels) else "unknown"
    risk_score = float(risk_probs[risk_idx])

    stage_estimate = None
    if "stage" in out:
        stage_probs = torch.softmax(out["stage"], dim=1)[0].cpu().numpy()
        stage_idx = int(np.argmax(stage_probs))
        stage_estimate = stage_labels[stage_idx] if stage_idx < len(stage_labels) else None

    trend_prediction = None
    if "trend" in out:
        trend_probs = torch.softmax(out["trend"], dim=1)[0].cpu().numpy()
        trend_idx = int(np.argmax(trend_probs))
        trend_prediction = trend_labels[trend_idx] if trend_idx < len(trend_labels) else None

    heatmap_path = None
    if generate_heatmap and heatmap_dir:
        try:
            from biovision_ai.models.explainability import (
                grad_cam,
                resize_heatmap,
                get_cnn_target_layer,
            )
            target_layer = get_cnn_target_layer(model.dermoscopy_backbone)
            if target_layer is not None:
                model.train()
                heatmap = grad_cam(
                    model.dermoscopy_backbone,
                    target_layer,
                    derm_tensor,
                    target_class=int(np.argmax(diagnosis_probs)),
                )
                model.eval()
                heatmap = resize_heatmap(heatmap, (224, 224))
                heatmap_path = str(heatmap_dir / f"{lesion_id or 'heatmap'}.png")
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                plt.imshow(heatmap, cmap="jet")
                plt.axis("off")
                plt.savefig(heatmap_path, bbox_inches="tight")
                plt.close()
        except Exception as e:
            logger.warning(f"Heatmap generation failed: {e}")

    logger.info(
        f"Inference lesion_id={lesion_id} risk={risk_category} "
        f"latency_ms={latency_ms:.1f} version={model_holder.version}"
    )

    return {
        "diagnosis_probabilities": diag_list,
        "risk_category": risk_category,
        "risk_score": risk_score,
        "stage_estimate": stage_estimate,
        "trend_prediction": trend_prediction,
        "heatmap_path": heatmap_path,
        "model_version": model_holder.version,
        "lesion_id": lesion_id,
        "latency_ms": latency_ms,
    }
