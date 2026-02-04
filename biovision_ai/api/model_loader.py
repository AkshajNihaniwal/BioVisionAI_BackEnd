"""
Model loading for API dependency injection.

Load model once at startup, inject into FastAPI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from biovision_ai.config import get_default_config
from biovision_ai.models.multimodal_model import BioVisionModel

logger = logging.getLogger(__name__)


class ModelHolder:
    """
    Holds loaded model and config for API.

    Used as FastAPI dependency.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        config: Optional[dict] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or get_default_config()
        model_config = self.config["model"]
        data_config = self.config["data"]

        self.model = BioVisionModel(
            dermoscopy_backbone=model_config.get("dermoscopy_backbone", "efficientnet_b0"),
            clinical_backbone=model_config.get("clinical_backbone", "efficientnet_b0"),
            dermoscopy_embed_dim=model_config.get("dermoscopy_embed_dim", 1280),
            clinical_embed_dim=model_config.get("clinical_embed_dim", 1280),
            clinical_data_embed_dim=model_config.get("clinical_data_embed_dim", 64),
            num_clinical_features=data_config.get("num_clinical_features", 32),
            fusion_type=model_config.get("fusion_type", "concat"),
            fusion_hidden_dim=model_config.get("fusion_hidden_dim", 512),
            num_diagnosis_classes=model_config.get("num_diagnosis_classes", 7),
            risk_levels=model_config.get("risk_levels", 3),
            num_stages=len(model_config.get("stage_labels", ["early", "intermediate", "advanced"])),
            num_trends=len(model_config.get("trend_labels", ["stable", "slowly_progressive", "rapidly_progressive"])),
            heads_config=model_config.get("heads", {}),
            segmentation_enabled=model_config.get("segmentation_enabled", False),
            pretrained=model_path is None,
        )

        if model_path and Path(model_path).exists():
            state = torch.load(model_path, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"], strict=False)
            else:
                self.model.load_state_dict(state, strict=False)
            logger.info(f"Loaded model from {model_path}")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

        self.diagnosis_classes = model_config.get("diagnosis_classes", [])
        self.risk_labels = model_config.get("risk_labels", [])
        self.stage_labels = model_config.get("stage_labels", [])
        self.trend_labels = model_config.get("trend_labels", [])
        self.version = "0.1.0"
