"""
Configuration loader for BIOVISION-AI.

Supports YAML and JSON config files. Provides sensible defaults for
model architecture, training, API, and deployment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.

    Args:
        config_path: Path to config file.

    Returns:
        Configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(f)
        elif path.suffix == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def get_default_config() -> dict[str, Any]:
    """
    Return default configuration for BIOVISION-AI.

    Covers model architecture, training, API, and deployment.
    """
    return {
        "model": {
            "dermoscopy_backbone": "efficientnet_b0",
            "clinical_backbone": "efficientnet_b0",
            "clinical_backbone_pretrained": True,
            "dermoscopy_embed_dim": 1280,
            "clinical_embed_dim": 1280,
            "clinical_data_embed_dim": 64,
            "fusion_type": "concat",
            "fusion_hidden_dim": 512,
            "num_diagnosis_classes": 7,
            "diagnosis_classes": [
                "nevus",
                "melanoma",
                "bcc",
                "scc",
                "seborrheic_keratosis",
                "inflammatory",
                "other",
            ],
            "risk_levels": 3,
            "risk_labels": ["low", "intermediate", "high"],
            "stage_labels": ["early", "intermediate", "advanced"],
            "trend_labels": ["stable", "slowly_progressive", "rapidly_progressive"],
            "segmentation_enabled": False,
            "heads": {
                "diagnosis": True,
                "risk": True,
                "stage": True,
                "trend": False,
            },
        },
        "data": {
            "image_size": 224,
            "dermoscopy_size": 224,
            "clinical_size": 224,
            "num_clinical_features": 32,
            "clinical_feature_names": [
                "age",
                "sex",
                "fitzpatrick_type",
                "anatomical_site",
                "symptom_duration_days",
                "rapid_change",
                "itching",
                "bleeding",
                "family_history",
            ],
        },
        "training": {
            "batch_size": 32,
            "num_epochs": 50,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "focal_loss_gamma": 2.0,
            "class_weights": None,
            "optimizer": "adamw",
            "scheduler": "cosine",
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "model_path": None,
            "max_batch_size": 8,
        },
        "export": {
            "onnx_opset": 14,
            "dynamic_axes": True,
        },
    }
