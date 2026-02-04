#!/usr/bin/env python3
"""
Export BIOVISION-AI model to ONNX for edge deployment (Jetson, etc.).

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/best.pt --output model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from biovision_ai.config import get_default_config
from biovision_ai.models import BioVisionModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="biovision_model.onnx")
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--dynamic", action="store_true", default=True)
    args = parser.parse_args()

    config = get_default_config()
    model = BioVisionModel(
        dermoscopy_backbone=config["model"]["dermoscopy_backbone"],
        clinical_backbone=config["model"]["clinical_backbone"],
        dermoscopy_embed_dim=config["model"]["dermoscopy_embed_dim"],
        clinical_embed_dim=config["model"]["clinical_embed_dim"],
        clinical_data_embed_dim=config["model"]["clinical_data_embed_dim"],
        num_clinical_features=config["data"]["num_clinical_features"],
        fusion_type=config["model"]["fusion_type"],
        fusion_hidden_dim=config["model"]["fusion_hidden_dim"],
        num_diagnosis_classes=config["model"]["num_diagnosis_classes"],
        risk_levels=config["model"]["risk_levels"],
        num_stages=len(config["model"]["stage_labels"]),
        num_trends=len(config["model"]["trend_labels"]),
        heads_config=config["model"]["heads"],
        pretrained=False,
    )

    if args.checkpoint and Path(args.checkpoint).exists():
        state = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)

    model.eval()

    B = 1
    dummy_derm = torch.randn(B, 3, 224, 224)
    dummy_clin = torch.randn(B, 3, 224, 224)
    dummy_cf = torch.randn(B, 32)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "dermoscopy": {0: "batch"},
            "clinical": {0: "batch"},
            "clinical_features": {0: "batch"},
            "diagnosis": {0: "batch"},
            "risk": {0: "batch"},
            "stage": {0: "batch"},
        }

    # ONNX export with multiple outputs - need wrapper for named outputs
    class ExportWrapper(torch.nn.Module):
        def __init__(self, m: BioVisionModel) -> None:
            super().__init__()
            self.model = m

        def forward(
            self,
            dermoscopy: torch.Tensor,
            clinical: torch.Tensor,
            clinical_features: torch.Tensor,
        ) -> tuple:
            out = self.model(
                dermoscopy=dermoscopy,
                clinical=clinical,
                clinical_features=clinical_features,
                has_clinical=torch.ones(dermoscopy.size(0), dtype=torch.bool),
            )
            return (out["diagnosis"], out["risk"], out["stage"])

    wrapped = ExportWrapper(model)
    torch.onnx.export(
        wrapped,
        (dummy_derm, dummy_clin, dummy_cf),
        args.output,
        opset_version=args.opset,
        input_names=["dermoscopy", "clinical", "clinical_features"],
        output_names=["diagnosis", "risk", "stage"],
        dynamic_axes=dynamic_axes,
    )
    print(f"Exported to {args.output}")


if __name__ == "__main__":
    main()
