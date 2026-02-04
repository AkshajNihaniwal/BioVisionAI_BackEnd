#!/usr/bin/env python3
"""
Evaluation script for BIOVISION-AI.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --data_dir /path/to/test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from biovision_ai.config import load_config, get_default_config
from biovision_ai.data import LesionDataset, get_dermoscopy_transforms, collate_lesion_batch
from biovision_ai.models import BioVisionModel
from biovision_ai.training.evaluation import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    config = get_default_config()
    if args.config:
        config = load_config(args.config)

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
        print(f"Loaded checkpoint from {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # TODO: Load real test dataset from args.data_dir
    # For now, create minimal dummy
    if args.data_dir and Path(args.data_dir).exists():
        # Placeholder: implement ISIC/test loader
        raise NotImplementedError("Implement data_dir loader for your dataset format")
    else:
        from biovision_ai.data.dummy import create_dummy_lesion_dataset
        test_ds = create_dummy_lesion_dataset(20)
        loader = DataLoader(
            test_ds,
            batch_size=8,
            shuffle=False,
            collate_fn=collate_lesion_batch,
        )
        results = evaluate_model(model, loader, device)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
        print(json.dumps(results, indent=2))
