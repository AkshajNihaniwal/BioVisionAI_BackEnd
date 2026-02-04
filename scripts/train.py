#!/usr/bin/env python3
"""
Training script for BIOVISION-AI.

Usage:
    python scripts/train.py --config config/default.yaml
    python scripts/train.py --data_dir /path/to/isic --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from biovision_ai.config import load_config, get_default_config
from biovision_ai.data import collate_lesion_batch
from biovision_ai.data.dummy import create_dummy_lesion_dataset
from biovision_ai.models import BioVisionModel
from biovision_ai.training import Trainer
from biovision_ai.utils.logging import setup_logging




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    setup_logging()
    config = get_default_config()
    if args.config:
        config = load_config(args.config)

    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr

    # TODO: Load real dataset from args.data_dir (ISIC, etc.)
    # For now use dummy
    train_ds = create_dummy_lesion_dataset(100)
    val_ds = create_dummy_lesion_dataset(20)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_lesion_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_lesion_batch,
    )

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
        segmentation_enabled=config["model"]["segmentation_enabled"],
        pretrained=True,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        num_epochs=config["training"]["num_epochs"],
        focal_gamma=config["training"].get("focal_loss_gamma", 2.0),
        heads_config=config["model"]["heads"],
        checkpoint_dir=Path(args.checkpoint_dir),
    )

    trainer.train()
    print("Training complete.")


if __name__ == "__main__":
    main()
