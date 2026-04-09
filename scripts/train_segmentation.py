#!/usr/bin/env python3
"""
Train U-Net for lesion segmentation (optional; requires masks in data/masks/).
Usage: python scripts/train_segmentation.py --config configs/default.yaml --data_root ./data
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.optim import Adam

from utils.config import load_config
from utils.logger import setup_logger
from data.datasets.skin_lesion import get_dataloaders
from models.segmentation.unet import UNet
from training.losses import DiceBCELoss


def segmentation_batch_metrics(
    pred_probs: torch.Tensor,
    true_masks: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
):
    """Compute Dice, IoU, and pixel accuracy for a batch of binary masks."""
    pred_bin = (pred_probs > threshold).float()
    true_bin = true_masks.float()

    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    true_flat = true_bin.view(true_bin.size(0), -1)

    intersection = (pred_flat * true_flat).sum(dim=1)
    pred_sum = pred_flat.sum(dim=1)
    true_sum = true_flat.sum(dim=1)
    union = pred_sum + true_sum - intersection

    dice = (2.0 * intersection + eps) / (pred_sum + true_sum + eps)
    iou = (intersection + eps) / (union + eps)
    acc = (pred_flat == true_flat).float().mean(dim=1)

    return dice.mean(), iou.mean(), acc.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_root", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root

    seg_cfg = config.get("segmentation", {})
    if not seg_cfg.get("enabled", True):
        print("Segmentation disabled in config. Enable segmentation.enabled and provide masks.")
        return

    logger = setup_logger("seg_train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]

    try:
        train_loader, val_loader, _, _, _, _ = get_dataloaders(
            data_root=data_cfg["root"],
            batch_size=data_cfg["batch_size"],
            num_workers=data_cfg.get("num_workers", 4),
            train_ratio=data_cfg["train_ratio"],
            val_ratio=data_cfg["val_ratio"],
            test_ratio=data_cfg["test_ratio"],
            image_size=(data_cfg["image_size"], data_cfg["image_size"]),
            mask_dir="masks",
            seed=config.get("seed", 42),
        )
    except Exception as e:
        logger.warning("Dataloaders with masks failed (no masks?): %s. Skipping segmentation training.", e)
        return

    # Filter batches that have at least one mask (optional: require all)
    model = UNet(
        in_channels=seg_cfg.get("in_channels", 3),
        out_channels=seg_cfg.get("out_channels", 1),
    ).to(device)
    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = Adam(model.parameters(), lr=seg_cfg.get("lr", 1e-4))
    ckpt_dir = Path(seg_cfg.get("checkpoint_dir", "checkpoints/segmentation"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "latest.pt"
    start_epoch = 0

    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, seg_cfg.get("epochs", 35)):
        model.train()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        total_acc = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["image"].to(device)
            mask = batch.get("mask")
            if mask is None:
                continue
            mask = mask.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # UNet currently returns probabilities (sigmoid already applied).
                pred_probs = pred.detach()
                dice, iou, acc = segmentation_batch_metrics(pred_probs, mask)

            total_loss += loss.item()
            total_dice += dice.item()
            total_iou += iou.item()
            total_acc += acc.item()
            n_batches += 1
        if n_batches == 0:
            logger.info("No batches with masks; skipping segmentation training.")
            break

        avg_loss = total_loss / n_batches
        avg_dice = total_dice / n_batches
        avg_iou = total_iou / n_batches
        avg_acc = total_acc / n_batches

        model.eval()
        val_total_loss = 0.0
        val_total_dice = 0.0
        val_total_iou = 0.0
        val_total_acc = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                mask = batch.get("mask")
                if mask is None:
                    continue
                mask = mask.to(device)
                pred = model(x)
                loss = criterion(pred, mask)
                dice, iou, acc = segmentation_batch_metrics(pred, mask)

                val_total_loss += loss.item()
                val_total_dice += dice.item()
                val_total_iou += iou.item()
                val_total_acc += acc.item()
                val_batches += 1

        if val_batches > 0:
            val_loss = val_total_loss / val_batches
            val_dice = val_total_dice / val_batches
            val_iou = val_total_iou / val_batches
            val_acc = val_total_acc / val_batches
            logger.info(
                "Epoch %d seg_loss=%.4f dice=%.4f iou=%.4f acc=%.4f val_seg_loss=%.4f val_dice=%.4f val_iou=%.4f val_acc=%.4f",
                epoch + 1,
                avg_loss,
                avg_dice,
                avg_iou,
                avg_acc,
                val_loss,
                val_dice,
                val_iou,
                val_acc,
            )
        else:
            logger.info(
                "Epoch %d seg_loss=%.4f dice=%.4f iou=%.4f acc=%.4f",
                epoch + 1,
                avg_loss,
                avg_dice,
                avg_iou,
                avg_acc,
            )
        torch.save(
            {
                # Loop variable `epoch` is the last completed epoch index (0-based).
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            ckpt_dir / "latest.pt",
        )

    logger.info("Segmentation training done.")


if __name__ == "__main__":
    main()
