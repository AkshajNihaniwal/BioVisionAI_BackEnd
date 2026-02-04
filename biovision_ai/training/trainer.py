"""
Training loop for BIOVISION-AI.

Supports multi-task training (diagnosis, risk, stage, trend).
Class weighting and focal loss for imbalance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from biovision_ai.training.losses import create_loss_fn
from biovision_ai.training.evaluation import evaluate_model

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for BioVisionModel.

    Multi-task training with configurable heads and losses.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        optimizer: str = "adamw",
        scheduler: Optional[str] = "cosine",
        num_epochs: int = 50,
        focal_gamma: float = 2.0,
        class_weights: Optional[dict[str, torch.Tensor]] = None,
        heads_config: Optional[dict[str, bool]] = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        heads_config = heads_config or {"diagnosis": True, "risk": True, "stage": True}

        # Optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        if optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

        if scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs
            )
        else:
            self.scheduler = None

        # Losses per head
        self.loss_fns: dict[str, nn.Module] = {}
        class_weights = class_weights or {}
        for head in ["diagnosis", "risk", "stage", "trend"]:
            if heads_config.get(head, False):
                w = class_weights.get(head)
                self.loss_fns[head] = create_loss_fn(
                    loss_type="focal" if head == "diagnosis" else "cross_entropy",
                    class_weights=w,
                    focal_gamma=focal_gamma,
                )

        self.heads_config = heads_config

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        loss_per_head: dict[str, float] = {}

        for batch_idx, batch in enumerate(self.train_loader):
            derm = batch["dermoscopy"].to(self.device)
            clinical = batch.get("clinical")
            if clinical is not None:
                clinical = clinical.to(self.device)
            cf = batch.get("clinical_features")
            if cf is not None:
                cf = cf.to(self.device)
            has_clinical = batch.get("has_clinical")
            if has_clinical is not None:
                has_clinical = has_clinical.to(self.device)

            out = self.model(
                dermoscopy=derm,
                clinical=clinical,
                clinical_features=cf,
                has_clinical=has_clinical,
            )

            loss = torch.tensor(0.0, device=self.device)
            for head, loss_fn in self.loss_fns.items():
                if head in batch and head in out:
                    l = loss_fn(out[head], batch[head].to(self.device))
                    loss = loss + l
                    loss_per_head[head] = loss_per_head.get(head, 0) + l.item()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        n = len(self.train_loader)
        avg_loss = total_loss / n
        for k in loss_per_head:
            loss_per_head[k] /= n
        if self.scheduler:
            self.scheduler.step()
        return {"loss": avg_loss, **loss_per_head}

    def train(self) -> list[dict[str, float]]:
        history = []
        best_val_acc = 0.0

        for epoch in range(self.num_epochs):
            metrics = self.train_epoch(epoch)
            history.append(metrics)
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs} train loss: {metrics['loss']:.4f}")

            if self.val_loader:
                results = evaluate_model(
                    self.model, self.val_loader, self.device,
                    head_names=list(self.heads_config.keys()),
                )
                if "diagnosis" in results:
                    acc = results["diagnosis"].get("accuracy", 0)
                    logger.info(f"Epoch {epoch + 1} val accuracy: {acc:.4f}")
                    if acc > best_val_acc and self.checkpoint_dir:
                        best_val_acc = acc
                        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                                "metrics": results,
                            },
                            self.checkpoint_dir / "best.pt",
                        )

        return history
