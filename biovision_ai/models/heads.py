"""
Multi-task heads for BIOVISION-AI.

- Diagnosis: multi-class over dermatologic classes
- Risk: malignancy/severity (low/intermediate/high)
- Stage: early/intermediate/advanced
- Trend: stable / slowly progressive / rapidly progressive
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class DiagnosisHead(nn.Module):
    """Multi-class diagnosis classifier."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class RiskHead(nn.Module):
    """Malignancy/severity risk classifier (binary or multi-level)."""

    def __init__(
        self,
        input_dim: int,
        num_levels: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_levels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class StageHead(nn.Module):
    """Stage/chronicity classifier (early/intermediate/advanced)."""

    def __init__(
        self,
        input_dim: int,
        num_stages: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_stages = num_stages
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_stages),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TrendHead(nn.Module):
    """Trend classifier (stable / slowly progressive / rapidly progressive)."""

    def __init__(
        self,
        input_dim: int,
        num_trends: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_trends = num_trends
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_trends),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def create_heads(
    fusion_dim: int,
    num_diagnosis_classes: int,
    num_risk_levels: int = 3,
    num_stages: int = 3,
    num_trends: int = 3,
    heads_config: Optional[dict[str, bool]] = None,
    dropout: float = 0.2,
) -> nn.ModuleDict:
    """
    Create multi-task heads based on config.

    Args:
        fusion_dim: Input dimension from fusion layer.
        num_diagnosis_classes: Number of diagnosis classes.
        num_risk_levels: Number of risk levels.
        num_stages: Number of stage classes.
        num_trends: Number of trend classes.
        heads_config: Dict with keys diagnosis, risk, stage, trend -> bool.
        dropout: Dropout rate.

    Returns:
        ModuleDict of active heads.
    """
    heads_config = heads_config or {
        "diagnosis": True,
        "risk": True,
        "stage": True,
        "trend": False,
    }
    heads = nn.ModuleDict()
    if heads_config.get("diagnosis", True):
        heads["diagnosis"] = DiagnosisHead(
            fusion_dim, num_diagnosis_classes, dropout
        )
    if heads_config.get("risk", True):
        heads["risk"] = RiskHead(fusion_dim, num_risk_levels, dropout)
    if heads_config.get("stage", True):
        heads["stage"] = StageHead(fusion_dim, num_stages, dropout)
    if heads_config.get("trend", False):
        heads["trend"] = TrendHead(fusion_dim, num_trends, dropout)
    return heads
