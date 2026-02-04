"""
Stage classifier: single-snapshot -> stage and trend.

Given current image + clinical data, predicts:
- Stage: early / intermediate / advanced
- Trend: stable / slowly progressive / rapidly progressive

Training target: stage/trend labels from longitudinal data or expert annotation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from biovision_ai.models.multimodal_model import BioVisionModel


class StageClassifier(BioVisionModel):
    """
    BioVisionModel with stage and trend heads enabled.

    The base BioVisionModel already has stage and trend heads.
    This subclass ensures they are active and provides a clear interface
    for trajectory/stage prediction from a single snapshot.
    """

    def __init__(
        self,
        heads_config: dict[str, bool] | None = None,
        **kwargs: object,
    ) -> None:
        heads_config = heads_config or {
            "diagnosis": True,
            "risk": True,
            "stage": True,
            "trend": True,
        }
        super().__init__(heads_config=heads_config, **kwargs)
