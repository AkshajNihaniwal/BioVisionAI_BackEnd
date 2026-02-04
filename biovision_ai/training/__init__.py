"""
Training and evaluation for BIOVISION-AI.

- Training loops for classification + risk
- Evaluation metrics (AUC, sensitivity, specificity, F1)
- Calibration (temperature scaling, isotonic regression)
"""

from biovision_ai.training.trainer import Trainer
from biovision_ai.training.evaluation import evaluate_model, compute_metrics

__all__ = ["Trainer", "evaluate_model", "compute_metrics"]
