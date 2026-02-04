"""
Evaluation metrics and calibration for BIOVISION-AI.

- AUC, sensitivity, specificity, F1, accuracy
- Per-class confusion matrix
- Reliability diagrams, temperature scaling, isotonic regression
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
) -> dict[str, Any]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (for AUC).
        num_classes: Number of classes (for multiclass AUC).

    Returns:
        Dict with accuracy, f1, sensitivity, specificity, auc, confusion_matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics: dict[str, Any] = {
        "accuracy": float(acc),
        "f1_weighted": float(f1),
        "confusion_matrix": cm.tolist(),
    }

    if y_prob is not None:
        try:
            if num_classes is None:
                num_classes = len(np.unique(y_true))
            if num_classes == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="weighted"
                )
            metrics["auc"] = float(auc)
        except ValueError:
            metrics["auc"] = 0.0

    # Per-class sensitivity/specificity from confusion matrix
    n = cm.shape[0]
    sensitivity = []
    specificity = []
    for i in range(n):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity.append(sens)
        specificity.append(spec)
    metrics["sensitivity_per_class"] = sensitivity
    metrics["specificity_per_class"] = specificity

    return metrics


def reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute calibration curve (reliability diagram).

    For binary: use y_prob[:, 1]. For multiclass: use max prob per sample.
    Returns (mean_predicted_value, fraction_of_positives).
    """
    if y_prob.shape[1] == 2:
        prob_pos = y_prob[:, 1]
    else:
        prob_pos = y_prob.max(axis=1)
    prob_true, prob_pred = calibration_curve(y_true, prob_pos, n_bins=n_bins)
    if save_path:
        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, "s-")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("Reliability diagram")
        plt.savefig(save_path)
        plt.close()
    return prob_true, prob_pred


class TemperatureScaling(nn.Module):
    """Temperature scaling for probability calibration."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def calibrate_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    lr: float = 0.01,
    max_iter: int = 50,
) -> float:
    """
    Learn temperature for scaling. Returns optimal temperature value.
    """
    T = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)

    def eval_fn() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(eval_fn)
    return T.item()


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    head_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Run evaluation on a dataloader.

    Returns metrics per head (diagnosis, risk, stage).
    """
    model.eval()
    head_names = head_names or ["diagnosis", "risk", "stage"]
    all_preds = {h: [] for h in head_names}
    all_labels = {h: [] for h in head_names}
    all_probs = {h: [] for h in head_names}

    with torch.no_grad():
        for batch in dataloader:
            derm = batch["dermoscopy"].to(device)
            clinical = batch.get("clinical")
            if clinical is not None:
                clinical = clinical.to(device)
            cf = batch.get("clinical_features")
            if cf is not None:
                cf = cf.to(device)
            has_clinical = batch.get("has_clinical")
            if has_clinical is not None:
                has_clinical = has_clinical.to(device)

            out = model(
                dermoscopy=derm,
                clinical=clinical,
                clinical_features=cf,
                has_clinical=has_clinical,
            )

            for h in head_names:
                if h in batch and h in out:
                    all_preds[h].append(out[h].argmax(dim=1).cpu().numpy())
                    all_probs[h].append(
                        torch.softmax(out[h], dim=1).cpu().numpy()
                    )
                    all_labels[h].append(batch[h].cpu().numpy())

    results = {}
    for h in head_names:
        if all_preds[h]:
            preds = np.concatenate(all_preds[h])
            labels = np.concatenate(all_labels[h])
            probs = np.concatenate(all_probs[h])
            results[h] = compute_metrics(labels, preds, probs)
    return results
