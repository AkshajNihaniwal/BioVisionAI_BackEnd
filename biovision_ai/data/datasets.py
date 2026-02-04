"""
Dataset classes for BIOVISION-AI.

Supports single-snapshot lesion data and longitudinal/sequence data
for trajectory modeling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LesionDataset(Dataset):
    """
    Dataset for single-snapshot lesion analysis.

    Each sample: dermoscopic image (mandatory), optional clinical image,
    tabular clinical features, and labels (diagnosis, risk, stage, trend).

    Compatible with ISIC, HAM10000, PH2, and similar dermoscopy datasets.
    """

    def __init__(
        self,
        dermoscopy_paths: list[str | Path],
        clinical_paths: Optional[list[str | Path]] = None,
        clinical_features: Optional[np.ndarray] = None,
        diagnosis_labels: Optional[np.ndarray] = None,
        risk_labels: Optional[np.ndarray] = None,
        stage_labels: Optional[np.ndarray] = None,
        trend_labels: Optional[np.ndarray] = None,
        segmentation_masks: Optional[list[str | Path]] = None,
        dermoscopy_transform: Optional[Any] = None,
        clinical_transform: Optional[Any] = None,
        lesion_ids: Optional[list[str]] = None,
    ) -> None:
        """
        Args:
            dermoscopy_paths: Paths to dermoscopic images.
            clinical_paths: Optional paths to clinical photos (None = no clinical).
            clinical_features: Optional [N, F] array of tabular features.
            diagnosis_labels: Optional [N] diagnosis class indices.
            risk_labels: Optional [N] risk class indices.
            stage_labels: Optional [N] stage class indices.
            trend_labels: Optional [N] trend class indices.
            segmentation_masks: Optional paths to lesion masks.
            dermoscopy_transform: Transform for dermoscopy images.
            clinical_transform: Transform for clinical images.
            lesion_ids: Optional IDs for logging/audit.
        """
        self.dermoscopy_paths = [Path(p) for p in dermoscopy_paths]
        self.n_samples = len(self.dermoscopy_paths)

        self.clinical_paths = clinical_paths
        if clinical_paths is not None:
            self.clinical_paths = [Path(p) for p in clinical_paths]
            assert len(self.clinical_paths) == self.n_samples

        self.clinical_features = clinical_features
        if clinical_features is not None:
            assert clinical_features.shape[0] == self.n_samples

        self.diagnosis_labels = diagnosis_labels
        self.risk_labels = risk_labels
        self.stage_labels = stage_labels
        self.trend_labels = trend_labels

        self.segmentation_masks = segmentation_masks
        if segmentation_masks is not None:
            self.segmentation_masks = [Path(p) for p in segmentation_masks]
            assert len(self.segmentation_masks) == self.n_samples

        self.dermoscopy_transform = dermoscopy_transform
        self.clinical_transform = clinical_transform
        self.lesion_ids = lesion_ids or [str(i) for i in range(self.n_samples)]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        derm_path = self.dermoscopy_paths[idx]
        derm_img = Image.open(derm_path).convert("RGB")
        if self.dermoscopy_transform:
            derm_img = self.dermoscopy_transform(derm_img)

        clinical_img: Optional[torch.Tensor] = None
        if self.clinical_paths is not None:
            clin_path = self.clinical_paths[idx]
            if clin_path.exists():
                clin_img = Image.open(clin_path).convert("RGB")
                if self.clinical_transform:
                    clin_img = self.clinical_transform(clin_img)
                clinical_img = clin_img
            # If path exists but load fails, clinical_img stays None

        clinical_feat: Optional[torch.Tensor] = None
        if self.clinical_features is not None:
            clinical_feat = torch.from_numpy(
                self.clinical_features[idx].astype(np.float32)
            )

        out: dict[str, Any] = {
            "dermoscopy": derm_img,
            "lesion_id": self.lesion_ids[idx],
            "has_clinical": clinical_img is not None,
        }
        if clinical_img is not None:
            out["clinical"] = clinical_img
        if clinical_feat is not None:
            out["clinical_features"] = clinical_feat

        if self.diagnosis_labels is not None:
            out["diagnosis"] = int(self.diagnosis_labels[idx])
        if self.risk_labels is not None:
            out["risk"] = int(self.risk_labels[idx])
        if self.stage_labels is not None:
            out["stage"] = int(self.stage_labels[idx])
        if self.trend_labels is not None:
            out["trend"] = int(self.trend_labels[idx])

        if self.segmentation_masks is not None:
            mask_path = self.segmentation_masks[idx]
            if mask_path.exists():
                mask = Image.open(mask_path)
                out["segmentation_mask"] = torch.from_numpy(
                    np.array(mask).astype(np.float32) / 255.0
                )

        return out


def collate_lesion_batch(
    batch: list[dict[str, Any]],
    pad_clinical: bool = True,
) -> dict[str, Any]:
    """
    Collate function for LesionDataset.

    Handles optional clinical images (zero-padding or masking when absent).
    """
    dermoscopy = torch.stack([b["dermoscopy"] for b in batch])

    has_clinical = [b["has_clinical"] for b in batch]
    clinical_list = [b.get("clinical") for b in batch]

    if any(has_clinical):
        # Create zero tensor for missing clinical images
        sample_with_clinical = next(b["clinical"] for b in batch if b.get("clinical") is not None)
        clinical_stacked = []
        for i, has_c in enumerate(has_clinical):
            if has_c and clinical_list[i] is not None:
                clinical_stacked.append(clinical_list[i])
            else:
                clinical_stacked.append(torch.zeros_like(sample_with_clinical))
        clinical = torch.stack(clinical_stacked)
    else:
        clinical = None

    clinical_features_list = [b.get("clinical_features") for b in batch]
    if all(cf is not None for cf in clinical_features_list):
        clinical_features = torch.stack(clinical_features_list)
    else:
        clinical_features = None

    out: dict[str, Any] = {
        "dermoscopy": dermoscopy,
        "clinical": clinical,
        "clinical_features": clinical_features,
        "has_clinical": torch.tensor(has_clinical, dtype=torch.bool),
        "lesion_ids": [b["lesion_id"] for b in batch],
    }

    if "diagnosis" in batch[0]:
        out["diagnosis"] = torch.tensor([b["diagnosis"] for b in batch], dtype=torch.long)
    if "risk" in batch[0]:
        out["risk"] = torch.tensor([b["risk"] for b in batch], dtype=torch.long)
    if "stage" in batch[0]:
        out["stage"] = torch.tensor([b["stage"] for b in batch], dtype=torch.long)
    if "trend" in batch[0]:
        out["trend"] = torch.tensor([b["trend"] for b in batch], dtype=torch.long)

    return out


class LesionSequenceDataset(Dataset):
    """
    Dataset for longitudinal lesion data (multiple timepoints per lesion).

    Used for trajectory/sequence modeling: predict future stage from
    past images and clinical data.

    Structure: lesion_id -> [(image_t1, features_t1), (image_t2, features_t2), ...]
    """

    def __init__(
        self,
        lesion_sequences: dict[str, list[dict[str, Any]]],
        max_sequence_length: int = 5,
        dermoscopy_transform: Optional[Any] = None,
        clinical_transform: Optional[Any] = None,
    ) -> None:
        """
        Args:
            lesion_sequences: Dict mapping lesion_id to list of timepoint dicts.
                Each timepoint: {"dermoscopy": path, "clinical": path?, "features": array,
                                 "diagnosis": int?, "stage": int?, "trend": int?, "timestamp": ...}
            max_sequence_length: Max timepoints per sequence.
            dermoscopy_transform: Transform for dermoscopy.
            clinical_transform: Transform for clinical.
        """
        self.lesion_sequences = lesion_sequences
        self.lesion_ids = list(lesion_sequences.keys())
        self.max_sequence_length = max_sequence_length
        self.dermoscopy_transform = dermoscopy_transform
        self.clinical_transform = clinical_transform

    def __len__(self) -> int:
        return len(self.lesion_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        lesion_id = self.lesion_ids[idx]
        timepoints = self.lesion_sequences[lesion_id]

        # Truncate or pad to max_sequence_length
        if len(timepoints) > self.max_sequence_length:
            timepoints = timepoints[-self.max_sequence_length :]

        dermoscopy_list = []
        clinical_list = []
        features_list = []
        stages = []
        trends = []

        for tp in timepoints:
            derm_path = tp["dermoscopy"]
            derm_img = Image.open(derm_path).convert("RGB")
            if self.dermoscopy_transform:
                derm_img = self.dermoscopy_transform(derm_img)
            dermoscopy_list.append(derm_img)

            if "clinical" in tp and tp["clinical"]:
                clin_img = Image.open(tp["clinical"]).convert("RGB")
                if self.clinical_transform:
                    clin_img = self.clinical_transform(clin_img)
                clinical_list.append(clin_img)
            else:
                clinical_list.append(None)

            if "features" in tp:
                features_list.append(tp["features"])
            if "stage" in tp:
                stages.append(tp["stage"])
            if "trend" in tp:
                trends.append(tp["trend"])

        # Pad sequence if shorter
        seq_len = len(dermoscopy_list)
        while len(dermoscopy_list) < self.max_sequence_length:
            dermoscopy_list.insert(0, dermoscopy_list[0])  # Repeat first
            clinical_list.insert(0, clinical_list[0] if clinical_list[0] is not None else None)
            if features_list:
                features_list.insert(0, features_list[0])

        out: dict[str, Any] = {
            "lesion_id": lesion_id,
            "dermoscopy_sequence": torch.stack(dermoscopy_list),
            "sequence_length": seq_len,
            "mask": torch.tensor(
                [1] * seq_len + [0] * (self.max_sequence_length - seq_len),
                dtype=torch.bool,
            ),
        }
        if clinical_list and any(c is not None for c in clinical_list):
            # TODO: proper padding for optional clinical
            pass
        if features_list:
            out["clinical_features_sequence"] = torch.tensor(
                np.array(features_list), dtype=torch.float32
            )
        if stages:
            out["stage_target"] = stages[-1]  # Predict last stage
        if trends:
            out["trend_target"] = trends[-1]

        return out
