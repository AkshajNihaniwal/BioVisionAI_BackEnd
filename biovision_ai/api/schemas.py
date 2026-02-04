"""
Pydantic schemas for BIOVISION-AI API.

Request/response models for lesion inference endpoints.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ClinicalFeatures(BaseModel):
    """Structured clinical features for inference."""

    age: Optional[int] = Field(None, ge=0, le=120)
    sex: Optional[str] = Field(None, description="M/F/O")
    fitzpatrick_type: Optional[int] = Field(None, ge=1, le=6)
    anatomical_site: Optional[str] = None
    symptom_duration_days: Optional[int] = Field(None, ge=0)
    rapid_change: Optional[bool] = None
    itching: Optional[bool] = None
    bleeding: Optional[bool] = None
    family_history: Optional[bool] = None


class DiagnosisProbability(BaseModel):
    """Diagnosis class with probability."""

    class_name: str
    probability: float


class LesionInferenceResponse(BaseModel):
    """Response for single lesion inference."""

    diagnosis_probabilities: list[DiagnosisProbability]
    risk_category: str
    risk_score: float
    stage_estimate: Optional[str] = None
    trend_prediction: Optional[str] = None
    heatmap_path: Optional[str] = None
    model_version: str
    lesion_id: Optional[str] = None


class BatchLesionItem(BaseModel):
    """Single item in batch request."""

    lesion_id: str
    clinical_features: Optional[ClinicalFeatures] = None


class BatchInferenceResponse(BaseModel):
    """Response for batch inference."""

    results: list[LesionInferenceResponse]
    total_processed: int
    model_version: str


class ModelInfoResponse(BaseModel):
    """Model metadata and capabilities."""

    version: str
    diagnosis_classes: list[str]
    risk_labels: list[str]
    stage_labels: list[str]
    trend_labels: list[str]
    supports_clinical_image: bool
    supports_heatmap: bool
