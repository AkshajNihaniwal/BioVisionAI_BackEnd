"""
FastAPI application for BIOVISION-AI.

Endpoints for lesion inference, batch processing, health, and model info.
Designed for PACS, EMR, and dermatoscope software integration.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from biovision_ai.api.model_loader import ModelHolder
from biovision_ai.api.inference import run_inference
from biovision_ai.api.schemas import (
    ClinicalFeatures,
    LesionInferenceResponse,
    ModelInfoResponse,
)

logger = logging.getLogger(__name__)

# Global model holder (set at startup)
_model_holder: Optional[ModelHolder] = None


def get_model_holder() -> ModelHolder:
    if _model_holder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _model_holder


def create_app(
    model_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    heatmap_dir: Optional[Path] = None,
) -> FastAPI:
    """
    Create FastAPI app with model loaded at startup.

    Args:
        model_path: Path to checkpoint (None = random init for demo).
        config_path: Path to config YAML.
        heatmap_dir: Directory for saving heatmaps.
    """
    global _model_holder

    app = FastAPI(
        title="BIOVISION-AI",
        description="AI-powered dermatology decision-support system",
        version="0.1.0",
    )

    @app.on_event("startup")
    async def startup() -> None:
        global _model_holder
        from biovision_ai.config import load_config, get_default_config
        config = get_default_config()
        if config_path and config_path.exists():
            config = load_config(config_path)
        _model_holder = ModelHolder(model_path=model_path, config=config)
        logger.info("Model loaded successfully")

    @app.get("/health")
    async def health() -> dict:
        """Health check for deployment."""
        return {"status": "ok", "model_loaded": _model_holder is not None}

    @app.get("/model/info", response_model=ModelInfoResponse)
    async def model_info(holder: ModelHolder = Depends(get_model_holder)) -> ModelInfoResponse:
        """Returns model version, supported classes, and capabilities."""
        return ModelInfoResponse(
            version=holder.version,
            diagnosis_classes=holder.diagnosis_classes,
            risk_labels=holder.risk_labels,
            stage_labels=holder.stage_labels,
            trend_labels=holder.trend_labels,
            supports_clinical_image=True,
            supports_heatmap=True,
        )

    class InferRequest(BaseModel):
        """Request body for JSON-based inference (e.g. base64 images)."""
        dermoscopy_base64: Optional[str] = None
        clinical_base64: Optional[str] = None
        clinical_features: Optional[ClinicalFeatures] = None
        lesion_id: Optional[str] = None
        generate_heatmap: bool = False

    @app.post("/infer/lesion", response_model=LesionInferenceResponse)
    async def infer_lesion(
        holder: ModelHolder = Depends(get_model_holder),
        dermoscopy: Optional[UploadFile] = File(None),
        clinical: Optional[UploadFile] = File(None),
        lesion_id: Optional[str] = Form(None),
        generate_heatmap: bool = Form(False),
        age: Optional[int] = Form(None),
        sex: Optional[str] = Form(None),
        fitzpatrick_type: Optional[int] = Form(None),
        anatomical_site: Optional[str] = Form(None),
        symptom_duration_days: Optional[int] = Form(None),
        rapid_change: Optional[bool] = Form(None),
        itching: Optional[bool] = Form(None),
        bleeding: Optional[bool] = Form(None),
        family_history: Optional[bool] = Form(None),
    ) -> LesionInferenceResponse:
        """
        Single lesion inference.

        Accepts dermoscopic image (file or base64), optional clinical image,
        and JSON/form metadata. Returns diagnosis probabilities, risk, stage, trend.
        """
        from PIL import Image

        if dermoscopy is None:
            raise HTTPException(status_code=400, detail="Dermoscopic image required")

        content = await dermoscopy.read()
        derm_img = Image.open(io.BytesIO(content)).convert("RGB")

        clin_img = None
        if clinical is not None:
            clin_content = await clinical.read()
            clin_img = Image.open(io.BytesIO(clin_content)).convert("RGB")

        features = None
        if any([age is not None, sex is not None, fitzpatrick_type is not None]):
            features = {
                "age": age,
                "sex": sex,
                "fitzpatrick_type": fitzpatrick_type,
                "anatomical_site": anatomical_site,
                "symptom_duration_days": symptom_duration_days,
                "rapid_change": rapid_change,
                "itching": itching,
                "bleeding": bleeding,
                "family_history": family_history,
            }

        result = run_inference(
            model_holder=holder,
            dermoscopy_image=derm_img,
            clinical_image=clin_img,
            clinical_features=features,
            lesion_id=lesion_id,
            generate_heatmap=generate_heatmap,
            heatmap_dir=Path(heatmap_dir) if heatmap_dir else None,
        )

        return LesionInferenceResponse(
            diagnosis_probabilities=result["diagnosis_probabilities"],
            risk_category=result["risk_category"],
            risk_score=result["risk_score"],
            stage_estimate=result.get("stage_estimate"),
            trend_prediction=result.get("trend_prediction"),
            heatmap_path=result.get("heatmap_path"),
            model_version=result["model_version"],
            lesion_id=result.get("lesion_id"),
        )

    @app.post("/infer/batch")
    async def infer_batch(
        holder: ModelHolder = Depends(get_model_holder),
        files: list[UploadFile] = File(...),
    ) -> JSONResponse:
        """
        Batch inference for multiple lesions.

        TODO: Accept structured batch format (e.g. ZIP with manifest).
        For now returns placeholder.
        """
        return JSONResponse(
            content={
                "results": [],
                "total_processed": 0,
                "model_version": holder.version,
                "message": "Batch endpoint: use /infer/lesion per image or implement batch manifest",
            }
        )

    return app
