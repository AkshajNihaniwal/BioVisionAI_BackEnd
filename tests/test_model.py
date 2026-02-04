"""Unit tests for BIOVISION-AI model."""

import pytest
import torch

from biovision_ai.models import BioVisionModel
from biovision_ai.models.fusion import create_fusion_layer, ConcatFusion
from biovision_ai.models.heads import create_heads


def test_bio_vision_model_forward() -> None:
    """Test full model forward pass."""
    model = BioVisionModel(
        dermoscopy_backbone="efficientnet_b0",
        clinical_backbone="efficientnet_b0",
        dermoscopy_embed_dim=1280,
        clinical_embed_dim=1280,
        clinical_data_embed_dim=64,
        num_clinical_features=32,
        num_diagnosis_classes=7,
        risk_levels=3,
        num_stages=3,
        num_trends=3,
        pretrained=False,
    )
    B = 2
    derm = torch.randn(B, 3, 224, 224)
    clin = torch.randn(B, 3, 224, 224)
    cf = torch.randn(B, 32)

    out = model(dermoscopy=derm, clinical=clin, clinical_features=cf)
    assert "diagnosis" in out
    assert out["diagnosis"].shape == (B, 7)
    assert out["risk"].shape == (B, 3)
    assert out["stage"].shape == (B, 3)


def test_model_without_clinical() -> None:
    """Test model when clinical image is None."""
    model = BioVisionModel(
        dermoscopy_backbone="efficientnet_b0",
        clinical_backbone="efficientnet_b0",
        dermoscopy_embed_dim=1280,
        clinical_embed_dim=1280,
        clinical_data_embed_dim=64,
        num_clinical_features=32,
        pretrained=False,
    )
    B = 2
    derm = torch.randn(B, 3, 224, 224)
    out = model(dermoscopy=derm, clinical=None, clinical_features=None)
    assert "diagnosis" in out


def test_concat_fusion() -> None:
    """Test ConcatFusion layer."""
    fusion = ConcatFusion(
        dermoscopy_dim=128,
        clinical_dim=128,
        clinical_data_dim=64,
        hidden_dim=256,
    )
    B = 4
    derm = torch.randn(B, 128)
    clin = torch.randn(B, 128)
    data = torch.randn(B, 64)
    out = fusion(derm, clin, data)
    assert out.shape == (B, 256)


def test_heads() -> None:
    """Test head creation."""
    heads = create_heads(
        fusion_dim=512,
        num_diagnosis_classes=7,
        num_risk_levels=3,
        heads_config={"diagnosis": True, "risk": True, "stage": True, "trend": False},
    )
    x = torch.randn(2, 512)
    assert heads["diagnosis"](x).shape == (2, 7)
    assert heads["risk"](x).shape == (2, 3)
    assert heads["stage"](x).shape == (2, 3)
    assert "trend" not in heads
