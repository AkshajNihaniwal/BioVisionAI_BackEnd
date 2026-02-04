"""
Dummy dataset for testing and development.

Creates synthetic data when real datasets are not available.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from biovision_ai.data.datasets import LesionDataset
from biovision_ai.data.augmentations import get_dermoscopy_transforms


def create_dummy_lesion_dataset(
    n_samples: int = 100,
    image_size: int = 224,
    num_classes: int = 7,
    num_risk_levels: int = 3,
    num_stages: int = 3,
    num_features: int = 32,
    temp_dir: Path | None = None,
) -> LesionDataset:
    """
    Create dummy LesionDataset for testing.

    Saves random images to temp directory. Use for quick training/eval tests.
    """
    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="biovision_dummy_"))
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i in range(n_samples):
        p = temp_dir / f"derm_{i}.png"
        img = Image.fromarray(
            (np.random.rand(image_size, image_size, 3) * 255).astype(np.uint8)
        )
        img.save(p)
        paths.append(str(p))

    features = np.random.randn(n_samples, num_features).astype(np.float32)
    diag = np.random.randint(0, num_classes, n_samples)
    risk = np.random.randint(0, num_risk_levels, n_samples)
    stage = np.random.randint(0, num_stages, n_samples)

    transform = get_dermoscopy_transforms(image_size, is_training=True)
    return LesionDataset(
        dermoscopy_paths=paths,
        clinical_features=features,
        diagnosis_labels=diag,
        risk_labels=risk,
        stage_labels=stage,
        dermoscopy_transform=transform,
    )
