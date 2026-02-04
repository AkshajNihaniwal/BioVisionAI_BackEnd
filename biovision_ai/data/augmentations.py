"""
Augmentations for dermoscopic and clinical skin images.

Designed to preserve dermatologic features: avoid unrealistic transforms
that would alter diagnostic appearance (e.g., aggressive color shifts).
"""

from __future__ import annotations

from typing import Optional

import torch
from torchvision import transforms


def get_dermoscopy_transforms(
    image_size: int = 224,
    is_training: bool = True,
    normalize_mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    normalize_std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Get transforms for dermoscopic images.

    Training: rotation, horizontal/vertical flip, mild color jitter, resize.
    Avoids aggressive augmentations that could distort dermatologic features.

    Args:
        image_size: Target size for resize.
        is_training: If True, apply augmentations; else only resize + normalize.
        normalize_mean: ImageNet normalization mean.
        normalize_std: ImageNet normalization std.

    Returns:
        Composed transforms.
    """
    if is_training:
        transform_list = [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.02,
            ),
        ]
    else:
        transform_list = [
            transforms.Resize((image_size, image_size)),
        ]

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])
    return transforms.Compose(transform_list)


def get_clinical_transforms(
    image_size: int = 224,
    is_training: bool = True,
    normalize_mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    normalize_std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Get transforms for clinical (macroscopic) photographs.

    Similar to dermoscopy but may allow slightly more variation
    for clinical context.
    """
    return get_dermoscopy_transforms(
        image_size=image_size,
        is_training=is_training,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
