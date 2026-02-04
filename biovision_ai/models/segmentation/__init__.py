"""
Segmentation module for lesion region extraction.

UNet/nnU-Net style. Can be trained separately and integrated
into the inference pipeline for optional lesion masking/cropping.
"""

from biovision_ai.models.segmentation.unet import LesionSegmentationUNet

__all__ = ["LesionSegmentationUNet"]
