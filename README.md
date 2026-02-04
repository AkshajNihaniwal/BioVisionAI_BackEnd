# BIOVISION-AI

AI-powered dermatology decision-support system for licensed dermatologists.

## Overview

BIOVISION-AI provides:

1. **Predictive growth / trajectory model** for skin lesions
   - Dermoscopic images (mandatory) + optional clinical photos + structured clinical data
   - Outputs: diagnosis probabilities, malignancy risk, stage, trend
   - Optional synthetic trajectory images (research-only)

2. **REST API** for integration with PACS, EMR, and dermatoscope software

## Installation

```bash
pip install -r requirements.txt
# Or: pip install -e .
```

## Quick Start

### Run API (with stubbed model)

```bash
python scripts/run_api.py --port 8000
```

Then:
- `GET /health` - Health check
- `GET /model/info` - Model metadata
- `POST /infer/lesion` - Single lesion inference (multipart: dermoscopy image + optional clinical + form fields)

### Train (dummy data)

```bash
python scripts/train.py --epochs 5 --checkpoint_dir checkpoints
```

### Export to ONNX

```bash
python scripts/export_onnx.py --checkpoint checkpoints/best.pt --output model.onnx
```

## Project Structure

```
biovision_ai/
├── config/          # Configuration loaders
├── data/             # Datasets, augmentations, collate
├── models/
│   ├── backbones/   # ViT, EfficientNet
│   ├── segmentation/ # UNet for lesion masks
│   ├── fusion.py    # Multimodal fusion
│   ├── heads.py     # Diagnosis, risk, stage, trend heads
│   ├── multimodal_model.py
│   ├── explainability.py  # Grad-CAM, attention rollout
│   └── trajectory/  # Stage classifier, sequence model skeleton
├── training/        # Trainer, evaluation, calibration
├── api/             # FastAPI app, schemas, inference
└── utils/
```

## Data Integration

To use real datasets (ISIC, HAM10000, PH2):

1. Organize data: dermoscopy paths, optional clinical paths, CSV with labels and clinical features
2. Implement a dataset loader that returns `LesionDataset`-compatible format
3. Update `scripts/train.py` to load from your paths
4. Configure `config/default.yaml` with your class mappings

## Clinical Disclaimer

This system is **decision-support only** for licensed dermatologists. It does not replace biopsy or histopathology. Trajectory estimates are probabilistic; synthetic images are illustrative, not true past/future images.

## License

Proprietary. For research and clinical decision-support use.
