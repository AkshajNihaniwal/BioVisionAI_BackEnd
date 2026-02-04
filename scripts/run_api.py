#!/usr/bin/env python3
"""
Run BIOVISION-AI FastAPI server.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --model_path checkpoints/best.pt --port 8000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from biovision_ai.api.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--heatmap_dir", type=str, default=None)
    args = parser.parse_args()

    app = create_app(
        model_path=Path(args.model_path) if args.model_path else None,
        config_path=Path(args.config) if args.config else None,
        heatmap_dir=Path(args.heatmap_dir) if args.heatmap_dir else None,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
