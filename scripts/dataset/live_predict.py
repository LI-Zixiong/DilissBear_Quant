"""Thin CLI wrapper — see :func:`src.predict.live_predictor.generate_live_predictions`.

Usage: python -m scripts.dataset.live_predict
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict.live_predictor import generate_live_predictions

if __name__ == "__main__":
    generate_live_predictions(
        exp_dir=Path("dataset/output/experiment_009"),
        smooth_window=10,
    )
