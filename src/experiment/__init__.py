"""
Experiment orchestration package.

This package contains reusable components for running model experiments:
configuration, data preparation, date splitting, return alignment,
model construction, training orchestration, and report generation.
"""

from src.experiment.config import ExperimentConfig
from src.experiment.data import (
    ExperimentData,
    load_experiment_raw_data,
    prepare_experiment_data,
    preprocess_experiment_data,
)
from src.experiment.returns import (
    align_predictions_to_returns,
    build_experiment_returns,
    build_returns_frame_from_next_target,
    build_returns_from_column,
)
from src.experiment.split import split_panel_by_date_ratio
from src.experiment.model_factory import (
    build_experiment_model,
    get_active_tabular_models,
    get_active_torch_models,
    get_model_family,
    get_model_params,
)
from src.experiment.tuning import run_grid_search

__all__ = [
    "ExperimentConfig",
    "ExperimentData",
    "load_experiment_raw_data",
    "prepare_experiment_data",
    "preprocess_experiment_data",
    "split_panel_by_date_ratio",
    "align_predictions_to_returns",
    "build_experiment_returns",
    "build_returns_frame_from_next_target",
    "build_returns_from_column",
    "build_experiment_model",
    "get_active_tabular_models",
    "get_active_torch_models",
    "get_model_family",
    "get_model_params",
    "run_grid_search",
]