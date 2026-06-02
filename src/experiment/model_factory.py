"""
Experiment model factory.

This module centralizes:
    1. model-family classification
    2. per-model training parameter resolution
    3. model construction

It should not contain training loops, prediction logic, or backtest logic.
"""

from __future__ import annotations

from typing import Any

from src.experiment.config import ExperimentConfig

from src.models.dlinear import DLinearConfig, build_dlinear_model
from src.models.itransformer import ITransformerConfig, build_itransformer_model
from src.models.lightgbm_model import LightGBMConfig, build_lightgbm_model
from src.models.patchtst import PatchTSTConfig, build_patchtst_model
from src.models.tsmixer import TSMixerConfig, build_tsmixer_model
from src.models.xgboost_model import XGBoostConfig, build_xgboost_model


TABULAR_MODELS = {
    "lightgbm",
    "xgboost",
}

TORCH_MODELS = {
    "dlinear",
    "itransformer",
    "patchtst",
    "tsmixer",
}


def get_model_family(model_name: str) -> str:
    """
    Return the model family used by the experiment runner.

    Returns
    -------
    str
        Either "tabular" or "torch".
    """

    normalized_name = model_name.lower()

    if normalized_name in TABULAR_MODELS:
        return "tabular"

    if normalized_name in TORCH_MODELS:
        return "torch"

    raise ValueError(f"Unsupported model_name: {model_name}")


def get_model_params(
    model_name: str,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """
    Resolve per-model training parameters.

    Default torch settings come from ExperimentConfig.
    Model-specific overrides come from config.model_params.

    Tabular models currently do not need these parameters, but returning a
    consistent dictionary keeps runner code simpler.
    """

    defaults: dict[str, Any] = {
        "seq_len": config.seq_len,
        "epochs": config.torch_epochs,
        "lr": config.torch_learning_rate,
        "wd": config.torch_weight_decay,
        "patience": config.torch_patience,
    }

    overrides = config.model_params.get(model_name.lower(), {})
    defaults.update(overrides)

    return defaults


def build_experiment_model(
    model_name: str,
    seed: int,
    seq_len: int,
    n_features: int,
    config: ExperimentConfig | None = None,
) -> Any:
    """
    Build one experiment model by name.

    Tree model parameters are read from config.model_params.
    Torch model parameters (seq_len, etc.) are passed by the caller.

    Parameters
    ----------
    model_name:
        Model identifier, e.g. "lightgbm", "dlinear".
    seed:
        Random seed used by models that support it.
    seq_len:
        Sequence length for torch sequence models.
    n_features:
        Number of input features.
    config:
        ExperimentConfig for model-specific parameters (tree models).
    """

    normalized_name = model_name.lower()
    params = config.model_params.get(normalized_name, {}) if config else {}

    if normalized_name == "lightgbm":
        return build_lightgbm_model(
            LightGBMConfig(
                n_estimators=int(params.get("n_estimators", 500)),
                learning_rate=float(params.get("learning_rate", 0.01)),
                num_leaves=int(params.get("num_leaves", 31)),
                early_stopping_rounds=int(params.get("early_stopping_rounds", 50)),
                verbose_eval=False,
                random_state=seed,
            )
        )

    if normalized_name == "xgboost":
        return build_xgboost_model(
            XGBoostConfig(
                n_estimators=int(params.get("n_estimators", 500)),
                max_depth=int(params.get("max_depth", 5)),
                learning_rate=float(params.get("learning_rate", 0.01)),
                subsample=float(params.get("subsample", 0.8)),
                colsample_bytree=float(params.get("colsample_bytree", 0.8)),
                reg_alpha=float(params.get("reg_alpha", 0.0)),
                reg_lambda=float(params.get("reg_lambda", 1.0)),
                random_state=seed,
                n_jobs=-1,
            )
        )

    if normalized_name == "dlinear":
        return build_dlinear_model(
            DLinearConfig(
                seq_len=seq_len,
                n_features=n_features,
                moving_avg_kernel=25,
                dropout=0.1,
            )
        )

    if normalized_name == "itransformer":
        return build_itransformer_model(
            ITransformerConfig(
                seq_len=seq_len,
                n_features=n_features,
                d_model=128,
                nhead=4,
                num_layers=2,
                dim_feedforward=256,
                dropout=0.1,
            )
        )

    if normalized_name == "patchtst":
        return build_patchtst_model(
            PatchTSTConfig(
                seq_len=seq_len,
                n_features=n_features,
                patch_len=8,
                stride=4,
                d_model=128,
                nhead=4,
                num_layers=2,
                dim_feedforward=256,
                dropout=0.1,
            )
        )

    if normalized_name == "tsmixer":
        return build_tsmixer_model(
            TSMixerConfig(
                seq_len=seq_len,
                n_features=n_features,
                num_blocks=2,
                dropout=0.1,
            )
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


def get_active_tabular_models(config: ExperimentConfig) -> tuple[str, ...]:
    """Return active tabular models from config.model_names."""

    return tuple(
        model_name
        for model_name in config.model_names
        if get_model_family(model_name) == "tabular"
    )


def get_active_torch_models(config: ExperimentConfig) -> tuple[str, ...]:
    """Return active torch models from config.model_names."""

    return tuple(
        model_name
        for model_name in config.model_names
        if get_model_family(model_name) == "torch"
    )