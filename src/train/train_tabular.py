"""
Training utilities for tabular tree-based models.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.dataset_builder import BuiltDataset

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate root mean squared error."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"but y_pred has shape {y_pred.shape}."
        )
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _validate_tabular_dataset(dataset: BuiltDataset, name: str) -> None:
    if not isinstance(dataset, BuiltDataset):
        raise ValueError(
            f"Invalid {name} dataset. Expected a BuiltDataset instance, "
            f"but got {type(dataset).__name__}."
        )
    
    if dataset.X.ndim != 2:
        raise ValueError(
            f"Invalid {name} dataset features. Expected a 2D array, "
            f"but got an array with shape {dataset.X.shape}."
        )
    
    if dataset.y.ndim != 1:
        raise ValueError(
            f"Invalid {name} dataset targets. Expected a 1D array, "
            f"but got an array with shape {dataset.y.shape}."
        )
    
    if dataset.X.shape[0] != dataset.y.shape[0]:
        raise ValueError(
            f"Sample size mismatch in {name} dataset. "
            f"Features have {dataset.X.shape[0]} samples, "
            f"but targets have {dataset.y.shape[0]} samples."
        )
    
    if len(dataset.meta) != dataset.X.shape[0]:
        raise ValueError(
            f"{name}.meta and {name}.X length mismatch: "
            f"{len(dataset.meta)} vs {dataset.X.shape[0]}"
        )
    
    if dataset.X.shape[0] == 0:
        raise ValueError(f"{name} dataset is empty. No samples found.")
    
def _save_tabular_model(model: Any, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = model.__class__.__name__
    inner_model = getattr(model, "model", model)

    if hasattr(inner_model, "booster_"):
        model_path = output_dir / f"{model_name}.txt"
        inner_model.booster_.save_model(model_path)
        return model_path
    
    if hasattr(inner_model, "save_model"):
        model_path = output_dir / f"{model_name}.json"
        inner_model.save_model(model_path)
        return model_path
    
    raise TypeError(
        f"Unsupported model type: {type(model).__name__}. "
        "Expected a LightGBM or XGBoost style model."
    )

def train_tabular_model(
    model: Any,
    train_data: BuiltDataset,
    valid_data: BuiltDataset,
    output_dir: str | Path
) -> dict[str, float | int | str]:
    """
    Train a tabular model on BuiltDataset splits.

    This function is designed for tree-based tabular models such as
    LightGBMReturnRegressor and XGBoostReturnRegressor.

    Parameters
    ----------
    model : Any
        A model instance with fit(...) and predict(...) methods.
    train_data : BuiltDataset
        Training split with 2D X and 1D y.
    valid_data : BuiltDataset
        Validation split with 2D X and 1D y.
    output_dir : str or Path
        Directory used to save the trained model.

    Returns
    -------
    dict
        Training summary containing RMSE metrics and saved model path.
    """
    _validate_tabular_dataset(train_data, "train_data")
    _validate_tabular_dataset(valid_data, "valid_data")

    if not hasattr(model, "fit"):
        raise ValueError(
            f"Model of type {type(model).__name__} does not have a fit method."
        )
    
    if not hasattr(model, "predict"):
        raise ValueError(
            f"Model of type {type(model).__name__} does not have a predict method."
        )
    
    model.fit(
        train_data.X,
        train_data.y,
        valid_data.X,
        valid_data.y
    )

    train_pred = model.predict(train_data.X)
    valid_pred = model.predict(valid_data.X)

    train_rmse = _rmse(train_data.y, train_pred)
    valid_rmse = _rmse(valid_data.y, valid_pred)

    # Validation diagnostics
    rmse_gap = valid_rmse - train_rmse
    rmse_ratio = float("inf") if train_rmse == 0.0 else (valid_rmse / train_rmse)

    model_path = _save_tabular_model(model, output_dir)

    inner_model = getattr(model, "model", model)
    best_n = None
    total_n = None

    # --- best_n ---
    if hasattr(inner_model, "best_iteration_"):
        best_n = int(inner_model.best_iteration_)  # LGBM, 1-indexed
    elif hasattr(inner_model, "best_iteration"):
        best_n = int(inner_model.best_iteration)    # XGBoost

    # --- total_n: read from evals_result_ (records every round, not pruned) ---
    evals = getattr(inner_model, "evals_result_", None)
    if evals is None and hasattr(inner_model, "evals_result"):
        try:
            evals = inner_model.evals_result()
        except Exception:
            evals = None
    evals = evals or {}
    for key in evals:
        for metric_name, values in evals[key].items():
            if values:
                total_n = len(values)
                break
        if total_n is not None:
            break

    return {
        "train_rmse": train_rmse,
        "valid_rmse": valid_rmse,
        "rmse_gap": rmse_gap,
        "rmse_ratio": rmse_ratio,
        "best_n": best_n if best_n is not None else None,
        "total_n": total_n if total_n is not None else None,
        "train_size": train_data.X.shape[0],
        "valid_size": valid_data.X.shape[0],
        "model_path": str(model_path)
    }

if __name__ == "__main__":
    from src.models.xgboost_model import XGBoostConfig, build_xgboost_model

    rng = np.random.default_rng(42)

    X_train = rng.normal(size=(200, 10)).astype(np.float32)
    y_train = rng.normal(size=200).astype(np.float32)

    X_valid = rng.normal(size=(50, 10)).astype(np.float32)
    y_valid = rng.normal(size=50).astype(np.float32)

    train_data = BuiltDataset(
        X=X_train,
        y=y_train,
        meta=pd.DataFrame({"sample_id": range(len(y_train))}),
    )

    valid_data = BuiltDataset(
        X=X_valid,
        y=y_valid,
        meta=pd.DataFrame({"sample_id": range(len(y_valid))}),
    )

    model = build_xgboost_model(
        XGBoostConfig(
            n_estimators=20,
            max_depth=3,
        )
    )

    result = train_tabular_model(
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        output_dir="dataset/output/smoke_test",
    )

    print(result)