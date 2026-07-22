"""
Utilities for generating model predictions from BuiltDataset objects.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset_builder import BuiltDataset


@dataclass
class PredictionConfig:
    """
    Configuration for generating predictions.
    """

    batch_size: int = 4096
    device: str = "auto"
    validate_finite: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
        
def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this system.")
    
    return torch.device(device)

def _validate_prediction_dataset(dataset: BuiltDataset, name: str = "dataset") -> None:
    if not isinstance(dataset, BuiltDataset):
        raise TypeError(
            f"{name} must be a BuiltDataset, got {type(dataset).__name__}"
        )

    if dataset.X.ndim not in {2, 3}:
        raise ValueError(
            f"{name}.X must be 2D or 3D, got shape {dataset.X.shape}"
        )

    if dataset.y.ndim != 1:
        raise ValueError(
            f"{name}.y must be 1D, got shape {dataset.y.shape}"
        )

    if dataset.X.shape[0] != dataset.y.shape[0]:
        raise ValueError(
            f"{name}.X and {name}.y length mismatch: "
            f"{dataset.X.shape[0]} vs {dataset.y.shape[0]}"
        )

    if len(dataset.meta) != dataset.X.shape[0]:
        raise ValueError(
            f"{name}.meta and {name}.X length mismatch: "
            f"{len(dataset.meta)} vs {dataset.X.shape[0]}"
        )

    if dataset.X.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    
def _to_1d_prediction(preds: Any, expected_len: int) -> np.ndarray:
    preds = np.asarray(preds).reshape(-1)

    if preds.shape[0] != expected_len:
        raise ValueError(f"Predictions length {preds.shape[0]} does not match expected length {expected_len}.")
    
    return preds.astype(np.float32, copy=False)

def _predict_with_tabular_model(model: Any, dataset: BuiltDataset) -> np.ndarray:
    if not hasattr(model, "predict"):
        raise ValueError("Model must have a predict method for tabular data.")
    
    preds = model.predict(dataset.X)
    return _to_1d_prediction(preds, expected_len=dataset.X.shape[0])

def _predict_with_torch_model(model: nn.Module, dataset: BuiltDataset, config: PredictionConfig) -> np.ndarray:
    device = _resolve_device(config.device)
    model = model.to(device)
    model.eval()

    X_tensor = torch.as_tensor(dataset.X, dtype=torch.float32)
    ind_id = getattr(dataset, "industry_id", None)
    if ind_id is not None:
        ind_tensor = torch.as_tensor(ind_id, dtype=torch.long)
        tensor_dataset = TensorDataset(X_tensor, ind_tensor)
    else:
        tensor_dataset = TensorDataset(X_tensor)
    data_loader = DataLoader(tensor_dataset, batch_size=config.batch_size, shuffle=False)

    preds_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in data_loader:
            batch_X = batch[0].to(device)
            kwargs = {}
            if len(batch) > 1 and getattr(model, "uses_industry_id", False):
                kwargs["industry_id"] = batch[1].to(device)
            batch_pred = model(batch_X, **kwargs)

            if batch_pred.ndim == 1:
                batch_pred = batch_pred.reshape(-1, 1)

            preds_list.append(batch_pred.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0).reshape(-1)
    return _to_1d_prediction(preds, expected_len=dataset.X.shape[0])

def generate_predictions(
    model: Any,
    dataset: BuiltDataset,
    model_name: str | None = None,
    config: PredictionConfig | None = None,
    required_meta_cols: tuple[str, ...] = ("date", "stock_id"),
) -> pd.DataFrame:
    """
    Generate aligned predictions from a trained model and a BuiltDataset.

    Parameters
    ----------
    model : Any
        Trained model. It can be a tabular model with predict(X), or a
        PyTorch nn.Module.
    dataset : BuiltDataset
        Dataset containing X, y, and aligned meta.
    model_name : str, optional
        Name stored in the output DataFrame. If None, the class name is used.
    config : PredictionConfig, optional
        Prediction configuration, mainly used for PyTorch batch inference.
    required_meta_cols : tuple[str, ...], optional
        Required metadata columns that must exist in the prediction output.

    Returns
    -------
    pd.DataFrame
        DataFrame containing meta columns, y_true, y_pred, and model_name.
    """
    if config is None:
        config = PredictionConfig()

    _validate_prediction_dataset(dataset, "dataset")

    resolved_model_name = model_name or model.__class__.__name__

    if isinstance(model, nn.Module):
        y_pred = _predict_with_torch_model(model=model, dataset=dataset, config=config)
    else:
        y_pred = _predict_with_tabular_model(model=model, dataset=dataset)

    if config.validate_finite and not np.isfinite(y_pred).all():
        raise ValueError("Predictions contain non-finite values.")
    
    pred_df = dataset.meta.reset_index(drop=True).copy()
    missing_cols = [col for col in required_meta_cols if col not in pred_df.columns]
    if missing_cols:
        raise ValueError(
            f"prediction meta is missing required columns: {missing_cols}"
        )

    pred_df["y_true"] = dataset.y.astype(np.float32, copy=False)
    pred_df["y_pred"] = y_pred
    pred_df["model_name"] = resolved_model_name

    return pred_df

def save_predictions(pred_df: pd.DataFrame, output_path: str | Path) -> Path:
    """
    Save prediction DataFrame to parquet or csv.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()

    if suffix == ".parquet":
        pred_df.to_parquet(output_path, index=False)
    elif suffix == ".csv":
        pred_df.to_csv(output_path, index=False)
    else:
        raise ValueError(
            f"Unsupported prediction file format: {suffix}. "
            "Expected .parquet or .csv."
        )

    return output_path

if __name__ == "__main__":
    class DummyTorchRegressor(nn.Module):
        def __init__(self, seq_len: int, n_features: int) -> None:
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear = nn.Linear(seq_len * n_features, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.flatten(x)
            return self.linear(x)

    rng = np.random.default_rng(42)

    n_samples = 20
    seq_len = 5
    n_features = 3

    X = rng.normal(size=(n_samples, seq_len, n_features)).astype(np.float32)
    y = rng.normal(size=n_samples).astype(np.float32)

    meta = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n_samples, freq="D"),
            "stock_id": [f"S{i:03d}" for i in range(n_samples)],
            "industry_code": ["IND_1"] * n_samples,
        }
    )

    dataset = BuiltDataset(X=X, y=y, meta=meta)

    model = DummyTorchRegressor(seq_len=seq_len, n_features=n_features)

    pred_df = generate_predictions(
        model=model,
        dataset=dataset,
        model_name="dummy_torch",
        config=PredictionConfig(batch_size=8, device="cpu"),
    )

    output_path = save_predictions(
        pred_df=pred_df,
        output_path="dataset/output/smoke_test/dummy_predictions.parquet",
    )

    print(pred_df.head())
    print("Prediction shape:", pred_df.shape)
    print("Saved predictions to:", output_path)