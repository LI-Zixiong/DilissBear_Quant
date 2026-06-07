"""
Training utilities for PyTorch time-series models.
"""

import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", "An input array is constant")
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset_builder import BuiltDataset

@dataclass
class TorchTrainConfig:
    """
    Configuration for PyTorch model training.
    """

    epochs: int = 20
    patience: int = 3
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: str = 'auto'
    shuffle_train: bool = True
    date_col: str = "time"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"Invalid epochs={self.epochs}. Expected a positive integer.")

        if self.patience < 0:
            raise ValueError(f"Invalid patience={self.patience}. Expected a non-negative integer.")
        
        if self.batch_size <= 0:
            raise ValueError(f"Invalid batch_size={self.batch_size}. Expected a positive integer.")
        
        if self.learning_rate <= 0:
            raise ValueError(f"Invalid learning_rate={self.learning_rate}. Expected a positive number.")
        
        if self.weight_decay < 0:
            raise ValueError(f"Invalid weight_decay={self.weight_decay}. Expected a non-negative number.")
        
        if self.device not in ('auto', 'cpu', 'cuda'):
            raise ValueError(f"Invalid device={self.device!r}. Expected 'auto', 'cpu', or 'cuda'.")
        if not isinstance(self.date_col, str) or self.date_col == "":
            raise ValueError("date_col must be a non-empty string")
    
def _resolve_device(device: str) -> torch.device:
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError("CUDA device specified but not available.")
    
    return torch.device(device)

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"but y_pred has shape {y_pred.shape}."
        )
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _compute_icir(y_true: np.ndarray, y_pred: np.ndarray,
                  meta: pd.DataFrame, date_col: str,
                  min_obs: int = 10) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    dates = meta[date_col].values

    df = pd.DataFrame({"date": dates, "y_true": y_true, "y_pred": y_pred})
    daily_ic = df.groupby("date").apply(
        lambda g: np.nan
        if len(g) < min_obs or g["y_true"].nunique() <= 1 or g["y_pred"].nunique() <= 1
        else g["y_true"].corr(g["y_pred"], method="spearman")
    ).dropna()

    daily_ic = daily_ic.astype(float)
    if len(daily_ic) < 2:
        return {"icir": np.nan, "mean_ic": float(daily_ic.mean()) if len(daily_ic) > 0 else np.nan,
                "n_dates": len(daily_ic)}

    icir = float(daily_ic.mean() / daily_ic.std(ddof=1))
    return {"icir": icir, "mean_ic": float(daily_ic.mean()), "n_dates": len(daily_ic)}


def _compute_valid_cumret(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    date_col: str,
    top_n: int = 50,
) -> dict:
    """Compute cumulative return of daily top-N equal-weight portfolio on valid set.

    Signal at date t uses the next trading day's ret_daily (forward return t→t+1),
    matching the backtest engine's _get_next_return_date logic.
    """
    if "ret_daily" not in meta.columns or "stock_id" not in meta.columns:
        return {"cumret": np.nan, "sharpe": np.nan, "n_dates": 0}

    df = pd.DataFrame({
        "time": pd.to_datetime(meta[date_col].values),
        "stock_id": meta["stock_id"].astype(str).values,
        "y_pred": y_pred,
        "ret_daily": meta["ret_daily"].values,
    })
    df = df.dropna(subset=["y_pred", "ret_daily"])

    # Forward-return lookup: (date, stock_id) -> ret_daily at that date
    # ret_daily[t+1] = return from t to t+1 → this is what we earn on signal at t
    fwd_ret = df.set_index(["time", "stock_id"])["ret_daily"]

    # Map each signal date to the next available trading date
    all_dates = np.sort(df["time"].unique())
    next_date = {pd.Timestamp(all_dates[i]): pd.Timestamp(all_dates[i + 1])
                 for i in range(len(all_dates) - 1)}

    daily_rets = []
    for signal_date, g in df.groupby("time", sort=True):
        rd = next_date.get(pd.Timestamp(signal_date))
        if rd is None:
            continue
        top = g.nlargest(top_n, "y_pred")
        day_rets = []
        for sid in top["stock_id"]:
            try:
                r = fwd_ret.loc[(rd, sid)]
                if np.isfinite(r):
                    day_rets.append(r)
            except KeyError:
                continue
        if day_rets:
            daily_rets.append(np.mean(day_rets))

    if not daily_rets:
        return {"cumret": np.nan, "sharpe": np.nan, "n_dates": 0}

    rets = pd.Series(daily_rets, dtype=float)
    cumret = float((1.0 + rets).prod() - 1.0)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    return {"cumret": cumret, "sharpe": sharpe, "n_dates": len(rets)}


def _validate_torch_dataset(dataset: BuiltDataset, name: str) -> None:
    if not isinstance(dataset, BuiltDataset):
        raise ValueError(
            f"Invalid {name} dataset. Expected a BuiltDataset instance, "
            f"but got {type(dataset).__name__}."
        )
    
    if dataset.X.ndim != 3:
        raise ValueError(
            f"Invalid {name} dataset features. Expected a 3D array, "
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
    
def _build_dataloader(
    dataset: BuiltDataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    X_tensor = torch.as_tensor(dataset.X, dtype=torch.float32)
    y_tensor = torch.as_tensor(dataset.y, dtype=torch.float32).reshape(-1, 1)

    tensor_dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(
        tensor_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle
    )

def _train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        pred = model(batch_X)

        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)

        loss = criterion(pred, batch_y)

        loss.backward()
        optimizer.step()

        batch_size = batch_X.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples if total_samples > 0 else 0.0

def _evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_X)

            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)

            loss = criterion(pred, batch_y)

            batch_size = batch_X.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    y_pred = np.concatenate(all_preds, axis=0).reshape(-1)
    y_true = np.concatenate(all_targets, axis=0).reshape(-1)

    loss_avg = total_loss / total_samples if total_samples > 0 else 0.0
    rmse = _rmse(y_true, y_pred)

    return loss_avg, rmse, y_true, y_pred

def train_torch_model(
    model: nn.Module,
    train_data: BuiltDataset,
    valid_data: BuiltDataset,
    output_dir: str | Path,
    config: TorchTrainConfig | None = None,
    ) -> dict[str, float | int | str]:
    """
    Train a PyTorch model on sequence BuiltDataset splits.

    This function is designed for PyTorch time-series models such as
    DLinear, TSMixer, PatchTST, and iTransformer.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to train.
    train_data : BuiltDataset
        Training split with 3D X and 1D y.
    valid_data : BuiltDataset
        Validation split with 3D X and 1D y.
    output_dir : str or Path
        Directory used to save the best model checkpoint.
    config : TorchTrainConfig, optional
        Training configuration.

    Returns
    -------
    dict
        Training summary containing loss, RMSE, best epoch, and checkpoint path.
    """
    if config is None:
        config = TorchTrainConfig()

    if not isinstance(model, nn.Module):
        raise ValueError(
            f"Invalid model. Expected a PyTorch nn.Module instance, "
            f"but got {type(model).__name__}."
        )

    _validate_torch_dataset(train_data, "train_data")
    _validate_torch_dataset(valid_data, "valid_data")

    device = _resolve_device(config.device)
    model = model.to(device)

    train_loader = _build_dataloader(
        dataset=train_data,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
    )

    valid_loader = _build_dataloader(
        dataset=valid_data,
        batch_size=config.batch_size,
        shuffle=False,
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = model.__class__.__name__
    checkpoint_path = output_dir / f"{model_name}.pt"

    MIN_DELTA = 1e-6

    best_valid_rmse = float("inf")
    best_rmse_epoch = -1
    best_icir = -np.inf
    best_icir_epoch = -1
    best_cumret = -np.inf
    best_cumret_epoch = -1
    cumret_checkpoint_path = output_dir / f"{model_name}_best_cumret.pt"

    epoch_records: list[dict] = []  # per-epoch cumret/sharpe for composite scoring
    composite_checkpoint_path = output_dir / f"{model_name}_best_composite.pt"
    epochs_without_improvement = 0

    final_train_loss = float("nan")
    final_valid_loss = float("nan")
    final_valid_rmse = float("nan")

    t_start = time.perf_counter()

    icir_checkpoint_path = output_dir / f"{model_name}_best_icir.pt"

    for epoch in range(1, config.epochs + 1):
        t_epoch = time.perf_counter()

        train_loss = _train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        valid_loss, valid_rmse, y_true, y_pred = _evaluate(
            model=model,
            data_loader=valid_loader,
            criterion=criterion,
            device=device
        )

        ic_result = _compute_icir(y_true, y_pred, valid_data.meta, config.date_col)
        icir = ic_result["icir"]
        mean_ic = ic_result["mean_ic"]
        n_ic_dates = ic_result["n_dates"]

        final_train_loss = train_loss
        final_valid_loss = valid_loss
        final_valid_rmse = valid_rmse

        if not np.isfinite(valid_rmse):
            epochs_without_improvement += 1
            continue

        if valid_rmse < best_valid_rmse - MIN_DELTA:
            best_valid_rmse = valid_rmse
            best_rmse_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        if np.isfinite(icir) and icir > best_icir:
            best_icir = icir
            best_icir_epoch = epoch
            torch.save(model.state_dict(), icir_checkpoint_path)

        cumret_result = _compute_valid_cumret(y_true, y_pred, valid_data.meta, config.date_col)
        cumret = cumret_result["cumret"]
        epoch_sharpe = cumret_result["sharpe"]
        if np.isfinite(cumret) and cumret > best_cumret:
            best_cumret = cumret
            best_cumret_epoch = epoch
            torch.save(model.state_dict(), cumret_checkpoint_path)

        epoch_records.append({
            "epoch": epoch,
            "cumret": cumret if np.isfinite(cumret) else -np.inf,
            "sharpe": epoch_sharpe if np.isfinite(epoch_sharpe) else -np.inf,
            "checkpoint": output_dir / f"{model_name}_epoch{epoch}.pt",
        })
        torch.save(model.state_dict(), epoch_records[-1]["checkpoint"])

        epoch_time = time.perf_counter() - t_epoch
        elapsed = time.perf_counter() - t_start
        print(
            f"[{model_name}  epoch {epoch:>3}/{config.epochs}] "
            f"train_loss={train_loss:.6f}  valid_rmse={valid_rmse:.6f}  "
            f"ICIR={icir:.4f}  mean_IC={mean_ic:.4f}  n_ic_dates={n_ic_dates}  "
            f"cumret={cumret:.4f}  "
            f"best_rmse={best_valid_rmse:.6f} @{best_rmse_epoch}  "
            f"best_ICIR={best_icir:.4f} @{best_icir_epoch}  "
            f"best_cumret={best_cumret:.4f} @{best_cumret_epoch}  "
            f"epoch={epoch_time:.1f}s  elapsed={elapsed:.1f}s"
        )

        if config.patience > 0 and epochs_without_improvement >= config.patience:
            print(
                f"[{model_name}] early stop: no improvement for "
                f"{config.patience} epochs (min_delta={MIN_DELTA})"
            )
            break

    # Composite score: 80% cumret rank + 20% sharpe rank (percentile)
    records_df = pd.DataFrame(epoch_records)
    best_composite_epoch = -1
    if len(records_df) >= 2:
        records_df["rank_cumret"] = records_df["cumret"].rank(pct=True)
        records_df["rank_sharpe"] = records_df["sharpe"].rank(pct=True)
        records_df["composite"] = 0.8 * records_df["rank_cumret"] + 0.2 * records_df["rank_sharpe"]
        best_idx = records_df["composite"].idxmax()
        best_composite_epoch = int(records_df.loc[best_idx, "epoch"])
        best_record = records_df.loc[best_idx]
        best_ckpt = best_record["checkpoint"]
        import shutil
        shutil.copy2(best_ckpt, composite_checkpoint_path)
        print(
            f"[{model_name}] best_composite: epoch={best_composite_epoch}  "
            f"cumret={best_record['cumret']:.4f}  sharpe={best_record['sharpe']:.4f}  "
            f"composite={best_record['composite']:.4f}"
        )

    # Use composite-best checkpoint for prediction (fallback: ICIR > RMSE)
    use_checkpoint = (composite_checkpoint_path if best_composite_epoch >= 1
                      else icir_checkpoint_path if best_icir_epoch >= 1
                      else checkpoint_path)
    use_epoch = (best_composite_epoch if best_composite_epoch >= 1
                 else best_icir_epoch if best_icir_epoch >= 1
                 else best_rmse_epoch)

    if use_epoch < 1 or not use_checkpoint.exists():
        raise RuntimeError(
            "Failed to produce a valid checkpoint. "
            f"final_train_loss={final_train_loss}, "
            f"final_valid_loss={final_valid_loss}, "
            f"final_valid_rmse={final_valid_rmse}, "
            f"best_valid_rmse={best_valid_rmse}, "
            f"best_rmse_epoch={best_rmse_epoch}."
        )

    model.load_state_dict(torch.load(use_checkpoint, map_location=device))

    return {
        "final_train_loss": final_train_loss,
        "final_valid_loss": final_valid_loss,
        "final_valid_rmse": final_valid_rmse,
        "best_valid_rmse": best_valid_rmse,
        "best_rmse_epoch": best_rmse_epoch,
        "best_icir": best_icir,
        "best_icir_epoch": best_icir_epoch,
        "best_cumret": best_cumret,
        "best_cumret_epoch": best_cumret_epoch,
        "best_composite_epoch": best_composite_epoch,
        "train_size": int(train_data.y.shape[0]),
        "valid_size": int(valid_data.y.shape[0]),
        "model_path": str(use_checkpoint),
    }

if __name__ == "__main__":
    import pandas as pd

    class DummyTimeSeriesRegressor(nn.Module):
        def __init__(self, lookback: int, n_features: int) -> None:
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear = nn.Linear(lookback * n_features, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.flatten(x)
            return self.linear(x)

    rng = np.random.default_rng(42)

    n_train = 200
    n_valid = 50
    lookback = 20
    n_features = 8

    X_train = rng.normal(size=(n_train, lookback, n_features)).astype(np.float32)
    y_train = rng.normal(size=n_train).astype(np.float32)

    X_valid = rng.normal(size=(n_valid, lookback, n_features)).astype(np.float32)
    y_valid = rng.normal(size=n_valid).astype(np.float32)

    train_data = BuiltDataset(
        X=X_train,
        y=y_train,
        meta=pd.DataFrame({"time": range(n_train)}),
    )

    valid_data = BuiltDataset(
        X=X_valid,
        y=y_valid,
        meta=pd.DataFrame({"time": range(n_valid)}),
    )

    model = DummyTimeSeriesRegressor(
        lookback=lookback,
        n_features=n_features,
    )

    config = TorchTrainConfig(
        epochs=3,
        batch_size=32,
        learning_rate=1e-3,
        device="cpu",
    )

    result = train_torch_model(
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        output_dir="dataset/output/smoke_test",
        config=config,
    )

    print(result)