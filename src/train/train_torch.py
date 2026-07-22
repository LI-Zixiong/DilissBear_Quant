"""
Training utilities for PyTorch time-series models.
"""

import os
import random
import shutil
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", "An input array is constant")
import torch

# Apply CPU optimizations early (single-process training, no DataLoader workers).
try:
    _CPU_COUNT = max(1, (os.cpu_count() or 1) - 2)
    torch.set_num_threads(_CPU_COUNT)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # already set by another module
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset_builder import BuiltDataset


def _set_torch_reproducible(seed: int) -> None:
    """Reset Python/NumPy/PyTorch RNGs and request deterministic kernels.

    This makes a torch model run independent of any RNG consumed by previous
    models in a larger experiment. `warn_only=True` avoids crashing if PyTorch
    hits an operation that has no deterministic implementation on the machine.
    """
    seed = int(seed)

    # For CUDA matmul determinism; best set before CUDA context is created.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older PyTorch has no warn_only argument.
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass


def _load_state_dict_safely(path: Path, device: torch.device) -> dict:
    """Load a checkpoint state_dict across PyTorch versions without warnings when possible."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # PyTorch versions before weights_only support.
        return torch.load(path, map_location=device)


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
    seed: int = 42
    date_col: str = "time"
    enable_compile: bool = False  # torch.compile — may fail on Windows without MSVC

    # Loss selection. Default keeps the original MSE training path unchanged.
    # Use loss_type="ranknet" to optimize same-day top-vs-rest pairwise ranking.
    loss_type: str = "mse"  # "mse" | "ranknet"

    # RankNet hyperparameters. Only used when loss_type="ranknet".
    rank_pos_quantile: float = 0.95
    rank_neg_quantile: float = 0.80
    rank_tail_neg_quantile: float = 0.20
    rank_tau: float = 0.20
    pairs_per_pos: int = 10
    dates_per_batch: int = 8
    rank_min_date_obs: int = 50
    rank_hard_neg_frac: float = 0.20
    rank_hard_neg_score_quantile: float = 0.90
    rank_hard_neg_y_quantile: float = 0.30    # toxic: only bottom N% by y qualify
    rank_hard_neg_mode: str = "score"         # "score" | "toxic"
    rank_hard_neg_warmup_epochs: int = 3
    rank_tail_neg_frac: float = 0.10
    rank_weight_mode: str = "return_diff_log_mad"  # "none" | "return_diff_log_mad" | "rank_gap"

    # Early stop monitor. "auto" means RMSE for MSE, cumret for RankNet.
    early_stop_metric: str = "auto"  # "auto" | "rmse" | "cumret" | "icir"

    # Extra valid diagnostics for RankNet runs (pair acc, reverse cumret, decile spread).
    rank_eval_diagnostics: bool = True

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

        if self.loss_type not in ("mse", "ranknet"):
            raise ValueError(
                f"Invalid loss_type={self.loss_type!r}. Expected 'mse' or 'ranknet'."
            )
        if not 0.0 < self.rank_tail_neg_quantile < self.rank_neg_quantile < self.rank_pos_quantile < 1.0:
            raise ValueError(
                "Expected 0 < rank_tail_neg_quantile < rank_neg_quantile "
                "< rank_pos_quantile < 1."
            )
        if self.rank_tau <= 0:
            raise ValueError("rank_tau must be positive.")
        if self.pairs_per_pos <= 0:
            raise ValueError("pairs_per_pos must be a positive integer.")
        if self.dates_per_batch <= 0:
            raise ValueError("dates_per_batch must be a positive integer.")
        if self.rank_min_date_obs <= 1:
            raise ValueError("rank_min_date_obs must be greater than 1.")
        if not 0.0 <= self.rank_hard_neg_frac <= 1.0:
            raise ValueError("rank_hard_neg_frac must be in [0, 1].")
        if not 0.0 <= self.rank_tail_neg_frac <= 1.0:
            raise ValueError("rank_tail_neg_frac must be in [0, 1].")
        if self.rank_hard_neg_frac + self.rank_tail_neg_frac > 1.0:
            raise ValueError("rank_hard_neg_frac + rank_tail_neg_frac must be <= 1.")
        if self.rank_weight_mode not in ("none", "return_diff_log_mad", "rank_gap"):
            raise ValueError(
                "rank_weight_mode must be one of: "
                "'none', 'return_diff_log_mad', 'rank_gap'."
            )
        if self.early_stop_metric not in ("auto", "rmse", "cumret", "icir"):
            raise ValueError(
                "early_stop_metric must be one of: 'auto', 'rmse', 'cumret', 'icir'."
            )
    
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

    Uses ret_daily (close-to-close). Signal at date t → buy close(t), sell close(t+1).
    ret_daily[t] = close(t)/close(t-1)-1, so the forward return for signal[t] is ret_daily[t+1].
    """
    fwd_col = "ret_daily"
    if "stock_id" not in meta.columns:
        return {"cumret": np.nan, "sharpe": np.nan, "n_dates": 0}
    if fwd_col not in meta.columns:
        fwd_col = None

    fwd_vals = y_true if fwd_col is None else meta[fwd_col].values
    fwd_label = "ret_daily"

    df = pd.DataFrame({
        "time": pd.to_datetime(meta[date_col].values),
        "stock_id": meta["stock_id"].astype(str).values,
        "y_pred": y_pred,
        fwd_label: fwd_vals,
    })
    df = df.dropna(subset=["y_pred", fwd_label])

    # Build next-date mapping: signal[t] → earn ret_daily[t+1] = close(t+1)/close(t)-1
    unique_dates = pd.DatetimeIndex(df["time"].drop_duplicates()).sort_values()
    next_date_map = {unique_dates[i]: unique_dates[i + 1] for i in range(len(unique_dates) - 1)}

    # Index by (time, stock_id) — ret_daily at each date is the return ending at that date
    fwd_ret = df.set_index(["time", "stock_id"])[fwd_label]

    daily_rets = []
    for signal_date, g in df.groupby("time", sort=True):
        top = g.nlargest(top_n, "y_pred")
        return_date = next_date_map.get(pd.Timestamp(signal_date))
        if return_date is None:
            continue
        day_rets = []
        for sid in top["stock_id"]:
            try:
                r = fwd_ret.loc[(return_date, sid)]
                if isinstance(r, pd.Series):
                    r = r.iloc[0] if len(r) else np.nan
                r = float(r)
                if np.isfinite(r):
                    day_rets.append(r)
            except (KeyError, TypeError, ValueError):
                continue
        if day_rets:
            daily_rets.append(float(np.mean(day_rets)))

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
    seed: int = 42,
) -> DataLoader:
    X_tensor = torch.as_tensor(dataset.X, dtype=torch.float32)
    y_tensor = torch.as_tensor(dataset.y, dtype=torch.float32).reshape(-1, 1)

    ind_id = getattr(dataset, "industry_id", None)
    if ind_id is not None:
        ind_tensor = torch.as_tensor(ind_id, dtype=torch.long)
        tensor_dataset = TensorDataset(X_tensor, y_tensor, ind_tensor)
    else:
        tensor_dataset = TensorDataset(X_tensor, y_tensor)

    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        sampler = torch.utils.data.RandomSampler(tensor_dataset, generator=g)
        return DataLoader(tensor_dataset, batch_size=batch_size, sampler=sampler)

    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)


def _group_indices_by_date(
    dataset: BuiltDataset,
    date_col: str,
    min_date_obs: int,
) -> list[np.ndarray]:
    """Build same-day index groups for cross-sectional ranking losses."""
    if date_col not in dataset.meta.columns:
        raise ValueError(
            f"RankNet loss requires date_col={date_col!r} in dataset.meta. "
            f"Available columns: {list(dataset.meta.columns)}"
        )

    y = np.asarray(dataset.y).reshape(-1)
    finite_y = np.isfinite(y)

    meta = dataset.meta.reset_index(drop=True).copy()
    meta["__row_idx"] = np.arange(len(meta), dtype=np.int64)
    meta = meta.loc[finite_y].copy()

    date_groups: list[np.ndarray] = []
    for _, g in meta.groupby(date_col, sort=True):
        idx = g["__row_idx"].to_numpy(dtype=np.int64)
        if idx.shape[0] >= min_date_obs:
            date_groups.append(idx)

    if not date_groups:
        raise ValueError(
            "RankNet loss could not find any date group with enough observations. "
            f"min_date_obs={min_date_obs}."
        )

    return date_groups


def _iter_date_batches(
    date_groups: list[np.ndarray],
    dates_per_batch: int,
    shuffle: bool,
    seed: int,
    epoch: int,
) -> list[list[np.ndarray]]:
    """Return date-group batches; shuffled at the date level, never across stocks."""
    order = np.arange(len(date_groups), dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed + epoch * 9973)
        rng.shuffle(order)

    batches: list[list[np.ndarray]] = []
    for start in range(0, len(order), dates_per_batch):
        selected = order[start:start + dates_per_batch]
        batches.append([date_groups[int(i)] for i in selected])
    return batches


def _rank_pct_numpy(y: np.ndarray) -> np.ndarray:
    """Fast rank percentile with average-free deterministic tie handling."""
    order = np.argsort(y, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    if len(y) <= 1:
        ranks.fill(1.0)
        return ranks
    ranks[order] = (np.arange(len(y), dtype=np.float32) + 1.0) / float(len(y))
    return ranks


def _sample_ranknet_pairs_for_date(
    y_np: np.ndarray,
    config: TorchTrainConfig,
    rng: np.random.Generator,
    pred_np: np.ndarray | None = None,
    epoch: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Sample same-day top-vs-rest pairs.

    When pred_np is provided, hard negatives are mined from high-score
    non-positive stocks (score-based), replacing the y-based hard_neg_pool.
    """
    y_np = np.asarray(y_np, dtype=np.float32).reshape(-1)
    # Caller guarantees no NaN. Return only local indices (no remapping).
    if not np.isfinite(y_np).all():
        # Defensive: drop NaN rows but keep indices local.
        finite = np.isfinite(y_np)
        y_work = y_np[finite]
        p_work = pred_np[finite] if pred_np is not None else None
    else:
        finite = None
        y_work = y_np
        p_work = pred_np

    if len(y_work) < config.rank_min_date_obs:
        return None

    pos_cut = float(np.nanquantile(y_work, config.rank_pos_quantile))
    neg_cut = float(np.nanquantile(y_work, config.rank_neg_quantile))
    tail_cut = float(np.nanquantile(y_work, config.rank_tail_neg_quantile))

    pos_pool = np.flatnonzero(y_work >= pos_cut)
    main_neg_pool = np.flatnonzero(y_work <= neg_cut)
    tail_neg_pool = np.flatnonzero(y_work <= tail_cut)

    if len(pos_pool) == 0 or len(main_neg_pool) == 0:
        return None

    # Hard negatives: ramp from 0 after warmup to avoid corrupting early learning.
    if epoch <= config.rank_hard_neg_warmup_epochs:
        effective_hard_frac = 0.0
    elif epoch == config.rank_hard_neg_warmup_epochs + 1:
        effective_hard_frac = 0.5 * config.rank_hard_neg_frac
    else:
        effective_hard_frac = config.rank_hard_neg_frac

    use_hard = (p_work is not None and effective_hard_frac > 0)
    if use_hard:
        non_pos_mask = np.ones(len(y_work), dtype=bool)
        non_pos_mask[pos_pool] = False
        if non_pos_mask.sum() > 0:
            score_cut = float(np.quantile(p_work[non_pos_mask], config.rank_hard_neg_score_quantile))
            if config.rank_hard_neg_mode == "toxic" and config.rank_hard_neg_y_quantile > 0:
                y_cut = float(np.quantile(y_work[non_pos_mask], config.rank_hard_neg_y_quantile))
                hard_neg_pool = np.flatnonzero(non_pos_mask & (p_work >= score_cut) & (y_work <= y_cut))
            else:
                hard_neg_pool = np.flatnonzero(non_pos_mask & (p_work >= score_cut))
        else:
            hard_neg_pool = np.empty(0, dtype=np.int64)
    else:
        hard_neg_pool = np.empty(0, dtype=np.int64)

    pairs_per_pos = int(config.pairs_per_pos)
    n_tail = int(round(pairs_per_pos * config.rank_tail_neg_frac))
    if use_hard:
        n_hard = int(round(pairs_per_pos * effective_hard_frac))
    else:
        n_hard = 0
    n_main = max(0, pairs_per_pos - n_hard - n_tail)

    pos_indices: list[int] = []
    neg_indices: list[int] = []

    def _choice(pool: np.ndarray, size: int) -> np.ndarray:
        if size <= 0:
            return np.empty(0, dtype=np.int64)
        if len(pool) == 0:
            pool = main_neg_pool
        return rng.choice(pool, size=size, replace=(len(pool) < size))

    for p in pos_pool:
        main_negs = _choice(main_neg_pool, n_main)
        hard_negs = _choice(hard_neg_pool, n_hard)
        tail_negs = _choice(tail_neg_pool, n_tail)
        negs = np.concatenate([main_negs, hard_negs, tail_negs])
        if len(negs) == 0:
            continue
        pos_indices.extend([int(p)] * len(negs))
        neg_indices.extend(int(n) for n in negs)

    if not pos_indices:
        return None

    pos_idx = np.asarray(pos_indices, dtype=np.int64)
    neg_idx = np.asarray(neg_indices, dtype=np.int64).reshape(-1)

    diff = y_work[pos_idx] - y_work[neg_idx]
    keep = np.isfinite(diff) & (diff > 0)
    if keep.sum() == 0:
        return None
    pos_idx = pos_idx[keep]
    neg_idx = neg_idx[keep]
    diff = diff[keep]

    if config.rank_weight_mode == "none":
        weights = np.ones_like(diff, dtype=np.float32)
    elif config.rank_weight_mode == "rank_gap":
        rank_pct = _rank_pct_numpy(y_work)
        gap = rank_pct[pos_idx] - rank_pct[neg_idx]
        weights = np.clip(1.0 + 2.0 * gap, 1.0, 3.0).astype(np.float32)
    else:
        med = float(np.nanmedian(y_work))
        mad = float(np.nanmedian(np.abs(y_work - med)))
        scale = mad if mad > 1e-12 else float(np.nanstd(y_work))
        scale = scale if scale > 1e-12 else 1.0
        raw = np.maximum(diff / scale, 0.0)
        weights = np.clip(np.log1p(raw), 0.1, 3.0).astype(np.float32)

    if finite is not None:
        pos_idx = np.flatnonzero(finite)[pos_idx]
        neg_idx = np.flatnonzero(finite)[neg_idx]

    return pos_idx, neg_idx, weights


def _weighted_same_day_ranknet_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    config: TorchTrainConfig,
    rng: np.random.Generator,
    epoch: int = 0,
) -> tuple[torch.Tensor, float, int] | None:
    """Weighted RankNet loss for one trading date."""
    pred = pred.reshape(-1)
    y = y.reshape(-1)

    valid = torch.isfinite(pred) & torch.isfinite(y)
    if int(valid.sum().item()) < config.rank_min_date_obs:
        return None

    pred_valid = pred[valid]
    y_valid = y[valid]

    y_np = y_valid.detach().cpu().numpy().astype(np.float32)
    p_np = pred_valid.detach().cpu().numpy().astype(np.float32)

    sampled = _sample_ranknet_pairs_for_date(y_np, config, rng, pred_np=p_np, epoch=epoch)
    if sampled is None:
        return None
    pos_idx_np, neg_idx_np, weights_np = sampled

    pos_idx = torch.as_tensor(pos_idx_np, dtype=torch.long, device=pred_valid.device)
    neg_idx = torch.as_tensor(neg_idx_np, dtype=torch.long, device=pred_valid.device)
    weights = torch.as_tensor(weights_np, dtype=pred_valid.dtype, device=pred_valid.device)

    score_diff = (pred_valid[pos_idx] - pred_valid[neg_idx]) / float(config.rank_tau)
    pair_loss = F.softplus(-score_diff) * weights
    loss = pair_loss.mean()

    with torch.no_grad():
        pair_acc = float((score_diff > 0).float().mean().item())

    return loss, pair_acc, int(pos_idx.numel())


def _train_one_epoch_ranknet(
    model: nn.Module,
    dataset: BuiltDataset,
    date_groups: list[np.ndarray],
    optimizer: torch.optim.Optimizer,
    config: TorchTrainConfig,
    device: torch.device,
    epoch: int,
) -> dict[str, float | int]:
    """Train one epoch with same-day top-vs-rest weighted RankNet loss."""
    model.train()

    batches = _iter_date_batches(
        date_groups=date_groups,
        dates_per_batch=config.dates_per_batch,
        shuffle=config.shuffle_train,
        seed=config.seed,
        epoch=epoch,
    )
    rng = np.random.default_rng(config.seed + epoch * 104729)

    total_loss = 0.0
    total_dates = 0
    total_pair_acc = 0.0
    total_pairs = 0
    skipped_dates = 0

    for date_batch in batches:
        optimizer.zero_grad()
        date_losses: list[torch.Tensor] = []
        batch_pair_acc = 0.0
        batch_pairs = 0

        for idx in date_batch:
            batch_X = torch.as_tensor(dataset.X[idx], dtype=torch.float32, device=device)
            batch_y = torch.as_tensor(dataset.y[idx], dtype=torch.float32, device=device)

            kwargs = {}
            ind_arr = getattr(dataset, "industry_id", None)
            if ind_arr is not None and getattr(model, "uses_industry_id", False):
                kwargs["industry_id"] = torch.as_tensor(ind_arr[idx], dtype=torch.long, device=device)
            pred = model(batch_X, **kwargs)
            if pred.ndim > 1:
                pred = pred.reshape(-1)

            result = _weighted_same_day_ranknet_loss(
                pred=pred,
                y=batch_y,
                config=config,
                rng=rng,
                epoch=epoch,
            )
            if result is None:
                skipped_dates += 1
                continue

            date_loss, pair_acc, n_pairs = result
            date_losses.append(date_loss)
            batch_pair_acc += pair_acc * n_pairs
            batch_pairs += n_pairs

        if not date_losses:
            continue

        loss = torch.stack(date_losses).mean()
        loss.backward()
        optimizer.step()

        n_dates = len(date_losses)
        total_loss += float(loss.item()) * n_dates
        total_dates += n_dates
        total_pair_acc += batch_pair_acc
        total_pairs += batch_pairs

    return {
        "loss": total_loss / total_dates if total_dates > 0 else float("nan"),
        "pair_acc": total_pair_acc / total_pairs if total_pairs > 0 else float("nan"),
        "n_pairs": int(total_pairs),
        "n_dates": int(total_dates),
        "skipped_dates": int(skipped_dates),
    }


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

    for batch in dataloader:
        batch_X = batch[0].to(device)
        batch_y = batch[1].to(device)
        batch_ind = batch[2].to(device) if len(batch) > 2 else None

        optimizer.zero_grad()

        kwargs = {}
        if batch_ind is not None and getattr(model, "uses_industry_id", False):
            kwargs["industry_id"] = batch_ind
        pred = model(batch_X, **kwargs)

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
        for batch in data_loader:
            batch_X = batch[0].to(device)
            batch_y = batch[1].to(device)
            batch_ind = batch[2].to(device) if len(batch) > 2 else None

            kwargs = {}
            if batch_ind is not None and getattr(model, "uses_industry_id", False):
                kwargs["industry_id"] = batch_ind
            pred = model(batch_X, **kwargs)

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


def _maybe_compile(model: nn.Module) -> nn.Module:
    """Apply torch.compile with graceful fallback (Windows may lack MSVC)."""
    if not hasattr(torch, "compile"):
        print("[torch.compile] not available — using eager.")
        return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        print("[torch.compile] enabled (reduce-overhead).")
        return compiled
    except Exception as e:
        print(f"[torch.compile] failed: {e} — using eager.")
        return model


def _compute_ranknet_pair_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    date_col: str,
    config: TorchTrainConfig,
    seed: int = 12345,
) -> dict:
    """Evaluate same-day RankNet pair accuracy on a full split after prediction.

    This reuses the same positive/negative sampling rule as training, but it runs
    on already-computed predictions. It is a diagnostic only; it does not affect
    gradients or checkpoint selection.
    """
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    dates = meta[date_col].values

    df = pd.DataFrame({"date": dates, "y_true": y_true, "y_pred": y_pred})
    df = df.dropna(subset=["y_true", "y_pred"])
    if df.empty:
        return {
            "pair_acc": np.nan, "pair_acc_rev": np.nan,
            "pair_margin": np.nan, "weighted_pair_acc": np.nan,
            "n_pairs": 0, "n_dates": 0,
        }

    rng = np.random.default_rng(seed)
    correct = 0; correct_rev = 0; total = 0
    weighted_correct = 0.0; total_weight = 0.0
    margins: list[float] = []
    n_dates = 0

    for _, g in df.groupby("date", sort=True):
        y_np = g["y_true"].to_numpy(dtype=np.float32)
        p_np = g["y_pred"].to_numpy(dtype=np.float32)
        sampled = _sample_ranknet_pairs_for_date(y_np, config, rng)
        if sampled is None:
            continue
        pos_idx, neg_idx, weights = sampled
        if len(pos_idx) == 0:
            continue

        diff = p_np[pos_idx] - p_np[neg_idx]
        ok = diff > 0; ok_rev = diff < 0
        correct += int(ok.sum()); correct_rev += int(ok_rev.sum())
        total += int(len(diff))
        weighted_correct += float((ok.astype(np.float32) * weights).sum())
        total_weight += float(weights.sum())
        margins.append(float(np.mean(diff)))
        n_dates += 1

    if total == 0:
        return {
            "pair_acc": np.nan, "pair_acc_rev": np.nan,
            "pair_margin": np.nan, "weighted_pair_acc": np.nan,
            "n_pairs": 0, "n_dates": 0,
        }

    return {
        "pair_acc": float(correct / total),
        "pair_acc_rev": float(correct_rev / total),
        "pair_margin": float(np.mean(margins)) if margins else np.nan,
        "weighted_pair_acc": float(weighted_correct / total_weight) if total_weight > 0 else np.nan,
        "n_pairs": int(total),
        "n_dates": int(n_dates),
    }


def _compute_score_decile_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    date_col: str,
    min_obs: int = 50,
) -> dict:
    """Mean realized return of top/bottom score deciles by day.

    If score direction is healthy, top-score decile return should be above
    bottom-score decile return. If the spread is negative while train pair_acc is
    high, the learned signal is likely reversed or regime-specific.
    """
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    dates = meta[date_col].values
    df = pd.DataFrame({"date": dates, "y_true": y_true, "y_pred": y_pred})
    df = df.dropna(subset=["y_true", "y_pred"])

    top_rets: list[float] = []; bottom_rets: list[float] = []; spreads: list[float] = []

    for _, g in df.groupby("date", sort=True):
        if len(g) < min_obs or g["y_pred"].nunique() <= 1:
            continue
        top_pct = 1.0 - g["y_pred"].rank(method="first", pct=True)
        top = g.loc[top_pct <= 0.10, "y_true"]      # highest predicted scores
        bottom = g.loc[top_pct >= 0.90, "y_true"]    # lowest predicted scores
        if len(top) == 0 or len(bottom) == 0:
            continue
        top_ret = float(top.mean()); bottom_ret = float(bottom.mean())
        top_rets.append(top_ret); bottom_rets.append(bottom_ret)
        spreads.append(top_ret - bottom_ret)

    if not spreads:
        return {
            "top_decile_ret": np.nan, "bottom_decile_ret": np.nan,
            "decile_spread": np.nan, "n_dates": 0,
        }

    return {
        "top_decile_ret": float(np.mean(top_rets)),
        "bottom_decile_ret": float(np.mean(bottom_rets)),
        "decile_spread": float(np.mean(spreads)),
        "n_dates": int(len(spreads)),
    }


def _compute_pred_bucket_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    date_col: str,
    min_obs: int = 50,
) -> dict:
    """Predicted score bucket table — shows where the toxic tail lives."""
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    dates = meta[date_col].values
    df = pd.DataFrame({"date": dates, "y_true": y_true, "y_pred": y_pred})
    df = df.dropna(subset=["y_true", "y_pred"])

    bins = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    rows: list[dict] = []

    for lo, hi in bins:
        mean_rets = []
        hit_rates = []
        bad_rates = []
        bad_mean = []
        medians = []
        for _, g in df.groupby("date", sort=True):
            if len(g) < min_obs or g["y_pred"].nunique() <= 1:
                continue
            top_pct = (1.0 - g["y_pred"].rank(method="first", pct=True)) * 100
            bucket = g[(top_pct >= lo) & (top_pct < hi)]
            if len(bucket) < 5:
                continue
            mean_rets.append(float(bucket["y_true"].mean()))
            top5_cut = g["y_true"].quantile(0.95)
            bot20_cut = g["y_true"].quantile(0.20)
            hit_rates.append(float((bucket["y_true"] >= top5_cut).mean()))
            bad_mask = bucket["y_true"] <= bot20_cut
            bad_rates.append(float(bad_mask.mean()))
            bad_mean.append(float(bucket.loc[bad_mask, "y_true"].mean()) if bad_mask.any() else 0.0)
            medians.append(float(bucket["y_true"].median()))
        rows.append({
            "bucket": f"{lo}-{hi}%",
            "mean_ret": float(np.mean(mean_rets)) if mean_rets else np.nan,
            "hit_rate": float(np.mean(hit_rates)) if hit_rates else np.nan,
            "bad_rate": float(np.mean(bad_rates)) if bad_rates else np.nan,
            "bad_avg": float(np.mean(bad_mean)) if bad_mean else np.nan,
            "median": float(np.mean(medians)) if medians else np.nan,
        })
    return {"bucket_table": rows}


def _compute_hard_negative_pair_acc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    date_col: str,
    config: TorchTrainConfig,
    seed: int = 12345,
) -> dict:
    """Split pair accuracy: random negatives vs hard (high-score) negatives."""
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    dates = meta[date_col].values
    df = pd.DataFrame({"date": dates, "y_true": y_true, "y_pred": y_pred})
    df = df.dropna(subset=["y_true", "y_pred"])

    rng = np.random.default_rng(seed)
    random_correct = 0; random_total = 0
    hard_correct = 0; hard_total = 0

    for _, g in df.groupby("date", sort=True):
        y_np = g["y_true"].to_numpy(dtype=np.float32)
        p_np = g["y_pred"].to_numpy(dtype=np.float32)
        if len(y_np) < config.rank_min_date_obs:
            continue

        pos_cut = float(np.nanquantile(y_np, config.rank_pos_quantile))
        pos_pool = np.flatnonzero(y_np >= pos_cut)
        neg_pool = np.flatnonzero(y_np < pos_cut)
        if len(pos_pool) == 0 or len(neg_pool) < 2:
            continue

        # Build hard negative pool: top-50% of pred scores among non-positive
        neg_preds = p_np[neg_pool]
        hard_cut = float(np.quantile(neg_preds, 0.5))
        hard_neg_pool = neg_pool[neg_preds >= hard_cut]

        n_pos = min(20, len(pos_pool))
        n_neg = min(10, len(neg_pool))
        pos_sample = rng.choice(pos_pool, size=n_pos, replace=False)

        # Random negatives
        rand_neg = rng.choice(neg_pool, size=n_neg, replace=False)
        for p in pos_sample:
            diff_r = p_np[p] - p_np[rand_neg]
            random_correct += int((diff_r > 0).sum())
            random_total += len(rand_neg)

        # Hard negatives
        if len(hard_neg_pool) >= 5:
            hard_neg = rng.choice(hard_neg_pool, size=min(n_neg, len(hard_neg_pool)), replace=False)
            for p in pos_sample:
                diff_h = p_np[p] - p_np[hard_neg]
                hard_correct += int((diff_h > 0).sum())
                hard_total += len(hard_neg)

    return {
        "random_pair_acc": float(random_correct / random_total) if random_total > 0 else np.nan,
        "hard_pair_acc": float(hard_correct / hard_total) if hard_total > 0 else np.nan,
    }


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

    # Re-seed inside the training function too, so the training loop is
    # order-invariant even when called after other models in a full experiment.
    _set_torch_reproducible(config.seed)

    if not isinstance(model, nn.Module):
        raise ValueError(
            f"Invalid model. Expected a PyTorch nn.Module instance, "
            f"but got {type(model).__name__}."
        )

    _validate_torch_dataset(train_data, "train_data")
    _validate_torch_dataset(valid_data, "valid_data")

    device = _resolve_device(config.device)
    model = model.to(device)

    # torch.compile: fuse small ops (depthwise conv + gelu + gate matmul) for
    # reduced kernel launch overhead on CPU. Falls back to eager on failure.
    if config.enable_compile:
        model = _maybe_compile(model)

    train_loader: DataLoader | None = None
    train_date_groups: list[np.ndarray] | None = None

    if config.loss_type == "mse":
        train_loader = _build_dataloader(
            dataset=train_data,
            batch_size=config.batch_size,
            shuffle=config.shuffle_train,
            seed=config.seed,
        )
    else:
        train_date_groups = _group_indices_by_date(
            dataset=train_data,
            date_col=config.date_col,
            min_date_obs=config.rank_min_date_obs,
        )
        print(
            f"[train_torch] loss_type=ranknet  "
            f"dates={len(train_date_groups)}  dates_per_batch={config.dates_per_batch}  "
            f"pos_q={config.rank_pos_quantile:.2f}  neg_q={config.rank_neg_quantile:.2f}  "
            f"hard_frac={config.rank_hard_neg_frac:.2f}  "
            f"hard_score_q={config.rank_hard_neg_score_quantile:.2f}  "
            f"tail_frac={config.rank_tail_neg_frac:.2f}  "
            f"tau={config.rank_tau:.4f}  weight_mode={config.rank_weight_mode}"
        )

    valid_loader = _build_dataloader(
        dataset=valid_data,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )

    # Keep MSE as the validation loss/RMSE diagnostic even when training uses RankNet.
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

    early_stop_metric = config.early_stop_metric
    if early_stop_metric == "auto":
        early_stop_metric = "rmse" if config.loss_type == "mse" else "cumret"
    early_stop_best = float("inf") if early_stop_metric == "rmse" else -np.inf

    final_train_loss = float("nan")
    final_train_pair_acc = float("nan")
    final_train_n_pairs = 0
    final_train_n_dates = 0
    final_valid_loss = float("nan")
    final_valid_rmse = float("nan")

    t_start = time.perf_counter()

    icir_checkpoint_path = output_dir / f"{model_name}_best_icir.pt"

    for epoch in range(1, config.epochs + 1):
        t_epoch = time.perf_counter()

        if config.loss_type == "mse":
            if train_loader is None:
                raise RuntimeError("train_loader is unexpectedly None for MSE training.")
            train_loss = _train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device
            )
            train_pair_acc = float("nan")
            train_n_pairs = 0
            train_n_dates = 0
            train_skipped_dates = 0
        else:
            if train_date_groups is None:
                raise RuntimeError("train_date_groups is unexpectedly None for RankNet training.")
            rank_stats = _train_one_epoch_ranknet(
                model=model,
                dataset=train_data,
                date_groups=train_date_groups,
                optimizer=optimizer,
                config=config,
                device=device,
                epoch=epoch,
            )
            train_loss = float(rank_stats["loss"])
            train_pair_acc = float(rank_stats["pair_acc"])
            train_n_pairs = int(rank_stats["n_pairs"])
            train_n_dates = int(rank_stats["n_dates"])
            train_skipped_dates = int(rank_stats["skipped_dates"])

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

        cumret_result = _compute_valid_cumret(y_true, y_pred, valid_data.meta, config.date_col)
        cumret = cumret_result["cumret"]
        epoch_sharpe = cumret_result["sharpe"]

        # RankNet diagnostics: valid pair acc, reverse-score metrics, decile spread.
        valid_pair_acc = float("nan")
        valid_pair_acc_rev = float("nan")
        reverse_cumret = float("nan")
        decile_spread = float("nan")
        if config.loss_type == "ranknet" and config.rank_eval_diagnostics:
            pair_diag = _compute_ranknet_pair_diagnostics(
                y_true=y_true, y_pred=y_pred, meta=valid_data.meta,
                date_col=config.date_col, config=config,
                seed=config.seed + epoch * 4099,
            )
            valid_pair_acc = pair_diag["pair_acc"]
            valid_pair_acc_rev = pair_diag["pair_acc_rev"]

            rev_cumret_result = _compute_valid_cumret(
                y_true, -y_pred, valid_data.meta, config.date_col)
            reverse_cumret = rev_cumret_result["cumret"]

            decile_diag = _compute_score_decile_diagnostics(
                y_true=y_true, y_pred=y_pred, meta=valid_data.meta,
                date_col=config.date_col,
            )
            decile_spread = decile_diag["decile_spread"]

        final_train_loss = train_loss
        final_train_pair_acc = train_pair_acc
        final_train_n_pairs = train_n_pairs
        final_train_n_dates = train_n_dates
        final_valid_loss = valid_loss
        final_valid_rmse = valid_rmse

        if np.isfinite(valid_rmse) and valid_rmse < best_valid_rmse - MIN_DELTA:
            best_valid_rmse = valid_rmse
            best_rmse_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

        if np.isfinite(icir) and icir > best_icir:
            best_icir = icir
            best_icir_epoch = epoch
            torch.save(model.state_dict(), icir_checkpoint_path)

        if np.isfinite(cumret) and cumret > best_cumret:
            best_cumret = cumret
            best_cumret_epoch = epoch
            torch.save(model.state_dict(), cumret_checkpoint_path)

        if early_stop_metric == "rmse":
            early_stop_value = valid_rmse
            early_stop_improved = (
                np.isfinite(early_stop_value)
                and early_stop_value < early_stop_best - MIN_DELTA
            )
        elif early_stop_metric == "cumret":
            early_stop_value = cumret
            early_stop_improved = (
                np.isfinite(early_stop_value)
                and early_stop_value > early_stop_best + MIN_DELTA
            )
        else:  # early_stop_metric == "icir"
            early_stop_value = icir
            early_stop_improved = (
                np.isfinite(early_stop_value)
                and early_stop_value > early_stop_best + MIN_DELTA
            )

        if early_stop_improved:
            early_stop_best = float(early_stop_value)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_records.append({
            "epoch": epoch,
            "loss_type": config.loss_type,
            "train_loss": train_loss,
            "train_pair_acc": train_pair_acc,
            "train_n_pairs": train_n_pairs,
            "train_n_dates": train_n_dates,
            "train_skipped_dates": train_skipped_dates,
            "valid_rmse": valid_rmse,
            "icir": icir,
            "mean_ic": mean_ic,
            "cumret": cumret if np.isfinite(cumret) else -np.inf,
            "sharpe": epoch_sharpe if np.isfinite(epoch_sharpe) else -np.inf,
            "early_stop_metric": early_stop_metric,
            "early_stop_value": early_stop_value if np.isfinite(early_stop_value) else np.nan,
            "checkpoint": output_dir / f"{model_name}_epoch{epoch}.pt",
            "valid_pair_acc": valid_pair_acc,
            "reverse_cumret": reverse_cumret,
            "decile_spread": decile_spread,
        })
        torch.save(model.state_dict(), epoch_records[-1]["checkpoint"])

        epoch_time = time.perf_counter() - t_epoch
        elapsed = time.perf_counter() - t_start

        if config.loss_type == "ranknet":
            print(
                f"[{model_name}  epoch {epoch:>3}/{config.epochs}] "
                f"train_loss={train_loss:.6f}  pair_acc={train_pair_acc:.4f}  "
                f"ICIR={icir:.4f}  mean_IC={mean_ic:.4f}  "
                f"cumret={cumret:.4f}  sharpe={epoch_sharpe:.4f}  "
                f"best_ICIR={best_icir:.4f} @{best_icir_epoch}  "
                f"best_cumret={best_cumret:.4f} @{best_cumret_epoch}  "
                f"epoch={epoch_time:.1f}s  elapsed={elapsed:.1f}s"
            )
            if config.rank_eval_diagnostics and not np.isnan(valid_pair_acc):
                hard_diag = _compute_hard_negative_pair_acc(
                    y_true=y_true, y_pred=y_pred, meta=valid_data.meta,
                    date_col=config.date_col, config=config,
                    seed=config.seed + epoch * 8191,
                )
                bucket_diag = _compute_pred_bucket_diagnostics(
                    y_true=y_true, y_pred=y_pred, meta=valid_data.meta,
                    date_col=config.date_col,
                )
                print(
                    f"  [diag] valid_pair_acc={valid_pair_acc:.4f}"
                    f"  rev_cumret={reverse_cumret:.4f}"
                    f"  decile_spread={decile_spread:.6f}"
                )
                print(
                    f"  [hard] random_pair_acc={hard_diag['random_pair_acc']:.4f}"
                    f"  hard_pair_acc={hard_diag['hard_pair_acc']:.4f}"
                )
                print(f"  [buckets] " + " | ".join(
                    f"{r['bucket']}: ret={r['mean_ret']:+.5f} hit={r['hit_rate']:.4f} bad={r['bad_rate']:.3f}"
                    for r in bucket_diag["bucket_table"][:7]
                ))
        else:
            print(
                f"[{model_name}  epoch {epoch:>3}/{config.epochs}] "
                f"train_loss={train_loss:.6f}  "
                f"valid_rmse={valid_rmse:.6f}  "
                f"ICIR={icir:.4f}  mean_IC={mean_ic:.4f}  n_ic_dates={n_ic_dates}  "
                f"cumret={cumret:.4f}  "
                f"best_rmse={best_valid_rmse:.6f} @{best_rmse_epoch}  "
                f"best_ICIR={best_icir:.4f} @{best_icir_epoch}  "
                f"best_cumret={best_cumret:.4f} @{best_cumret_epoch}  "
                f"epoch={epoch_time:.1f}s  elapsed={elapsed:.1f}s"
            )

        if config.patience > 0 and epochs_without_improvement >= config.patience:
            print(
                f"[{model_name}] early stop: {early_stop_metric} did not improve for "
                f"{config.patience} epochs (min_delta={MIN_DELTA})"
            )
            break

    # Composite score: 80% cumret rank + 20% sharpe rank (percentile).
    # Keep an epoch-level CSV because it is the fastest way to verify
    # full-run vs single-run reproducibility from epoch 1 onward.
    records_df = pd.DataFrame(epoch_records)
    epoch_records_path = output_dir / f"{model_name}_epoch_records.csv"
    best_composite_epoch = -1

    if not records_df.empty:
        finite_score_df = records_df[
            np.isfinite(records_df["cumret"].to_numpy(dtype=float))
            & np.isfinite(records_df["sharpe"].to_numpy(dtype=float))
        ].copy()

        if len(finite_score_df) >= 1:
            finite_score_df["rank_cumret"] = finite_score_df["cumret"].rank(pct=True)
            finite_score_df["rank_sharpe"] = finite_score_df["sharpe"].rank(pct=True)
            finite_score_df["composite"] = (
                0.8 * finite_score_df["rank_cumret"]
                + 0.2 * finite_score_df["rank_sharpe"]
            )
            best_idx = finite_score_df["composite"].idxmax()
            best_composite_epoch = int(finite_score_df.loc[best_idx, "epoch"])
            best_record = finite_score_df.loc[best_idx]
            best_ckpt = Path(best_record["checkpoint"])
            shutil.copy2(best_ckpt, composite_checkpoint_path)
            print(
                f"[{model_name}] best_composite: epoch={best_composite_epoch}  "
                f"cumret={best_record['cumret']:.4f}  sharpe={best_record['sharpe']:.4f}  "
                f"composite={best_record['composite']:.4f}"
            )

            records_df = records_df.merge(
                finite_score_df[["epoch", "rank_cumret", "rank_sharpe", "composite"]],
                on="epoch",
                how="left",
            )

        records_to_save = records_df.copy()
        records_to_save["checkpoint"] = records_to_save["checkpoint"].astype(str)
        records_to_save.to_csv(epoch_records_path, index=False)

    # Use composite-best checkpoint for prediction (fallback: ICIR > RMSE).
    # The selected_* fields are deliberately returned for reports/debugging.
    if best_composite_epoch >= 1:
        use_checkpoint = composite_checkpoint_path
        use_epoch = best_composite_epoch
        selection_metric = "composite"
    elif best_icir_epoch >= 1:
        use_checkpoint = icir_checkpoint_path
        use_epoch = best_icir_epoch
        selection_metric = "icir"
    else:
        use_checkpoint = checkpoint_path
        use_epoch = best_rmse_epoch
        selection_metric = "rmse"

    if use_epoch < 1 or not use_checkpoint.exists():
        raise RuntimeError(
            "Failed to produce a valid checkpoint. "
            f"final_train_loss={final_train_loss}, "
            f"final_valid_loss={final_valid_loss}, "
            f"final_valid_rmse={final_valid_rmse}, "
            f"best_valid_rmse={best_valid_rmse}, "
            f"best_rmse_epoch={best_rmse_epoch}."
        )

    model.load_state_dict(_load_state_dict_safely(use_checkpoint, device))

    return {
        "loss_type": config.loss_type,
        "early_stop_metric": early_stop_metric,
        "final_train_loss": final_train_loss,
        "final_train_pair_acc": final_train_pair_acc,
        "final_train_n_pairs": final_train_n_pairs,
        "final_train_n_dates": final_train_n_dates,
        "final_valid_loss": final_valid_loss,
        "final_valid_rmse": final_valid_rmse,
        "best_valid_rmse": best_valid_rmse,
        "best_rmse_epoch": best_rmse_epoch,
        "best_icir": best_icir,
        "best_icir_epoch": best_icir_epoch,
        "best_cumret": best_cumret,
        "best_cumret_epoch": best_cumret_epoch,
        "best_composite_epoch": best_composite_epoch,
        "selected_epoch": use_epoch,
        "selection_metric": selection_metric,
        "epoch_records_path": str(epoch_records_path),
        "train_size": int(train_data.y.shape[0]),
        "valid_size": int(valid_data.y.shape[0]),
        "model_path": str(use_checkpoint),
        "valid_pair_acc": valid_pair_acc,
        "valid_pair_acc_rev": valid_pair_acc_rev,
        "reverse_cumret": reverse_cumret,
        "decile_spread": decile_spread,
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