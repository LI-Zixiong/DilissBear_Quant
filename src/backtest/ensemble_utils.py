"""
Ensemble evaluation for Strategy V1.

This script loads existing valid/test predictions from run_experiment and evaluates
single models, equal-weight ensembles, Rank-Ridge ensembles, and non-50/50 weight
checks. It does not retrain any model.

Key design choices:
    1. Valid is used for parameter/variant selection; test is report-only.
    2. Rank-Ridge is fit on daily ranks and applied on daily ranks as well.
    3. Smoothing is causal trailing prediction smoothing, never centered.
    4. Backtest return is always return_1d, even when model predictions target 5d_next_raw.
    5. Prediction IC/Rank IC metrics are reported for every scored variant.
    6. Best variants are selected by validation rank ICIR by default.
    7. 50/50 grid is omitted because it is already covered by equal-weight.

Outputs:
    reports/strategy_v1/ensemble_results.csv
    reports/strategy_v1/ensemble_summary.txt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from src.backtest.engine import TransactionCostConfig, run_backtest
from src.backtest.metrics import prediction_ic_summary
from src.backtest.portfolio import PortfolioConfig
from src.experiment.returns import align_predictions_to_returns


DEFAULT_MODELS = ("lightgbm", "dlinear")
DEFAULT_SMOOTH_WINDOWS = (1, 3, 5)
DEFAULT_GRID_WEIGHTS = ((0.75, 0.25), (0.25, 0.75))  # 50/50 is equal-weight.
DEFAULT_RIDGE_ALPHAS = (0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0)
DEFAULT_SELECTION_METRIC = "rank_ic_ir"

ALL_SECTIONS = (
    "single",
    "equal_weight_raw",
    "equal_weight_rank",
    "rank_ridge",
    "ridge_raw",
    "fixed_weight_rank",
)
BEST_SELECTION_SECTIONS = ("equal_weight_raw", "equal_weight_rank", "rank_ridge", "ridge_raw")
IC_METRIC_KEYS = (
    "ic_mean",
    "ic_std",
    "ic_ir",
    "rank_ic_mean",
    "rank_ic_std",
    "rank_ic_ir",
)


@dataclass(frozen=True)
class EnsembleConfig:
    exp_dir: Path = Path("dataset/output/experiment_003")
    returns_path: Path = Path("dataset/processed/unified_daily_panel.parquet")
    output_dir: Path = Path("reports/strategy_v1")
    models: tuple[str, ...] = DEFAULT_MODELS
    smooth_windows: tuple[int, ...] = DEFAULT_SMOOTH_WINDOWS
    grid_weights: tuple[tuple[float, ...], ...] = DEFAULT_GRID_WEIGHTS
    ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS
    top_n: int = 50
    allow_legacy_test_path: bool = True
    selection_metric: str = DEFAULT_SELECTION_METRIC

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("models must be non-empty")
        if not all(w >= 1 for w in self.smooth_windows):
            raise ValueError(f"smooth_windows must all be >= 1, got {list(self.smooth_windows)}")
        if self.top_n <= 0:
            raise ValueError(f"top_n must be > 0, got {self.top_n}")
        if self.selection_metric not in {"sharpe", "ic_ir", "rank_ic_ir"}:
            raise ValueError(
                "selection_metric must be one of {'sharpe', 'ic_ir', 'rank_ic_ir'}, "
                f"got {self.selection_metric!r}"
            )
        for i, weights in enumerate(self.grid_weights):
            if len(weights) != len(self.models):
                raise ValueError(
                    f"grid_weights[{i}] has {len(weights)} weight(s) but {len(self.models)} model(s). "
                    f"Weight tuple: {weights}"
                )


# ---------------------------------------------------------------------------
# Loading and normalization
# ---------------------------------------------------------------------------


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize time and stock_id keys for safe joins."""
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"]).dt.normalize()
    out["stock_id"] = out["stock_id"].astype(str).str.strip().str.zfill(6)
    return out


def prediction_path(exp_dir: Path, split: str, model: str, allow_legacy_test_path: bool) -> Path:
    """Resolve prediction file path.

    Valid split must be explicit. Test split may fall back to legacy
    predictions_{model}.parquet for backward compatibility.
    """
    split_path = exp_dir / f"predictions_{split}_{model}.parquet"
    if split_path.exists():
        return split_path

    legacy_path = exp_dir / f"predictions_{model}.parquet"
    if split == "test" and allow_legacy_test_path and legacy_path.exists():
        return legacy_path

    raise FileNotFoundError(
        f"Missing prediction file for split={split!r}, model={model!r}. "
        f"Expected {split_path}"
        + (f" or legacy {legacy_path}" if split == "test" and allow_legacy_test_path else "")
    )


def load_prediction_file(exp_dir: Path, split: str, model: str, allow_legacy_test_path: bool) -> pd.DataFrame:
    """Load and normalize one model's predictions."""
    path = prediction_path(exp_dir, split, model, allow_legacy_test_path)
    df = pd.read_parquet(path)
    required = {"time", "stock_id", "y_pred", "y_true"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = normalize_keys(df)
    return df[["time", "stock_id", "y_true", "y_pred"]].rename(columns={"y_pred": model})


def load_predictions(config: EnsembleConfig, split: str) -> pd.DataFrame:
    """Load predictions for all models and inner-join on (time, stock_id)."""
    parts: list[pd.DataFrame] = []
    for model in config.models:
        part = load_prediction_file(
            exp_dir=config.exp_dir,
            split=split,
            model=model,
            allow_legacy_test_path=config.allow_legacy_test_path,
        )
        parts.append(part)

    merged = parts[0]
    for part in parts[1:]:
        model_col = [c for c in part.columns if c not in {"time", "stock_id", "y_true"}][0]
        merged = merged.merge(
            part[["time", "stock_id", model_col]],
            on=["time", "stock_id"],
            how="inner",
        )

    # y_true is taken from the first model after inner join. It should be the same target.
    merged = merged.sort_values(["time", "stock_id"]).reset_index(drop=True)
    return merged


def load_returns(path: Path) -> pd.DataFrame:
    """Load realized daily returns used by the backtest."""
    panel = pd.read_parquet(path)
    required = {"time", "stock_id", "ret_daily"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    ret = panel[["time", "stock_id", "ret_daily"]].rename(columns={"ret_daily": "return_1d"}).copy()
    ret = normalize_keys(ret)
    ret["return_1d"] = pd.to_numeric(ret["return_1d"], errors="coerce")
    n_before = len(ret)
    ret = ret.dropna(subset=["return_1d"]).sort_values(["time", "stock_id"]).reset_index(drop=True)
    n_dropped = n_before - len(ret)
    if n_dropped:
        print(f"  [load_returns] Dropped {n_dropped:,} rows with non-numeric return_1d ({n_dropped / n_before:.2%})")
    return ret


# ---------------------------------------------------------------------------
# Prediction transformations
# ---------------------------------------------------------------------------


def smooth_predictions(df: pd.DataFrame, model_cols: Iterable[str], window: int) -> pd.DataFrame:
    """Apply causal per-stock trailing rolling mean to model predictions.

    window=1 is the raw prediction. The operation uses only current and past
    predictions for the same stock, so it does not introduce future leakage.
    """
    if window < 1:
        raise ValueError(f"smooth window must be >= 1, got {window}")

    out = df.sort_values(["stock_id", "time"]).reset_index(drop=True).copy()
    if window == 1:
        return out.sort_values(["time", "stock_id"]).reset_index(drop=True)

    for col in model_cols:
        out[col] = out.groupby("stock_id", sort=False)[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    return out.sort_values(["time", "stock_id"]).reset_index(drop=True)


def add_daily_model_ranks(df: pd.DataFrame, model_cols: Iterable[str]) -> pd.DataFrame:
    """Add daily cross-sectional percentile ranks for model prediction columns."""
    out = df.copy()
    for col in model_cols:
        out[f"{col}_r"] = out.groupby("time")[col].rank(pct=True)
    return out


def add_daily_target_rank(df: pd.DataFrame, target_col: str = "y_true") -> pd.DataFrame:
    """Add daily cross-sectional percentile rank for y_true."""
    out = df.copy()
    out["y_true_r"] = out.groupby("time")[target_col].rank(pct=True)
    return out


# ---------------------------------------------------------------------------
# Scoring methods
# ---------------------------------------------------------------------------


def add_equal_weight_score(
    df: pd.DataFrame,
    model_cols: Iterable[str],
    score_col: str,
    rank_space: bool,
) -> pd.DataFrame:
    """Add equal-weight score in raw prediction space or rank space."""
    out = df.copy()
    cols = list(model_cols)
    if rank_space:
        out = add_daily_model_ranks(out, cols)
        rank_cols = [f"{m}_r" for m in cols]
        out[score_col] = out[rank_cols].mean(axis=1)
    else:
        out[score_col] = out[cols].mean(axis=1)
    return out


def fit_rank_ridge(valid_df: pd.DataFrame, model_cols: Iterable[str], alphas: Iterable[float]) -> tuple[pd.Series, float, int]:
    """Fit RidgeCV on valid daily-ranked model predictions against target rank."""
    cols = list(model_cols)
    ranked = add_daily_target_rank(add_daily_model_ranks(valid_df, cols))
    rank_cols = [f"{m}_r" for m in cols]
    fit_cols = rank_cols + ["y_true_r"]
    fit_df = ranked.dropna(subset=fit_cols)
    if fit_df.empty:
        raise ValueError("No valid rows to fit Rank-Ridge after rank/dropna.")

    ridge = RidgeCV(alphas=list(alphas), fit_intercept=False)
    ridge.fit(fit_df[rank_cols].to_numpy(), fit_df["y_true_r"].to_numpy())

    weights = pd.Series(ridge.coef_, index=cols, dtype=float)
    denom = weights.abs().sum()
    if not np.isfinite(denom) or denom <= 0:
        # Defensive fallback. Should be rare, but avoids silently producing NaNs.
        weights = pd.Series(1.0 / len(cols), index=cols, dtype=float)
    else:
        weights = weights / denom
    return weights, float(ridge.alpha_), int(len(fit_df))


def add_rank_ridge_score(
    df: pd.DataFrame,
    model_cols: Iterable[str],
    weights: pd.Series,
    score_col: str,
) -> pd.DataFrame:
    """Apply Rank-Ridge weights to daily-ranked model predictions."""
    cols = list(model_cols)
    out = add_daily_model_ranks(df, cols)
    out[score_col] = 0.0
    for col in cols:
        out[score_col] += float(weights[col]) * out[f"{col}_r"]
    return out


def fit_raw_ridge(valid_df: pd.DataFrame, model_cols: Iterable[str], alphas: Iterable[float]) -> tuple[pd.Series, float, int]:
    """Fit RidgeCV on valid raw model predictions against raw y_true."""
    cols = list(model_cols)
    fit_cols = cols + ["y_true"]
    fit_df = valid_df.dropna(subset=fit_cols)
    if fit_df.empty:
        raise ValueError("No valid rows to fit Raw-Ridge after dropna.")

    ridge = RidgeCV(alphas=list(alphas), fit_intercept=True)
    ridge.fit(fit_df[cols].to_numpy(), fit_df["y_true"].to_numpy())

    weights = pd.Series(ridge.coef_, index=cols, dtype=float)
    denom = weights.abs().sum()
    if not np.isfinite(denom) or denom <= 0:
        weights = pd.Series(1.0 / len(cols), index=cols, dtype=float)
    else:
        weights = weights / denom
    return weights, float(ridge.alpha_), int(len(fit_df))


def add_raw_ridge_score(
    df: pd.DataFrame,
    model_cols: Iterable[str],
    weights: pd.Series,
    score_col: str,
) -> pd.DataFrame:
    """Apply Raw-Ridge weights to raw model predictions."""
    cols = list(model_cols)
    out = df.copy()
    out[score_col] = 0.0
    for col in cols:
        out[score_col] += float(weights[col]) * out[col]
    return out


def add_weighted_rank_score(
    df: pd.DataFrame,
    model_cols: Iterable[str],
    weights: tuple[float, ...],
    score_col: str,
) -> pd.DataFrame:
    """Apply fixed weights to daily-ranked model predictions."""
    cols = list(model_cols)
    if len(cols) != len(weights):
        raise ValueError(f"weights length {len(weights)} != number of models {len(cols)}")
    out = add_daily_model_ranks(df, cols)
    out[score_col] = 0.0
    for col, weight in zip(cols, weights):
        out[score_col] += float(weight) * out[f"{col}_r"]
    return out


# ---------------------------------------------------------------------------
# Backtest, IC metrics, and result formatting
# ---------------------------------------------------------------------------


def backtest_score(
    df: pd.DataFrame, returns: pd.DataFrame, score_col: str, top_n: int,
    portfolio_config: PortfolioConfig | None = None,
) -> dict:
    """Backtest score column using realized next-day return_1d."""
    pred = df[["time", "stock_id", score_col]].rename(columns={score_col: "y_pred"}).dropna()
    aligned = align_predictions_to_returns(
        pred_df=pred,
        returns_df=returns,
        date_col="time",
        stock_col="stock_id",
        pred_col="y_pred",
        return_col="return_1d",
    )
    if portfolio_config is None:
        portfolio_config = PortfolioConfig(strategy="top_n", top_n=top_n, pred_col="y_pred", stock_col="stock_id")
    return run_backtest(
        pred_df=aligned,
        returns_df=returns,
        portfolio_config=portfolio_config,
        return_col="return_1d",
        date_col="time",
        stock_col="stock_id",
        cost_config=TransactionCostConfig(),
    )["summary"]


def compute_prediction_ic(df: pd.DataFrame, score_col: str) -> dict[str, float]:
    """Compute IC and Rank IC summary for a scored prediction DataFrame."""
    needed = ["time", "y_true", score_col]
    scored = df.dropna(subset=needed)
    if scored.empty:
        return {key: np.nan for key in IC_METRIC_KEYS}

    summary = prediction_ic_summary(
        scored,
        date_col="time",
        y_true_col="y_true",
        y_pred_col=score_col,
    )
    return {key: float(summary.get(key, np.nan)) for key in IC_METRIC_KEYS}


def metric(summary: dict, key: str, default: float = np.nan) -> float:
    return float(summary.get(key, default))


def make_result_row(
    *,
    section: str,
    strategy: str,
    split: str,
    window: int,
    summary: dict,
    ic_summary: dict[str, float],
    weights: str = "",
    ridge_alpha: float | None = None,
    n_fit: int | None = None,
    selected_by_valid: bool = False,
) -> dict:
    row = {
        "section": section,
        "strategy": strategy,
        "split": split,
        "window": window,
        "weights": weights,
        "ridge_alpha": ridge_alpha,
        "n_fit": n_fit,
        "selected_by_valid": selected_by_valid,
        "sharpe": metric(summary, "sharpe_ratio"),
        "annret": metric(summary, "annualized_return"),
        "maxdd": metric(summary, "max_drawdown"),
        "nav": metric(summary, "final_nav"),
        "turnover": metric(summary, "mean_turnover"),
        "winrate": metric(summary, "hit_rate"),
    }
    row.update({key: metric(ic_summary, key) for key in IC_METRIC_KEYS})
    return row


def evaluate_split_pair(
    *,
    section: str,
    strategy: str,
    window: int,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    returns: pd.DataFrame,
    score_col: str,
    top_n: int,
    weights: str = "",
    ridge_alpha: float | None = None,
    n_fit: int | None = None,
) -> list[dict]:
    valid_summary = backtest_score(valid_df, returns, score_col, top_n)
    test_summary = backtest_score(test_df, returns, score_col, top_n)
    valid_ic = compute_prediction_ic(valid_df, score_col)
    test_ic = compute_prediction_ic(test_df, score_col)
    return [
        make_result_row(
            section=section,
            strategy=strategy,
            split="valid",
            window=window,
            summary=valid_summary,
            ic_summary=valid_ic,
            weights=weights,
            ridge_alpha=ridge_alpha,
            n_fit=n_fit,
        ),
        make_result_row(
            section=section,
            strategy=strategy,
            split="test",
            window=window,
            summary=test_summary,
            ic_summary=test_ic,
            weights=weights,
            ridge_alpha=ridge_alpha,
            n_fit=n_fit,
        ),
    ]


def weights_to_str(weights: pd.Series | tuple[float, ...], models: tuple[str, ...]) -> str:
    if isinstance(weights, pd.Series):
        return ";".join(f"{m}={weights[m]:.6f}" for m in models)
    return ";".join(f"{m}={w:.6f}" for m, w in zip(models, weights))


# ---------------------------------------------------------------------------
# Evaluation blocks
# ---------------------------------------------------------------------------


def evaluate_single_models(
    valid: pd.DataFrame,
    test: pd.DataFrame,
    returns: pd.DataFrame,
    config: EnsembleConfig,
) -> list[dict]:
    """Evaluate each single model under every smoothing window.

    This makes the single-model baseline comparable with smoothed ensembles;
    otherwise LGBM_1d vs RankRidge_5d would mix model effect and smoothing effect.
    """
    rows: list[dict] = []
    for window in config.smooth_windows:
        valid_s = smooth_predictions(valid, config.models, window)
        test_s = smooth_predictions(test, config.models, window)
        for model in config.models:
            score_col = f"_{model}_w{window}"
            valid_scored = valid_s.assign(**{score_col: valid_s[model]})
            test_scored = test_s.assign(**{score_col: test_s[model]})
            rows.extend(
                evaluate_split_pair(
                    section="single",
                    strategy=model,
                    window=window,
                    valid_df=valid_scored,
                    test_df=test_scored,
                    returns=returns,
                    score_col=score_col,
                    top_n=config.top_n,
                )
            )
    return rows


def evaluate_equal_weight(
    valid: pd.DataFrame,
    test: pd.DataFrame,
    returns: pd.DataFrame,
    config: EnsembleConfig,
    *,
    rank_space: bool,
) -> list[dict]:
    rows: list[dict] = []
    section = "equal_weight_rank" if rank_space else "equal_weight_raw"
    strategy = "EW_rank" if rank_space else "EW_raw"
    score_col = f"_{strategy}"
    ew_weights = weights_to_str(tuple(1.0 / len(config.models) for _ in config.models), config.models)
    for window in config.smooth_windows:
        valid_s = smooth_predictions(valid, config.models, window)
        test_s = smooth_predictions(test, config.models, window)
        valid_scored = add_equal_weight_score(valid_s, config.models, score_col, rank_space=rank_space)
        test_scored = add_equal_weight_score(test_s, config.models, score_col, rank_space=rank_space)
        rows.extend(
            evaluate_split_pair(
                section=section,
                strategy=strategy,
                window=window,
                valid_df=valid_scored,
                test_df=test_scored,
                returns=returns,
                score_col=score_col,
                top_n=config.top_n,
                weights=ew_weights,
            )
        )
    return rows


def evaluate_rank_ridge(
    valid: pd.DataFrame,
    test: pd.DataFrame,
    returns: pd.DataFrame,
    config: EnsembleConfig,
) -> list[dict]:
    rows: list[dict] = []
    score_col = "_rr"
    for window in config.smooth_windows:
        print(f"      window={window}d: fitting RidgeCV on valid ranked predictions...")
        valid_s = smooth_predictions(valid, config.models, window)
        test_s = smooth_predictions(test, config.models, window)
        weights, alpha, n_fit = fit_rank_ridge(valid_s, config.models, config.ridge_alphas)
        print(
            f"        alpha={alpha}, n_fit={n_fit:,}, "
            + ", ".join(f"{m}={weights[m]:.4f}" for m in config.models)
        )
        valid_scored = add_rank_ridge_score(valid_s, config.models, weights, score_col)
        test_scored = add_rank_ridge_score(test_s, config.models, weights, score_col)
        rows.extend(
            evaluate_split_pair(
                section="rank_ridge",
                strategy="RankRidge_rank",
                window=window,
                valid_df=valid_scored,
                test_df=test_scored,
                returns=returns,
                score_col=score_col,
                top_n=config.top_n,
                weights=weights_to_str(weights, config.models),
                ridge_alpha=alpha,
                n_fit=n_fit,
            )
        )
    return rows


def evaluate_ridge_raw(
    valid: pd.DataFrame,
    test: pd.DataFrame,
    returns: pd.DataFrame,
    config: EnsembleConfig,
) -> list[dict]:
    """Fit RidgeCV on valid raw predictions, apply on raw predictions.

    Mirrors evaluate_rank_ridge exactly — the only difference is that
    features and targets are raw predictions instead of daily ranks,
    so fit space and apply space are consistent.
    """
    rows: list[dict] = []
    score_col = "_raw_rr"
    for window in config.smooth_windows:
        print(f"      window={window}d: fitting RidgeCV on valid raw predictions...")
        valid_s = smooth_predictions(valid, config.models, window)
        test_s = smooth_predictions(test, config.models, window)
        weights, alpha, n_fit = fit_raw_ridge(valid_s, config.models, config.ridge_alphas)
        print(
            f"        alpha={alpha}, n_fit={n_fit:,}, "
            + ", ".join(f"{m}={weights[m]:.4f}" for m in config.models)
        )
        valid_scored = add_raw_ridge_score(valid_s, config.models, weights, score_col)
        test_scored = add_raw_ridge_score(test_s, config.models, weights, score_col)
        rows.extend(
            evaluate_split_pair(
                section="ridge_raw",
                strategy="Ridge_raw",
                window=window,
                valid_df=valid_scored,
                test_df=test_scored,
                returns=returns,
                score_col=score_col,
                top_n=config.top_n,
                weights=weights_to_str(weights, config.models),
                ridge_alpha=alpha,
                n_fit=n_fit,
            )
        )
    return rows


def evaluate_fixed_weight_grid(
    valid: pd.DataFrame,
    test: pd.DataFrame,
    returns: pd.DataFrame,
    config: EnsembleConfig,
) -> list[dict]:
    """Evaluate non-50/50 fixed-weight checks in rank space.

    50/50 is intentionally omitted because equal_weight_rank already covers it.
    These rows are diagnostics, not final parameter selection.
    """
    rows: list[dict] = []
    score_col = "_grid"
    n_total = len(config.smooth_windows) * len(config.grid_weights)
    counter = 0
    for window in config.smooth_windows:
        valid_s = smooth_predictions(valid, config.models, window)
        test_s = smooth_predictions(test, config.models, window)
        for weights in config.grid_weights:
            counter += 1
            if len(weights) != len(config.models):
                raise ValueError("grid_weights currently must match number of models")
            if all(abs(w - 1.0 / len(weights)) < 1e-12 for w in weights):
                # Defensive: 50/50 is already equal-weight.
                continue
            label = "GridRank_" + "_".join(f"{m}{w:.2f}" for m, w in zip(config.models, weights))
            print(f"      [{counter}/{n_total}] {label} window={window}d")
            valid_scored = add_weighted_rank_score(valid_s, config.models, weights, score_col)
            test_scored = add_weighted_rank_score(test_s, config.models, weights, score_col)
            rows.extend(
                evaluate_split_pair(
                    section="fixed_weight_rank",
                    strategy=label,
                    window=window,
                    valid_df=valid_scored,
                    test_df=test_scored,
                    returns=returns,
                    score_col=score_col,
                    top_n=config.top_n,
                    weights=weights_to_str(weights, config.models),
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Selection, reporting, and CLI
# ---------------------------------------------------------------------------


def mark_best_by_valid(results: pd.DataFrame, section: str, selection_metric: str) -> pd.DataFrame:
    """Mark one best window/strategy within a section using valid-only metric."""
    out = results.copy()
    valid_rows = out[(out["section"] == section) & (out["split"] == "valid")].copy()
    if valid_rows.empty:
        return out
    valid_rows = valid_rows.replace([np.inf, -np.inf], np.nan).dropna(subset=[selection_metric])
    if valid_rows.empty:
        return out
    idx = valid_rows[selection_metric].idxmax()
    best_strategy = out.loc[idx, "strategy"]
    best_window = out.loc[idx, "window"]
    mask = (out["section"] == section) & (out["strategy"] == best_strategy) & (out["window"] == best_window)
    out.loc[mask, "selected_by_valid"] = True
    return out


def result_display_columns(include_section: bool = False) -> list[str]:
    cols = [
        "strategy",
        "window",
        "split",
        "rank_ic_ir",
        "rank_ic_mean",
        "ic_ir",
        "sharpe",
        "maxdd",
        "turnover",
        "nav",
        "weights",
        "selected_by_valid",
    ]
    if include_section:
        return ["section"] + cols
    return cols


def build_summary(results: pd.DataFrame, config: EnsembleConfig) -> str:
    lines: list[str] = []
    lines.append("Strategy V1 Ensemble Summary")
    lines.append("============================")
    lines.append(f"Experiment dir:     {config.exp_dir}")
    lines.append(f"Returns path:       {config.returns_path}")
    lines.append(f"Models:             {', '.join(config.models)}")
    lines.append(f"Smooth windows:     {list(config.smooth_windows)}")
    lines.append(f"TopN:               {config.top_n}")
    lines.append(f"Selection metric:   valid {config.selection_metric}")
    lines.append("")
    lines.append("Important conventions")
    lines.append("---------------------")
    lines.append("- Model predictions target 5d_next_raw.")
    lines.append("- Smoothing is causal trailing smoothing of predictions, not returns.")
    lines.append("- Single models are evaluated under the same smoothing windows as ensembles.")
    lines.append("- Rank-Ridge is fit on daily-ranked valid predictions and applied to daily-ranked test predictions.")
    lines.append("- Backtest NAV always uses realized return_1d.")
    lines.append(f"- Best variants are selected by VALID {config.selection_metric}; test is report-only.")
    lines.append("- 50/50 grid is omitted because equal_weight_rank covers it.")
    lines.append("")

    def append_section(title: str, section: str) -> None:
        lines.append(title)
        lines.append("-" * len(title))
        view = results[results["section"] == section].copy()
        if view.empty:
            lines.append("(no rows)")
            lines.append("")
            return
        cols = result_display_columns(include_section=False)
        lines.append(view[cols].sort_values(["strategy", "window", "split"]).to_string(index=False))
        lines.append("")

    append_section("Single models", "single")
    append_section("Equal-weight raw-prediction scores", "equal_weight_raw")
    append_section("Equal-weight rank scores", "equal_weight_rank")
    append_section("Rank-Ridge rank scores", "rank_ridge")
    append_section("Ridge raw-fit-raw-apply scores", "ridge_raw")
    append_section("Fixed non-50/50 rank-weight checks", "fixed_weight_rank")

    selected = results[results["selected_by_valid"]].copy()
    lines.append("Selected-by-valid variants")
    lines.append("--------------------------")
    if selected.empty:
        lines.append("(none)")
    else:
        cols = result_display_columns(include_section=True)
        lines.append(selected[cols].sort_values(["section", "strategy", "window", "split"]).to_string(index=False))
    lines.append("")
    return "\n".join(lines)


def print_compact_tables(results: pd.DataFrame) -> None:
    """Print compact, readable valid/test comparison tables."""
    cols = result_display_columns(include_section=True)
    with pd.option_context("display.width", 220, "display.max_colwidth", 80, "display.max_rows", 250):
        for section in ALL_SECTIONS:
            view = results[results["section"] == section][cols]
            if view.empty:
                continue
            print(f"\n=== {section} ===")
            print(view.sort_values(["strategy", "window", "split"]).to_string(index=False))


def parse_tuple_of_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def parse_models(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in text.split(",") if x.strip())


def parse_weight_tuples(text: str) -> tuple[tuple[float, ...], ...]:
    """Parse semicolon-separated weight tuples, e.g. '0.75,0.25;0.25,0.75'."""
    tuples: list[tuple[float, ...]] = []
    if not text.strip():
        return tuple()
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        values = tuple(float(x.strip()) for x in item.split(",") if x.strip())
        if values:
            tuples.append(values)
    return tuple(tuples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Strategy V1 model ensembles from saved predictions.")
    parser.add_argument("--exp-dir", type=Path, default=Path("dataset/output/experiment_003"))
    parser.add_argument("--returns-path", type=Path, default=Path("dataset/processed/unified_daily_panel.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/strategy_v1"))
    parser.add_argument("--models", type=parse_models, default=DEFAULT_MODELS, help="Comma-separated model names.")
    parser.add_argument("--smooth-windows", type=parse_tuple_of_ints, default=DEFAULT_SMOOTH_WINDOWS, help="Comma-separated windows, e.g. 1,3,5")
    parser.add_argument(
        "--grid-weights",
        type=parse_weight_tuples,
        default=DEFAULT_GRID_WEIGHTS,
        help="Semicolon-separated weight tuples, e.g. '0.75,0.25;0.25,0.75'. Do not include equal-weight.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("rank_ic_ir", "ic_ir", "sharpe"),
        default=DEFAULT_SELECTION_METRIC,
        help="Valid-only metric used to mark selected variants. Default: rank_ic_ir.",
    )
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--no-legacy-test-path", action="store_true", help="Disable legacy predictions_{model}.parquet fallback for test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EnsembleConfig(
        exp_dir=args.exp_dir,
        returns_path=args.returns_path,
        output_dir=args.output_dir,
        models=args.models,
        smooth_windows=args.smooth_windows,
        grid_weights=args.grid_weights,
        top_n=args.top_n,
        allow_legacy_test_path=not args.no_legacy_test_path,
        selection_metric=args.selection_metric,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading predictions and returns...")
    valid = load_predictions(config, "valid")
    test = load_predictions(config, "test")
    returns = load_returns(config.returns_path)
    print(f"  valid:  {len(valid):,} rows, {valid['time'].nunique()} dates")
    print(f"  test:   {len(test):,} rows, {test['time'].nunique()} dates")
    print(f"  returns:{len(returns):,} rows, {returns['time'].nunique()} dates")

    rows: list[dict] = []

    n_models = len(config.models)
    n_windows = len(config.smooth_windows)

    print("\n[1/6] Single models with smoothing sweep...")
    print(f"      {n_models} model(s) × {n_windows} window(s) × 2 splits = {n_models * n_windows * 2} backtests")
    rows.extend(evaluate_single_models(valid, test, returns, config))

    print("\n[2/6] Equal-weight (raw prediction space)...")
    print(f"      {n_windows} smoothing window(s) × 2 splits = {n_windows * 2} backtests")
    rows.extend(evaluate_equal_weight(valid, test, returns, config, rank_space=False))

    print("\n[3/6] Equal-weight (rank space)...")
    print(f"      {n_windows} smoothing window(s) × 2 splits = {n_windows * 2} backtests")
    rows.extend(evaluate_equal_weight(valid, test, returns, config, rank_space=True))

    print("\n[4/6] Rank-Ridge (fit on valid ranks, apply on ranks)...")
    print(f"      {n_windows} smoothing window(s): fitting RidgeCV + backtesting valid + test")
    rows.extend(evaluate_rank_ridge(valid, test, returns, config))

    print("\n[5/6] Ridge (fit on valid raw, apply on raw)...")
    print(f"      {n_windows} smoothing window(s): fitting RidgeCV + backtesting valid + test")
    rows.extend(evaluate_ridge_raw(valid, test, returns, config))

    n_grid = len(config.grid_weights)
    print("\n[6/6] Fixed-weight grid (rank space, diagnostic)...")
    print(f"      {n_grid} weight combo(s) × {n_windows} window(s) × 2 splits = {n_grid * n_windows * 2} backtests")
    rows.extend(evaluate_fixed_weight_grid(valid, test, returns, config))

    results = pd.DataFrame(rows)
    for section in BEST_SELECTION_SECTIONS:
        results = mark_best_by_valid(results, section, config.selection_metric)

    results_path = config.output_dir / "ensemble_results.csv"
    summary_path = config.output_dir / "ensemble_summary.txt"
    results.to_csv(results_path, index=False)
    summary = build_summary(results, config)
    summary_path.write_text(summary, encoding="utf-8")

    print_compact_tables(results)
    print(f"\nSaved: {results_path}")
    print(f"Saved: {summary_path}")

    # Save best Ridge variant predictions for strategy use
    ridge_sections = ["rank_ridge", "ridge_raw"]
    ridge_results = results[results["section"].isin(ridge_sections) & (results["split"] == "valid")].copy()
    if not ridge_results.empty:
        ridge_results = ridge_results.replace([np.inf, -np.inf], np.nan).dropna(subset=[config.selection_metric])
        if not ridge_results.empty:
            best_idx = ridge_results[config.selection_metric].idxmax()
            best_section = results.loc[best_idx, "section"]
            best_window = int(results.loc[best_idx, "window"])
            best_weights_str = str(results.loc[best_idx, "weights"])
            print(f"\nBest Ridge variant: section={best_section}, window={best_window}d")
            print(f"  Weights: {best_weights_str}")

            test_s = smooth_predictions(test, config.models, best_window)
            if best_section == "rank_ridge":
                weights, _, _ = fit_rank_ridge(
                    smooth_predictions(valid, config.models, best_window),
                    config.models, config.ridge_alphas,
                )
                test_scored = add_rank_ridge_score(test_s, config.models, weights, "_best_ridge")
            else:
                weights, _, _ = fit_raw_ridge(
                    smooth_predictions(valid, config.models, best_window),
                    config.models, config.ridge_alphas,
                )
                test_scored = add_raw_ridge_score(test_s, config.models, weights, "_best_ridge")

            score_path = config.output_dir / "best_ridge_score_test.parquet"
            weights_path = config.output_dir / "best_ridge_weights.csv"
            test_scored[["time", "stock_id", "y_true", "_best_ridge"]].to_parquet(score_path, index=False)
            weights.to_csv(weights_path, header=["weight"])
            print(f"  Saved: {score_path}")
            print(f"  Saved: {weights_path}")


if __name__ == "__main__":
    main()
