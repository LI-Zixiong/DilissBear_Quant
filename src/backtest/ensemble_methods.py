"""
Pure-function ensemble fusion methods — no I/O, no CLI.
Callers handle prediction loading; these functions handle signal fusion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from src.backtest.engine import TransactionCostConfig, run_backtest
from src.backtest.portfolio import PortfolioConfig


# ── Shared helpers ──────────────────────────────────────────

def _backtest(
    scores_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    score_col: str,
    top_n: int = 50,
) -> dict:
    """Common backtest wrapper: score column → top-N equal-weight → summary."""
    pred = scores_df[["time", "stock_id", score_col]].rename(
        columns={score_col: "y_pred"}
    ).dropna()
    pfolio = PortfolioConfig(strategy="top_n", top_n=top_n, pred_col="y_pred", stock_col="stock_id")
    return run_backtest(
        pred_df=pred,
        returns_df=returns_df,
        portfolio_config=pfolio,
        return_col="return_1d",
        date_col="time",
        stock_col="stock_id",
        cost_config=TransactionCostConfig(),
    )["summary"]


def _smooth(df: pd.DataFrame, model_cols: list[str], window: int) -> pd.DataFrame:
    """Per-stock causal trailing mean, same as ensemble.smooth_predictions."""
    if window <= 1:
        return df
    out = df.sort_values(["stock_id", "time"]).reset_index(drop=True)
    for c in model_cols:
        out[c] = out.groupby("stock_id", sort=False)[c].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    return out.sort_values(["time", "stock_id"]).reset_index(drop=True)


def _smooth_with_history(
    df: pd.DataFrame,
    model_cols: list[str],
    window: int,
    history: pd.DataFrame | None,
) -> pd.DataFrame:
    """Causally smooth an evaluation split with earlier warm-up rows."""
    if history is None or history.empty or window <= 1:
        return _smooth(df, model_cols, window)
    eval_keys = df[["time", "stock_id"]].drop_duplicates()
    cols = ["time", "stock_id"] + model_cols
    combined = pd.concat([history[cols], df[cols]], ignore_index=True)
    combined = combined.drop_duplicates(["time", "stock_id"], keep="last")
    return _smooth(combined, model_cols, window).merge(
        eval_keys, on=["time", "stock_id"], how="inner",
    )


def _rank_pct(df: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    """Add {col}_r = daily cross-sectional percentile rank."""
    out = df.copy()
    for c in model_cols:
        out[f"{c}_r"] = out.groupby("time")[c].rank(pct=True)
    return out


# ── Public methods ──────────────────────────────────────────

def single_model(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    model_cols: list[str],
    smooth_window: int = 3,
    top_n: int = 50,
    history_predictions: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """Backtest each model's rank individually."""
    results = {}
    df = _rank_pct(_smooth_with_history(
        predictions, model_cols, smooth_window, history_predictions,
    ), model_cols)
    for m in model_cols:
        results[m] = _backtest(df.rename(columns={f"{m}_r": f"_rank_{m}"}), returns, f"_rank_{m}", top_n)
    return results


def equal_weight(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    model_cols: list[str],
    smooth_window: int = 3,
    top_n: int = 50,
    history_predictions: pd.DataFrame | None = None,
) -> dict:
    """Equal-weight average of model rank percentiles."""
    df = _rank_pct(_smooth_with_history(
        predictions, model_cols, smooth_window, history_predictions,
    ), model_cols)
    rank_cols = [f"{m}_r" for m in model_cols]
    df["_ew"] = df[rank_cols].mean(axis=1)
    return _backtest(df, returns, "_ew", top_n)


def dual_model(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    model_cols: list[str],
    smooth_window: int = 3,
    top_n: int = 50,
) -> dict:
    """Equal-weight of the first two models only (typically LGBM + XGBoost)."""
    return equal_weight(predictions, returns, model_cols[:2], smooth_window, top_n)


def rank_ridge(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    model_cols: list[str],
    valid_predictions: pd.DataFrame,
    smooth_window: int = 3,
    top_n: int = 50,
    alphas: list[float] | None = None,
    history_predictions: pd.DataFrame | None = None,
) -> tuple[dict, pd.Series]:
    """
    Rank-Ridge: fit RidgeCV on valid daily-ranked predictions,
    apply on eval split (typically test).
    Returns (summary, weights_series).
    """
    if alphas is None:
        alphas = [1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0, 5.0, 10.0]
    # Fit on valid
    v = _rank_pct(_smooth(valid_predictions, model_cols, smooth_window), model_cols)
    rank_cols = [f"{m}_r" for m in model_cols]
    v["yt_r"] = v.groupby("time")["y_true"].rank(pct=True)
    fit = v.dropna(subset=rank_cols + ["yt_r"])
    ridge = RidgeCV(alphas=alphas, fit_intercept=False)
    ridge.fit(fit[rank_cols].to_numpy(), fit["yt_r"].to_numpy())
    weights = pd.Series(ridge.coef_, index=model_cols)
    weights = weights / weights.abs().sum()

    # Apply on eval
    df = _rank_pct(_smooth_with_history(
        predictions, model_cols, smooth_window, history_predictions,
    ), model_cols)
    df["_rr"] = sum(float(weights[m]) * df[f"{m}_r"] for m in model_cols)
    return _backtest(df, returns, "_rr", top_n), weights


def prepare_predictions(
    preds_df: pd.DataFrame,
    model_cols: list[str],
    returns_df: pd.DataFrame,
    smooth_window: int = 3,
) -> pd.DataFrame:
    """Smooth then rank — shared pre-processing for all methods."""
    df = _smooth(preds_df, model_cols, smooth_window)
    return _rank_pct(df, model_cols)
