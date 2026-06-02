"""
Strategy V0 — prediction output smoothing.
Fits Rank-Ridge on valid, sweeps output smooth window on valid,
evaluates on test with transaction costs.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import TransactionCostConfig, run_backtest
from src.backtest.portfolio import PortfolioConfig
from src.experiment.returns import align_predictions_to_returns

EXP = Path("dataset/output/experiment_003")
MODELS = ["lightgbm", "dlinear"]
WINDOWS = [1, 3, 5]


def load_preds(split: str):
    parts = {}
    for m in MODELS:
        p = EXP / f"predictions_{split}_{m}.parquet"
        if not p.exists():
            p = EXP / f"predictions_{m}.parquet"
        df = pd.read_parquet(p)
        df["time"] = pd.to_datetime(df["time"])
        df["stock_id"] = df["stock_id"].astype(str).str.strip()
        parts[m] = df
    merged = parts[MODELS[0]][["time", "stock_id", "y_true", "y_pred"]].rename(
        columns={"y_pred": MODELS[0]})
    for m in MODELS[1:]:
        merged = merged.merge(
            parts[m][["time", "stock_id", "y_pred"]].rename(columns={"y_pred": m}),
            on=["time", "stock_id"], how="inner")
    return merged


def build_returns():
    u = pd.read_parquet("dataset/processed/unified_daily_panel.parquet")
    u["time"] = pd.to_datetime(u["time"])
    u["stock_id"] = u["stock_id"].astype(str).str.strip().str.zfill(6)
    ret = u[["time", "stock_id", "ret_daily"]].rename(columns={"ret_daily": "return_1d"})
    ret["return_1d"] = pd.to_numeric(ret["return_1d"], errors="coerce")
    return ret.dropna(subset=["return_1d"])


def smooth(df, w):
    s = df.sort_values(["stock_id", "time"]).reset_index(drop=True)
    s["y_pred_s"] = s.groupby("stock_id")["y_pred"].transform(
        lambda x: x.rolling(w, min_periods=1).mean())
    return s[["time", "stock_id", "y_pred_s"]].rename(columns={"y_pred_s": "y_pred"})


def bt_with_cost(df, ret):
    d = df[["time", "stock_id", "y_pred"]].dropna(subset=["y_pred"])
    d = align_predictions_to_returns(
        pred_df=d, returns_df=ret, date_col="time", stock_col="stock_id",
        pred_col="y_pred", return_col="return_1d")
    pfolio = PortfolioConfig(strategy="top_n", top_n=50, pred_col="y_pred", stock_col="stock_id")
    return run_backtest(
        pred_df=d, returns_df=ret, portfolio_config=pfolio,
        return_col="return_1d", date_col="time", stock_col="stock_id",
        cost_config=TransactionCostConfig())["summary"]


def main():
    ret = build_returns()
    valid = load_preds("valid")
    test = load_preds("test")

    # ---- Rank-Ridge on valid ----
    print("=== Rank-Ridge on valid ===")
    valid = valid.sort_values(["stock_id", "time"]).reset_index(drop=True)
    valid["yt_r"] = valid.groupby("time")["y_true"].rank(pct=True)
    for m in MODELS:
        valid[f"{m}_r"] = valid.groupby("time")[m].rank(pct=True)
    rank_cols = [f"{m}_r" for m in MODELS]
    fm = valid[rank_cols + ["yt_r"]].notna().all(axis=1)
    ridge = RidgeCV(alphas=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0], fit_intercept=False)
    ridge.fit(valid.loc[fm, rank_cols].values, valid.loc[fm, "yt_r"].values)
    w = pd.Series(ridge.coef_, index=MODELS) / ridge.coef_.sum()
    print(f"  weights: {dict(w.round(4))}")

    valid["y_pred"] = sum(w[m] * valid[m] for m in MODELS)
    test["y_pred"] = sum(w[m] * test[m] for m in MODELS)

    # ---- Smooth sweep on valid ----
    print(f"\n=== Smooth sweep on valid ===")
    best_w, best_s = 1, -999
    for window in WINDOWS:
        s = bt_with_cost(smooth(valid, window), ret)["sharpe_ratio"]
        print(f"  rolling_{window}d: Sharpe={s:.4f}")
        if s > best_s: best_s, best_w = s, window
    print(f"  Best: rolling_{best_w}d")

    # ---- Test ----
    print(f"\n=== TEST ===")
    print(f"{'Window':>10s}  {'Sharpe':>8s}  {'MaxDD':>8s}  {'Turnover':>8s}  {'NAV':>8s}")
    print("-" * 52)
    for window in WINDOWS:
        s = bt_with_cost(smooth(test, window), ret)
        marker = " <--" if window == best_w else ""
        print(f"  {window:>3d}d     {s['sharpe_ratio']:.4f}   {s['max_drawdown']:.4f}    "
              f"{s['mean_turnover']:.4f}   {s['final_nav']:.4f}{marker}")


if __name__ == "__main__":
    main()
