"""
Ensemble evaluation: load existing valid/test predictions, run Rank-Ridge + grid.
Uses predictions already saved by run_experiment — no retraining.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from src.backtest.engine import TransactionCostConfig, run_backtest
from src.backtest.portfolio import PortfolioConfig
from src.experiment.returns import align_predictions_to_returns

EXP = Path("dataset/output/experiment_003")
MODELS = ["lightgbm", "dlinear"]
COMBO_WEIGHTS = [(0.75, 0.25), (0.25, 0.75)]
SMOOTH_WINDOWS = [1, 3, 5]  # output smoothing windows to sweep


def load_preds(split: str):
    """Load predictions for all models, inner join."""
    parts = {}
    for m in MODELS:
        p = EXP / f"predictions_{split}_{m}.parquet"
        if not p.exists():
            p = EXP / f"predictions_{m}.parquet"  # legacy test path
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


def smooth(df, cols, w):
    """Per-stock rolling mean over w days for prediction columns."""
    s = df.sort_values(["stock_id", "time"]).reset_index(drop=True)
    for c in cols:
        s[c] = s.groupby("stock_id")[c].transform(lambda x: x.rolling(w, min_periods=1).mean())
    return s


def build_returns():
    """Build returns from ret_daily in unified panel."""
    u = pd.read_parquet("dataset/processed/unified_daily_panel.parquet")
    u["time"] = pd.to_datetime(u["time"])
    u["stock_id"] = u["stock_id"].astype(str).str.strip().str.zfill(6)
    ret = u[["time", "stock_id", "ret_daily"]].rename(columns={"ret_daily": "return_1d"}).copy()
    ret["return_1d"] = pd.to_numeric(ret["return_1d"], errors="coerce")
    return ret.dropna(subset=["return_1d"])


def bt(df, returns, yp_col="_yp"):
    """Backtest ensemble predictions."""
    d = df[["time", "stock_id", yp_col]].rename(columns={yp_col: "y_pred"}).dropna()
    d = align_predictions_to_returns(pred_df=d, returns_df=returns,
        date_col="time", stock_col="stock_id",
        pred_col="y_pred", return_col="return_1d")
    pfolio = PortfolioConfig(strategy="top_n", top_n=50, pred_col="y_pred", stock_col="stock_id")
    return run_backtest(pred_df=d, returns_df=returns, portfolio_config=pfolio,
        return_col="return_1d", date_col="time", stock_col="stock_id",
        cost_config=TransactionCostConfig())["summary"]


def main():
    print("Loading predictions...")
    test = load_preds("test")
    valid = load_preds("valid")
    returns = build_returns()
    print(f"  valid: {len(valid):,} rows  test: {len(test):,} rows  returns: {len(returns):,} rows")

    # 1. Single-model baselines
    print("\n=== Single-model baselines ===")
    for m in MODELS:
        s = bt(test.assign(_yp=test[m]), returns)
        print(f"  {m:>12s}: Sharpe={s['sharpe_ratio']:.4f}  MaxDD={s['max_drawdown']:.4f}  "
              f"NAV={s['final_nav']:.4f}  TO={s['mean_turnover']:.4f}")

    # 2. Equal-Weight
    print("\n=== Equal-Weight ===")
    test["_ew"] = test[MODELS].mean(axis=1)
    s_ew = bt(test, returns, "_ew")
    print(f"  EW: Sharpe={s_ew['sharpe_ratio']:.4f}  MaxDD={s_ew['max_drawdown']:.4f}  "
          f"NAV={s_ew['final_nav']:.4f}  TO={s_ew['mean_turnover']:.4f}")

    # 3. Rank-Ridge on valid — sweep smooth windows
    print(f"\n=== Rank-Ridge (smooth sweep: {SMOOTH_WINDOWS}) ===")
    best_rr_w, best_rr_wind = None, 1
    best_rr_s = -999
    for w in SMOOTH_WINDOWS:
        vs = smooth(valid.copy(), MODELS, w)
        ts = smooth(test.copy(), MODELS, w)
        for m in MODELS:
            vs[f"{m}_r"] = vs.groupby("time")[m].rank(pct=True)
        vs["yt_r"] = vs.groupby("time")["y_true"].rank(pct=True)
        rank_cols = [f"{m}_r" for m in MODELS]
        ridge = RidgeCV(alphas=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0], fit_intercept=False)
        ridge.fit(vs[rank_cols].values, vs["yt_r"].values)
        w_rr = pd.Series(ridge.coef_, index=MODELS)
        w_rr = w_rr / w_rr.abs().sum()

        ts["_rr"] = sum(w_rr[m] * ts[m] for m in MODELS)
        s_rr = bt(ts, returns, "_rr")
        print(f"  {w}d_smooth: weights={dict(w_rr.round(4))}  "
              f"Sharpe={s_rr['sharpe_ratio']:.4f}  MaxDD={s_rr['max_drawdown']:.4f}  "
              f"NAV={s_rr['final_nav']:.4f}  TO={s_rr['mean_turnover']:.4f}")
        if s_rr["sharpe_ratio"] > best_rr_s:
            best_rr_s, best_rr_wind, best_rr_w = s_rr["sharpe_ratio"], w, w_rr
    print(f"  Best: {best_rr_wind}d_smooth  weights={dict(best_rr_w.round(4))}  "
          f"Sharpe={best_rr_s:.4f}")

    # 4. Grid
    print(f"\n=== Grid ({len(COMBO_WEIGHTS)} combos) ===")
    for w_l, w_d in COMBO_WEIGHTS:
        test["_gr"] = w_l * test[MODELS[0]] + w_d * test[MODELS[1]]
        s = bt(test, returns, "_gr")
        print(f"  LGBM={w_l:.2f}_DL={w_d:.2f}: Sharpe={s['sharpe_ratio']:.4f}  "
              f"MaxDD={s['max_drawdown']:.4f}  NAV={s['final_nav']:.4f}  TO={s['mean_turnover']:.4f}")


if __name__ == "__main__":
    main()
