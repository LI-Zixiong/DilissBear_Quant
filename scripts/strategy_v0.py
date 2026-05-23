"""
Strategy V0 — prediction output smoothing.
Fits Rank-Ridge on valid (1d/3d/5d), sweeps output smooth window on valid,
evaluates on test with transaction costs.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest
from src.backtest.metrics import cumulative_nav, summarize_backtest
from src.backtest.portfolio import PortfolioConfig
from scripts.run_experiment import (
    ExperimentConfig, _load_experiment_raw_data, build_returns_frame_from_next_target,
)
from src.data.preprocess import PreprocessConfig, preprocess_panel_data

OUT = Path("dataset/output/experiment_001")
MODELS = ["lightgbm", "xgboost", "dlinear", "itransformer", "tsmixer"]
WINDOWS = [1, 3, 5]
BUY_COST = 0.0003
SELL_COST = 0.0008


def load_test():
    preds = {}
    for m in MODELS:
        df = pd.read_parquet(OUT / f"predictions_{m}.parquet")
        df["time"] = pd.to_datetime(df["time"]).dt.normalize()
        df["stock_id"] = df["stock_id"].astype(str).str.strip()
        preds[m] = df
    merged = preds["lightgbm"][["time", "stock_id", "y_true"]].reset_index(drop=True).copy()
    merged["lightgbm"] = preds["lightgbm"]["y_pred"].reset_index(drop=True)
    for m in ["xgboost", "dlinear", "itransformer", "tsmixer"]:
        merged = merged.merge(
            preds[m][["time", "stock_id", "y_pred"]].rename(columns={"y_pred": m}),
            on=["time", "stock_id"], how="inner")
    return merged


def generate_valid():
    from scripts.ensemble import generate_valid as gv
    return gv()


def build_returns():
    config = ExperimentConfig()
    raw = _load_experiment_raw_data(config.data_path)
    if "time" not in raw.columns and raw.index.name == "time":
        raw = raw.reset_index()
    clean = preprocess_panel_data(raw, PreprocessConfig(
        date_col=config.date_col, stock_col=config.stock_col,
        feature_cols=list(config.feature_cols), target_col=config.target_col,
        meta_cols=list(config.meta_cols),
        replace_inf_with_nan=True, drop_rows_with_missing_keys=True,
        drop_rows_with_missing_target=True, duplicate_policy="raise", sort_values=True,
    )).df
    ret = build_returns_frame_from_next_target(
        clean, config.date_col, config.stock_col,
        config.backtest_return_source, config.return_col)
    ret["time"] = pd.to_datetime(ret["time"]).dt.normalize()
    ret["stock_id"] = ret["stock_id"].astype(str).str.strip()
    return ret


def smooth(df, w):
    s = df.sort_values(["stock_id", "time"]).reset_index(drop=True)
    s["y_pred_s"] = s.groupby("stock_id")["y_pred"].transform(
        lambda x: x.rolling(w, min_periods=1).mean())
    return s[["time", "stock_id", "y_pred_s"]].rename(columns={"y_pred_s": "y_pred"})


def bt_with_cost(df, ret):
    pfolio = PortfolioConfig(strategy="top_n", top_n=50, pred_col="y_pred", stock_col="stock_id")
    result = run_backtest(
        pred_df=df, returns_df=ret, portfolio_config=pfolio,
        return_col="return_1d", date_col="time", stock_col="stock_id")
    daily_ret = np.nan_to_num(result["daily_returns"].values, nan=0, posinf=0, neginf=0)
    daily_turn = np.nan_to_num(result["daily_turnover"].values, nan=0, posinf=0, neginf=0)
    cost = daily_turn * 0.5 * (BUY_COST + SELL_COST)
    daily_ret = pd.Series(daily_ret - cost, index=result["daily_returns"].index)
    nav = cumulative_nav(daily_ret)
    summary = summarize_backtest(daily_ret)
    summary["final_nav"] = float(nav.iloc[-1])
    summary["max_drawdown"] = float((nav / nav.cummax() - 1).min()) * (-1)
    summary["mean_turnover"] = float(result["daily_turnover"].mean())
    return summary


def main():
    ret = build_returns()

    # ---- 1. Fit Rank-Ridge on valid (1d, 3d, 5d) ----
    print("Fitting Rank-Ridge on valid...")
    valid = generate_valid()
    valid = valid.sort_values(["stock_id", "time"]).reset_index(drop=True)
    valid["yt_r"] = valid.groupby("time")["y_true"].rank(pct=True)

    def pct_rank(s):
        return s.groupby(valid["time"]).rank(pct=True)

    def fit_ridge(df, suffix):
        rcols = [f"{m}{suffix}" for m in MODELS]
        fm = df[rcols + ["yt_r"]].notna().all(axis=1)
        ridge = RidgeCV(alphas=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0], fit_intercept=False)
        ridge.fit(df.loc[fm, rcols].values, df.loc[fm, "yt_r"].values)
        return pd.Series(ridge.coef_, index=MODELS) / ridge.coef_.sum()

    # 1d: raw model ranks
    for m in MODELS:
        valid[f"{m}_1d"] = pct_rank(valid[m])
    w_1d = fit_ridge(valid, "_1d")
    print(f"  1d weights: {dict(w_1d.round(4))}")

    # 3d: smoothed model ranks
    for m in MODELS:
        valid[f"{m}_s3"] = valid.groupby("stock_id")[m].transform(
            lambda x: x.rolling(3, min_periods=1).mean())
        valid[f"{m}_s3r"] = pct_rank(valid[f"{m}_s3"])
    w_3d = fit_ridge(valid, "_s3r")
    print(f"  3d weights: {dict(w_3d.round(4))}")

    # 5d
    for m in MODELS:
        valid[f"{m}_s5"] = valid.groupby("stock_id")[m].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        valid[f"{m}_s5r"] = pct_rank(valid[f"{m}_s5"])
    w_5d = fit_ridge(valid, "_s5r")
    print(f"  5d weights: {dict(w_5d.round(4))}")

    # ---- 2. VALID sweep (1d weights, output smooth) ----
    valid["y_pred"] = sum(w_1d[m] * valid[m] for m in MODELS)
    print(f"\n=== VALID sweep (1d weights) ===")
    best_w, best_s = 1, -999
    for window in WINDOWS:
        s = bt_with_cost(smooth(valid, window), ret)["sharpe_ratio"]
        print(f"  rolling_{window}d: valid_Sharpe={s:.4f}")
        if s > best_s: best_s, best_w = s, window
    print(f"  Best: rolling_{best_w}d")

    # ---- 3. TEST ----
    test = load_test()
    test["y_pred"] = sum(w_1d[m] * test[m] for m in MODELS)
    print(f"\n=== TEST (1d weights) ===")
    print(f"{'Method':>20s}  {'Sharpe':>8}  {'MaxDD':>8}  {'Turnover':>8}  {'NAV':>8}  {'WinRate':>8}")
    print("-" * 70)
    for window in WINDOWS:
        s = bt_with_cost(smooth(test, window), ret)
        marker = " <-- best" if window == best_w else ""
        print(f"  rolling_{window}d       {s['sharpe_ratio']:.4f}   {s['max_drawdown']:.4f}    "
              f"{s['mean_turnover']:.4f}   {s['final_nav']:.4f}    {s['hit_rate']:.4f}{marker}")

    # ---- 4. Bull/bear dynamic position sizing ----
    hs300 = pd.read_csv("dataset/input/original_data/TRD_Index.csv")
    hs300 = hs300[hs300["Indexcd"] == 300].copy()
    hs300["time"] = pd.to_datetime(hs300["Trddt"]).dt.normalize()
    hs300["bull"] = hs300["Clsindex"].pct_change(60) > 0
    bull_map = hs300.set_index("time")["bull"].to_dict()

    test_s = smooth(test, best_w)
    test_s["_bull"] = test_s["time"].map(bull_map).fillna(True)
    valid_s = smooth(valid, best_w)
    valid_s["_bull"] = valid_s["time"].map(bull_map).fillna(True)

    # Fast version: filter predictions to top-N per date, then single backtest
    pfolio = PortfolioConfig(strategy="top_n", top_n=50, pred_col="y_pred", stock_col="stock_id")

    for bn in [20, 30, 40, 50]:
        vs = valid_s.copy()
        vs["_n"] = np.where(vs["_bull"], 50, bn)
        # Keep top-N per date
        vs = vs.sort_values(["time", "y_pred"], ascending=[True, False])
        vs["_rank"] = vs.groupby("time").cumcount()
        vs = vs[vs["_rank"] < vs["_n"]].drop(columns=["_rank", "_n"])
        sv = bt_with_cost(vs, ret)
        print(f"  bear_n={bn:>3d}: valid_Sharpe={sv['sharpe_ratio']:.4f}")
        if bn == 20 or sv["sharpe_ratio"] > best_bs:
            best_bs = sv["sharpe_ratio"]
            best_bn = bn

    print(f"\n=== VALID Bull/Bear (bull=50, bear=[20,30,40,50]) ===")
    best_bn, best_bs = 50, -999
    for bn in [20, 30, 40, 50]:
        vs = valid_s.copy()
        vs["_n"] = np.where(vs["_bull"], 50, bn)
        vs = vs.sort_values(["time", "y_pred"], ascending=[True, False])
        vs["_rank"] = vs.groupby("time").cumcount()
        vs = vs[vs["_rank"] < vs["_n"]].drop(columns=["_rank", "_n"])
        s = bt_with_cost(vs, ret)["sharpe_ratio"]
        print(f"  bear_n={bn:>3d}: valid_Sharpe={s:.4f}")
        if s > best_bs: best_bs, best_bn = s, bn
    print(f"  Best: bear_n={best_bn}")

    print(f"\n=== TEST Bull/Bear ===")
    print(f"{'bear_n':>8s}  {'Sharpe':>8}  {'MaxDD':>8}  {'Turnover':>8}  {'NAV':>8}")
    print("-" * 52)
    for bn in [20, 30, 40, 50]:
        ts = test_s.copy()
        ts["_n"] = np.where(ts["_bull"], 50, bn)
        ts = ts.sort_values(["time", "y_pred"], ascending=[True, False])
        ts["_rank"] = ts.groupby("time").cumcount()
        ts = ts[ts["_rank"] < ts["_n"]].drop(columns=["_rank", "_n"])
        s = bt_with_cost(ts, ret)
        m = " <-- best valid" if bn == best_bn else ""
        print(f"  {bn:>6d}  {s['sharpe_ratio']:.4f}   {s['max_drawdown']:.4f}    "
              f"{s['mean_turnover']:.4f}   {s['final_nav']:.4f}{m}")


if __name__ == "__main__":
    main()
