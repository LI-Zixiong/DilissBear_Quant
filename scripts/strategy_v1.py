"""
Strategy V1 — buffer zone grid search.
in = [1%, 2%, 3%] of 1500, out = [4%, 8%, 12%] of 1500.
Sweep on valid, evaluate on test.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

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
W = {"lightgbm": 0.1381, "xgboost": 0.1233, "dlinear": 0.2291,
     "itransformer": 0.2624, "tsmixer": 0.2472}
BUY_COST, SELL_COST = 0.0003, 0.0008
N_TOTAL = 1500
GRID_IN = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
GRID_OUT = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18]


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


def smooth_ensemble(df):
    df = df.sort_values(["stock_id", "time"]).reset_index(drop=True)
    df["y_pred"] = sum(W[m] * df[m] for m in MODELS)
    df["y_pred"] = df.groupby("stock_id")["y_pred"].transform(
        lambda x: x.rolling(3, min_periods=1).mean())
    return df


def backtest_buffered(preds, returns, top_in, top_out):
    """Simple buffer: enter top_in, exit top_out, equal weight. With costs."""
    dates = pd.DatetimeIndex(preds["time"].unique()).sort_values()
    ret_lookup = returns.set_index(["time", "stock_id"])["return_1d"]
    held = set()
    daily_returns = []
    daily_turnover = []
    prev_weights = pd.Series(dtype=float)

    for d in dates:
        day = preds[preds["time"] == d].sort_values("y_pred", ascending=False)
        day["rank"] = range(1, len(day) + 1)

        in_set = set(day[day["rank"] <= top_in]["stock_id"])
        out_set = set(day[day["rank"] <= top_out]["stock_id"])
        keep = held & out_set
        new = [s for s in day["stock_id"] if s in in_set and s not in keep]
        selected = list(keep) + new[:max(0, top_in - len(keep))]
        weights = pd.Series(1.0 / max(len(selected), 1), index=selected, dtype=float)
        weights = weights / weights.sum()

        next_dates = dates[dates > d]
        if len(next_dates) == 0:
            break
        nd = next_dates[0]

        day_ret = sum(
            weights.get(s, 0) * ret_lookup.get((nd, s), 0) for s in weights.index
        )

        turn = 0.0
        if len(prev_weights) > 0:
            all_s = set(weights.index) | set(prev_weights.index)
            turn = sum(abs(weights.get(s, 0) - prev_weights.get(s, 0)) for s in all_s) * 0.5

        daily_returns.append(day_ret)
        daily_turnover.append(turn)
        prev_weights = weights.copy()
        held = set(selected)

    dr = np.nan_to_num(np.array(daily_returns), nan=0, posinf=0, neginf=0)
    dt = np.nan_to_num(np.array(daily_turnover), nan=0, posinf=0, neginf=0)
    dr = pd.Series(dr - dt * 0.5 * (BUY_COST + SELL_COST))
    nav = cumulative_nav(dr)
    summary = summarize_backtest(dr)
    summary["final_nav"] = float(nav.iloc[-1])
    summary["max_drawdown"] = float((nav / nav.cummax() - 1).min()) * (-1)
    summary["mean_turnover"] = float(dt.mean())
    return summary


def v0_baseline(test, returns):
    pfolio = PortfolioConfig(strategy="top_n", top_n=50, pred_col="y_pred", stock_col="stock_id")
    r = run_backtest(pred_df=test[["time", "stock_id", "y_pred"]], returns_df=returns,
                     portfolio_config=pfolio, return_col="return_1d", date_col="time", stock_col="stock_id")
    dr = np.nan_to_num(r["daily_returns"].values, nan=0, posinf=0, neginf=0)
    dt = np.nan_to_num(r["daily_turnover"].values, nan=0, posinf=0, neginf=0)
    dr = pd.Series(dr - dt * 0.5 * (BUY_COST + SELL_COST), index=r["daily_returns"].index)
    nav = cumulative_nav(dr); s = summarize_backtest(dr)
    s["final_nav"] = float(nav.iloc[-1])
    s["max_drawdown"] = float((nav / nav.cummax() - 1).min()) * (-1)
    s["mean_turnover"] = float(r["daily_turnover"].mean())
    return s


def main():
    ret = build_returns()
    test = load_test()
    test = smooth_ensemble(test)

    print("=== V0 Baseline (top-50, 3d smooth) ===")
    s0 = v0_baseline(test, ret)
    print(f"  Sharpe={s0['sharpe_ratio']:.4f}  MaxDD={s0['max_drawdown']:.4f}  "
          f"Turnover={s0['mean_turnover']:.4f}  NAV={s0['final_nav']:.4f}")

    # ---- VALID grid ----
    print("\n=== VALID Grid ===")
    valid = generate_valid()
    valid = smooth_ensemble(valid)
    grid = []
    for pin in GRID_IN:
        for pout in GRID_OUT:
            if pout <= pin: continue
            top_in = int(N_TOTAL * pin)
            top_out = int(N_TOTAL * pout)
            s = backtest_buffered(valid, ret, top_in, top_out)
            grid.append((pin, pout, top_in, top_out, s["sharpe_ratio"]))
            print(f"  in={pin:.0%}({top_in:>3d}) out={pout:.0%}({top_out:>3d}): valid_Sharpe={s['sharpe_ratio']:.4f}")
    grid.sort(key=lambda x: x[4], reverse=True)
    best_pin, best_pout, best_tin, best_tout, _ = grid[0]
    print(f"  Best: in={best_pin:.0%}({best_tin}) out={best_pout:.0%}({best_tout})")

    # ---- TEST all grid ----
    print(f"\n=== TEST Grid ===")
    print(f"{'in%':>6s} {'out%':>6s} {'in_n':>5s} {'out_n':>5s}  {'Sharpe':>8}  {'MaxDD':>8}  {'Turn':>8}  {'NAV':>8}")
    print("-" * 68)
    for pin, pout, top_in, top_out, _ in grid:
        s = backtest_buffered(test, ret, top_in, top_out)
        marker = " <-- best valid" if pin == best_pin and pout == best_pout else ""
        print(f"  {int(pin*100):>3d}%  {int(pout*100):>3d}%  {top_in:>5d} {top_out:>5d}  "
              f"{s['sharpe_ratio']:.4f}   {s['max_drawdown']:.4f}   "
              f"{s['mean_turnover']:.4f}   {s['final_nav']:.4f}{marker}")


if __name__ == "__main__":
    main()
