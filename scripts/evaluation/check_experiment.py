"""
Model quality report: IC, pairwise correlation, backtest comparison.

Reads experiment_001 outputs and produces a quality summary.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = Path("dataset/output/experiment_001")
MODELS = ["lightgbm", "xgboost", "dlinear", "itransformer", "tsmixer"]


def load_predictions(model: str) -> pd.DataFrame:
    path = OUTPUT_DIR / f"predictions_{model}.parquet"
    if not path.exists():
        print(f"  [SKIP] {model} — predictions not found")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    df["stock_id"] = df["stock_id"].astype(str).str.strip()
    return df


def main():
    print("=" * 60)
    print("  Model Quality Report")
    print("=" * 60)

    preds: dict[str, pd.DataFrame] = {}
    for m in MODELS:
        df = load_predictions(m)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"]).dt.normalize()
            df["stock_id"] = df["stock_id"].astype(str).str.strip()
            preds[m] = df
            dates = pd.DatetimeIndex(df["time"].unique()).sort_values()
            print(f"  {m:15s}: {len(df):,} rows, "
                  f"{df['stock_id'].nunique()} stocks, "
                  f"{df['time'].nunique()} dates, "
                  f"range={dates.min().date()}~{dates.max().date()}")

    if not preds:
        print("No predictions found. Run run_experiment.py first.")
        return

    # ---- IC per model ----
    print(f"\n{'=' * 60}")
    print("  IC / Rank IC")
    print("=" * 60)
    ic_results = {}
    for model, df in preds.items():
        ic = df.groupby("time").apply(
            lambda g: g["y_true"].corr(g["y_pred"], method="pearson")
        )
        ric = df.groupby("time").apply(
            lambda g: g["y_true"].corr(g["y_pred"], method="spearman")
        )
        ic_results[model] = {
            "ic_mean": ic.mean(), "ic_std": ic.std(),
            "ic_ir": ic.mean() / ic.std() if ic.std() > 0 else 0,
            "ric_mean": ric.mean(), "ric_std": ric.std(),
            "ric_ir": ric.mean() / ric.std() if ric.std() > 0 else 0,
            "ic_pos_rate": (ic > 0).mean(),
            "ric_pos_rate": (ric > 0).mean(),
            "hit_rate": (df["y_true"] * df["y_pred"] > 0).mean(),
        }
        r = ic_results[model]
        print(f"\n  {model}:")
        print(f"    IC   mean={r['ic_mean']:.4f}  std={r['ic_std']:.4f}  "
              f"IR={r['ic_ir']:.3f}  pos%={r['ic_pos_rate']:.1%}")
        print(f"    Rank mean={r['ric_mean']:.4f}  std={r['ric_std']:.4f}  "
              f"IR={r['ric_ir']:.3f}  pos%={r['ric_pos_rate']:.1%}")
        print(f"    Hit  rate={r['hit_rate']:.4f}")

    # ---- Pairwise prediction correlation ----
    print(f"\n{'=' * 60}")
    print("  Pairwise Prediction Rank Correlation (daily avg)")
    print("=" * 60)
    models = sorted(preds.keys())
    corr_matrix = np.zeros((len(models), len(models)))
    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            if i < j:
                merged = preds[mi][["time", "stock_id", "y_pred"]].merge(
                    preds[mj][["time", "stock_id", "y_pred"]],
                    on=["time", "stock_id"], suffixes=("_a", "_b")
                )
                common_dates = merged["time"].nunique()
                common_stocks = merged["stock_id"].nunique()
                cors = []
                for _, g in merged.groupby("time"):
                    if len(g) < 10:
                        continue
                    r = g["y_pred_a"].rank().corr(g["y_pred_b"].rank())
                    if pd.notna(r):
                        cors.append(r)
                avg_corr = np.mean(cors) if cors else np.nan
                print(f"    {mi:15s} vs {mj:15s}: overlap={len(merged):,} rows, "
                      f"dates={common_dates}, stocks={common_stocks}, corr={avg_corr:.4f}")
                corr_matrix[i, j] = avg_corr
                corr_matrix[j, i] = avg_corr

    for i, mi in enumerate(models):
        row = "  " + "  ".join(
            f"{corr_matrix[i, j]:.3f}" if i != j else "  -  "
            for j in range(len(models))
        )
        print(f"    {mi:15s} {row}")

    # ---- Backtest comparison ----
    print(f"\n{'=' * 60}")
    print("  Backtest Comparison")
    print("=" * 60)
    for model in models:
        nav_path = OUTPUT_DIR / f"daily_nav_{model}.csv"
        ret_path = OUTPUT_DIR / f"daily_returns_{model}.csv"
        if nav_path.exists():
            nav = pd.read_csv(nav_path, index_col=0, parse_dates=True).iloc[:, 0]
            ret = pd.read_csv(ret_path, index_col=0, parse_dates=True).iloc[:, 0]
            sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
            ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (252 / len(ret)) - 1
            maxdd = (nav / nav.cummax() - 1).min()
            win_rate = (ret > 0).mean()
            print(f"  {model:15s}: Sharpe={sharpe:.4f}  AnnRet={ann_ret:.4f}  "
                  f"MaxDD={maxdd:.4f}  FinalNAV={nav.iloc[-1]:.4f}  "
                  f"WinRate={win_rate:.4f}")


if __name__ == "__main__":
    main()
