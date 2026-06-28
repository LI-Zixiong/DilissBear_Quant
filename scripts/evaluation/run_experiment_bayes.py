"""Apply Bayes+Gate to a standard (non-walk-forward) experiment output."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.bayes_blender import BayesBlenderConfig, build_bayes_scores, apply_online_gate
from src.backtest.engine import TransactionCostConfig, run_backtest
from src.backtest.portfolio import PortfolioConfig
from src.experiment.data import prepare_experiment_data
from src.experiment.config import ExperimentConfig
from src.experiment.returns import align_predictions_to_returns
from src.config.paths import PATHS

MODELS = ["lightgbm", "xgboost", "dlinear", "gated_dwtcn"]
META_COLS = ["industry_sw", "list_date", "1d_next_raw"]


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"]).dt.normalize()
    out["stock_id"] = out["stock_id"].astype(str).str.strip().str.zfill(6)
    return out


def add_model_ranks_and_H(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_keys(df)
    if "industry_sw" not in out.columns:
        raise ValueError("Bayes requires industry_sw column.")
    for m in MODELS:
        if m not in out.columns:
            raise ValueError(f"Missing model prediction column: {m}")
        out[f"{m}_r"] = out.groupby("time")[m].rank(pct=True)

    h_source = "1d_next_raw" if "1d_next_raw" in out.columns else "ret_daily"
    if h_source not in out.columns:
        raise ValueError(f"H source column '{h_source}' not found.")
    out[h_source] = pd.to_numeric(out[h_source], errors="coerce")
    out["H"] = (
        out.groupby("time")[h_source]
        .rank(pct=True)
        .ge(0.95)
        .astype(int)
    )
    keep = ["time", "stock_id", "industry_sw", "H"] + [f"{m}_r" for m in MODELS]
    for c in ["y_true", "ret_daily", "1d_next_raw", "list_date"]:
        if c in out.columns:
            keep.append(c)
    result = out[keep].dropna(subset=[f"{m}_r" for m in MODELS] + ["H"])
    return result


def evaluate_score(scores, returns_df, top_n, score_col="bayes_score"):
    pred = scores[["time", "stock_id", score_col]].rename(columns={score_col: "y_pred"}).dropna()
    aligned = align_predictions_to_returns(
        pred_df=pred, returns_df=returns_df, date_col="time",
        stock_col="stock_id", pred_col="y_pred", return_col="return_1d",
    )
    pfolio = PortfolioConfig(strategy="top_n", top_n=top_n, pred_col="y_pred", stock_col="stock_id")
    return run_backtest(
        pred_df=aligned, returns_df=returns_df, portfolio_config=pfolio,
        return_col="return_1d", date_col="time", stock_col="stock_id",
        cost_config=TransactionCostConfig(),
    )["summary"]


def summary_row(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "total_return": float(summary.get("total_return", np.nan)),
        "sharpe": float(summary.get("sharpe_ratio", np.nan)),
        "nav": float(summary.get("final_nav", np.nan)),
        "maxdd": float(summary.get("max_drawdown", np.nan)),
        "turnover": float(summary.get("mean_turnover", np.nan)),
        "hit_rate": float(summary.get("hit_rate", np.nan)),
    }


def load_wide_from_experiment(experiment_dir: Path, train_df, valid_df, test_df) -> pd.DataFrame:
    """Merge per-model predictions into wide format with meta from split data."""
    parts = []
    for split_name, split_df in [("test", test_df)]:
        preds = {}
        for m in MODELS:
            path = experiment_dir / f"predictions_{split_name}_{m}.parquet"
            if not path.exists():
                path = experiment_dir / f"predictions_{m}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Missing prediction file for {m}: {path}")
            df = pd.read_parquet(path)
            df = normalize_keys(df)
            preds[m] = df[["time", "stock_id", "y_pred"]].rename(columns={"y_pred": m})

        base = preds[MODELS[0]]
        for m in MODELS[1:]:
            base = base.merge(preds[m], on=["time", "stock_id"], how="inner", validate="one_to_one")

        # Merge meta from split data
        meta_cols_available = ["time", "stock_id"] + [c for c in META_COLS if c in split_df.columns]
        meta = split_df[meta_cols_available].copy()
        meta = normalize_keys(meta).drop_duplicates(["time", "stock_id"], keep="last")
        base = base.merge(meta, on=["time", "stock_id"], how="left", validate="one_to_one")
        parts.append(base)

    return pd.concat(parts, ignore_index=True).sort_values(["time", "stock_id"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=str, default="dataset/output/experiment_004")
    parser.add_argument("--half-life", type=int, default=252)
    parser.add_argument("--gate", action="store_true", default=True)
    parser.add_argument("--rolling-window", type=int, default=63)
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    print(f"=== Bayes+Gate on {exp_dir} (half_life={args.half_life}) ===")

    cfg = ExperimentConfig()
    data = prepare_experiment_data(cfg)

    wide = load_wide_from_experiment(exp_dir, data.train_df, data.valid_df, data.test_df)
    print(f"Wide test: {len(wide):,} rows, {wide['time'].nunique()} dates")

    merged_ranked = add_model_ranks_and_H(wide)
    test_dates = set(pd.to_datetime(wide["time"].unique()))

    bayes_cfg = BayesBlenderConfig(
        models=tuple(MODELS),
        rolling_window=args.rolling_window,
        half_life=args.half_life,
        auto_alpha=True, auto_clip=True,
        auto_weights=True, clip_percentile=85.0,
        use_sector=True,
    )

    if args.gate:
        scores_detail = build_bayes_scores(
            merged_ranked=merged_ranked, models=MODELS, config=bayes_cfg,
            eval_dates=test_dates, eval_burnin=252, detailed=True,
        )
        scores_detail = normalize_keys(scores_detail)
        scores, gate_hist = apply_online_gate(
            scores_detail=scores_detail, returns_df=data.returns_df,
            models=MODELS, top_n=50, gate_burnin=63, reward_window=1,
            return_col="return_1d",
        )
        gate_hist.to_csv(exp_dir / "bayes_gate_history.csv", index=False)
    else:
        scores = build_bayes_scores(
            merged_ranked=merged_ranked, models=MODELS, config=bayes_cfg,
            eval_dates=test_dates, eval_burnin=252,
        )

    scores = normalize_keys(scores)
    out_path = exp_dir / "bayes_scores_test.parquet"
    scores.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")

    summary = evaluate_score(scores, data.returns_df, top_n=50, score_col="bayes_score")
    s = summary_row(summary)
    print(f"\nBayes+Gate result:")
    print(f"  Total Return: {s['total_return']:.4f}")
    print(f"  Sharpe:       {s['sharpe']:.4f}")
    print(f"  NAV:          {s['nav']:.4f}")
    print(f"  MaxDD:        {s['maxdd']:.4f}")
    print(f"  Turnover:     {s['turnover']:.4f}")

    # Also evaluate single models for comparison
    print(f"\nSingle model baselines:")
    for m in MODELS:
        if m not in wide.columns:
            continue
        smry = evaluate_score(
            wide[["time", "stock_id", m]].rename(columns={m: "y_pred"}),
            data.returns_df, top_n=50, score_col="y_pred",
        )
        sr = summary_row(smry)
        print(f"  {m:12s}  SR={sr['sharpe']:.4f}  Ret={sr['total_return']:.4f}  MaxDD={sr['maxdd']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
