"""
Unified evaluation runner — all fusion methods, one script, one output CSV.

Usage:
    python -m scripts.evaluation.run_full_eval                      # defaults: test, w3, bayes_w63, clip2
    python -m scripts.evaluation.run_full_eval --eval-split valid   # override split
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.backtest.ensemble_methods import (
    single_model, equal_weight, dual_model, rank_ridge, prepare_predictions,
)
from src.backtest.bayes_blender import (
    BayesBlenderConfig, IndustryBayesCalibrator, _clean_industry,
)
from src.backtest.ensemble_utils import normalize_keys, backtest_score, smooth_predictions

# ── Config ──────────────────────────────────────────────────

MODELS = ["lightgbm", "xgboost", "dlinear", "gated_dwtcn"]

@dataclass
class EvalConfig:
    eval_split: str = "test"
    mode: str = "full"  # "full" | "bayes"
    smooth_window: int = 3
    top_n: int = 50
    bayes_window: int = 63
    bayes_clip: float = 2.0
    output_dir: Path = Path("reports/strategy_v1/evidence")
    exp_dir: Path = Path("dataset/output/experiment_004")
    returns_path: Path = Path("dataset/processed/unified_daily_panel.parquet")
    factor_path: Path = Path("dataset/processed/factor_panel_1500_54_ind.parquet")

    @property
    def tag(self):
        return f"{self.eval_split}_w{self.smooth_window}_bw{self.bayes_window}"


# ── Data loading ────────────────────────────────────────────

def _load_predictions(cfg: EvalConfig) -> pd.DataFrame:
    merged = None
    for m in MODELS:
        p = cfg.exp_dir / f"predictions_{cfg.eval_split}_{m}.parquet"
        if not p.exists():
            p = cfg.exp_dir / f"predictions_{m}.parquet"
        df = pd.read_parquet(p)
        df = normalize_keys(df)
        df = df[["time", "stock_id", "y_pred"]].rename(columns={"y_pred": m})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["time", "stock_id"], how="inner")
    return merged.sort_values(["time", "stock_id"]).reset_index(drop=True)


def _load_valid_predictions(cfg: EvalConfig) -> pd.DataFrame:
    merged = None
    for m in MODELS:
        p = cfg.exp_dir / f"predictions_valid_{m}.parquet"
        df = pd.read_parquet(p)
        df = normalize_keys(df)
        df = df[["time", "stock_id", "y_pred", "y_true"]].rename(columns={"y_pred": m})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df[["time", "stock_id", m]], on=["time", "stock_id"], how="inner")
    # y_true from first model
    merged["y_true"] = pd.read_parquet(cfg.exp_dir / f"predictions_valid_{MODELS[0]}.parquet")["y_true"]
    return merged.sort_values(["time", "stock_id"]).reset_index(drop=True)


def _load_returns(cfg: EvalConfig) -> pd.DataFrame:
    panel = pd.read_parquet(cfg.returns_path)
    panel = normalize_keys(panel)
    ret = panel[["time", "stock_id", "ret_daily"]].rename(columns={"ret_daily": "return_1d"}).copy()
    ret["return_1d"] = pd.to_numeric(ret["return_1d"], errors="coerce")
    return ret.dropna(subset=["return_1d"]).sort_values(["time", "stock_id"]).reset_index(drop=True)


def _load_industry(cfg: EvalConfig) -> pd.DataFrame:
    ind = pd.read_parquet(cfg.factor_path, columns=["time", "stock_id", "industry_sw"])
    ind = normalize_keys(ind)
    ind["industry_sw"] = _clean_industry(ind["industry_sw"])
    return ind


# ── Bayes runner ─────────────────────────────────────────────

def _run_bayes_v1(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    ind_panel: pd.DataFrame,
    cfg: EvalConfig,
) -> dict:
    # Need valid warmup for rolling calibration
    valid_preds = _load_valid_predictions(cfg)
    all_preds = pd.concat([valid_preds, predictions], ignore_index=True)
    all_preds = all_preds.sort_values(["time", "stock_id"]).reset_index(drop=True)

    # Smooth + rank on combined data
    all_data = smooth_predictions(all_preds, MODELS, cfg.smooth_window)
    from src.backtest.ensemble_utils import add_daily_model_ranks
    all_data = add_daily_model_ranks(all_data, MODELS)
    all_data = all_data.merge(ind_panel, on=["time", "stock_id"], how="left")

    # Forward return + H
    dates = np.sort(all_data["time"].unique())
    next_date = {pd.Timestamp(dates[i]): pd.Timestamp(dates[i+1]) for i in range(len(dates)-1)}
    all_data["return_date"] = all_data["time"].map(next_date)
    all_data = all_data.dropna(subset=["return_date"])
    fwd = returns.rename(columns={"time": "return_date"})
    all_data = all_data.merge(
        fwd[["return_date", "stock_id", "return_1d"]],
        on=["return_date", "stock_id"], how="inner",
    )
    all_data["H"] = all_data.groupby("time")["return_1d"].transform(
        lambda x: (x.rank(pct=True) > 0.95).astype(int))

    # Only eval split dates
    test_time_set = set(predictions["time"].unique())
    eval_dates = set(all_data.loc[all_data["time"].isin(test_time_set), "time"].unique())

    all_dates = sorted(all_data["time"].unique())
    cal_cfg = BayesBlenderConfig(rolling_window=cfg.bayes_window, llr_clip=cfg.bayes_clip)
    cal = IndustryBayesCalibrator(cal_cfg)
    import time as _time; _b_start = _time.perf_counter()

    scores_list = []
    first_eval = True
    for i, t in enumerate(all_dates):
        if t not in eval_dates or i < cfg.bayes_window:
            continue

        if first_eval:
            # Init with full window
            win = all_data[(all_data["time"] >= all_dates[i - cfg.bayes_window]) &
                             (all_data["time"] <= all_dates[i - 1])]
            full_ranks = {}
            for m in MODELS:
                full_ranks[m] = win[["time", "stock_id", "industry_sw", f"{m}_r", "H"]].rename(
                    columns={f"{m}_r": "rank_pct"})
            cal.update_incremental(full_ranks, None)  # full init
            first_eval = False
        else:
            # Incremental: +yesterday, -oldest
            evict_idx = i - cfg.bayes_window - 1
            drop_ranks = None
            if evict_idx >= 0:
                evict_t = all_dates[evict_idx]
                drop_ranks = {}
                for m in MODELS:
                    drop_ranks[m] = all_data.loc[
                        all_data["time"] == evict_t,
                        ["time", "stock_id", "industry_sw", f"{m}_r", "H"]
                    ].rename(columns={f"{m}_r": "rank_pct"})

            new_idx = i - 1
            new_ranks = {}
            for m in MODELS:
                new_ranks[m] = all_data.loc[
                    all_data["time"] == all_dates[new_idx],
                    ["time", "stock_id", "industry_sw", f"{m}_r", "H"]
                ].rename(columns={f"{m}_r": "rank_pct"})

            cal.update_incremental(new_ranks, drop_ranks)

        today = all_data.loc[all_data["time"] == t, ["time", "stock_id", "industry_sw"]]
        t_ranks = {}
        for m in MODELS:
            t_ranks[m] = all_data.loc[all_data["time"] == t,
                         ["time", "stock_id", f"{m}_r"]].rename(columns={f"{m}_r": "rank_pct"})
        scores = cal.score(today, t_ranks)
        scores = scores.reset_index()
        scores.columns = ["stock_id", "bayes_score"]
        scores["time"] = t
        scores_list.append(scores)
        if (i + 1) % 100 == 0:
            elapsed = _time.perf_counter() - _b_start
            eta = elapsed / (i - cfg.bayes_window + 1) * (len(all_dates) - i) if i > cfg.bayes_window else 0
            print(f"  Bayes [{i+1}/{len(all_dates)}] {t.date()}  elapsed={elapsed:.0f}s  ETA~{eta:.0f}s")

    all_scores = pd.concat(scores_list, ignore_index=True)
    all_scores["stock_id"] = all_scores["stock_id"].astype(str).str.strip().str.zfill(6)

    # Save daily Bayes scores
    score_out = cfg.output_dir / f"bayes_scores_{cfg.eval_split}.parquet"
    all_scores.to_parquet(score_out, index=False)

    return backtest_score(all_scores.rename(columns={"bayes_score": "_bayes"}), returns, "_bayes", cfg.top_n)


# ── Main ─────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    # All parameters have defaults in EvalConfig.  Only override when needed.
    parser.add_argument("--mode", default=None, choices=["full", "bayes"])
    parser.add_argument("--eval-split", default=None)
    parser.add_argument("--bayes-window", type=int, default=None)
    parser.add_argument("--bayes-clip", type=float, default=None)
    parser.add_argument("--smooth-window", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    args = parser.parse_args()

    cfg = EvalConfig()
    if args.mode is not None:             cfg.mode = args.mode
    if args.eval_split is not None:       cfg.eval_split = args.eval_split
    if args.bayes_window is not None:     cfg.bayes_window = args.bayes_window
    if args.bayes_clip is not None:       cfg.bayes_clip = args.bayes_clip
    if args.smooth_window is not None:    cfg.smooth_window = args.smooth_window
    if args.top_n is not None:            cfg.top_n = args.top_n
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Unified evaluation — {cfg.eval_split}")
    print(f"  smooth={cfg.smooth_window}d, bayes_w={cfg.bayes_window}, bayes_clip={cfg.bayes_clip}")

    print("\n[1] Loading data ...")
    preds = _load_predictions(cfg)
    returns = _load_returns(cfg)
    ind_panel = _load_industry(cfg)
    print(f"  predictions: {len(preds):,} rows, {preds['time'].nunique()} dates")
    print(f"  returns:     {len(returns):,} rows")

    rows = []

    if cfg.mode == "full":
        # ── Single models ──
        print("\n[2/6] Single models ...")
        import time as _t
        sm_results = {}
        for m in MODELS:
            _t0 = _t.perf_counter()
            sm_results[m] = single_model(preds, returns, [m], cfg.smooth_window, cfg.top_n)[m]
            print(f"  {m:>12s}: elapsed={_t.perf_counter()-_t0:.0f}s")
        for m, s in sm_results.items():
            rows.append({"strategy": m, **{k: s.get(k, None) for k in
                           ["sharpe_ratio", "final_nav", "max_drawdown", "mean_turnover",
                            "annualized_return", "hit_rate"]}})
            print(f"  {m:>12s}: Sharpe={s.get('sharpe_ratio',0):.4f}  NAV={s.get('final_nav',0):.4f}")

        # ── EW ──
        print("\n[3/6] Equal-weight rank 3d ...")
        ew = equal_weight(preds, returns, MODELS, cfg.smooth_window, cfg.top_n)
        rows.append({"strategy": "EW rank 3d", **{k: ew.get(k, None) for k in
                       ["sharpe_ratio", "final_nav", "max_drawdown", "mean_turnover",
                        "annualized_return", "hit_rate"]}})
        print(f"  EW rank 3d: Sharpe={ew.get('sharpe_ratio',0):.4f}  NAV={ew.get('final_nav',0):.4f}")

        # ── LGBM+XGB ──
        print("\n[4/6] LGBM+XGB rank 3d ...")
        dual = dual_model(preds, returns, MODELS, cfg.smooth_window, cfg.top_n)
        rows.append({"strategy": "LGBM+XGB rank 3d", **{k: dual.get(k, None) for k in
                         ["sharpe_ratio", "final_nav", "max_drawdown", "mean_turnover",
                          "annualized_return", "hit_rate"]}})
        print(f"  LGBM+XGB: Sharpe={dual.get('sharpe_ratio',0):.4f}  NAV={dual.get('final_nav',0):.4f}")

        # ── Rank-Ridge ──
        print("\n[5/6] Rank-Ridge 3d ...")
        if cfg.eval_split == "test":
            valid_preds = _load_valid_predictions(cfg)
            rr, rr_w = rank_ridge(preds, returns, MODELS, valid_preds, cfg.smooth_window, cfg.top_n)
        else:
            rr, rr_w = rank_ridge(preds, returns, MODELS, preds, cfg.smooth_window, cfg.top_n)
        rows.append({"strategy": "Rank-Ridge 3d", **{k: rr.get(k, None) for k in
                         ["sharpe_ratio", "final_nav", "max_drawdown", "mean_turnover",
                          "annualized_return", "hit_rate"]}})
        print(f"  RR: Sharpe={rr.get('sharpe_ratio',0):.4f}  NAV={rr.get('final_nav',0):.4f}")
        print(f"  weights: " + " ".join(f"{m}={rr_w[m]:.3f}" for m in MODELS))

    # ── Bayes V2 ──
    step_label = "6/6" if cfg.mode == "full" else "2/2"
    print(f"\n[{step_label}] Bayes V2 (rolling) ...")
    bayes_result = _run_bayes_v1(preds, returns, ind_panel, cfg)
    rows.append({"strategy": "Bayes V2", **{k: bayes_result.get(k, None) for k in
                    ["sharpe_ratio", "final_nav", "max_drawdown", "mean_turnover",
                     "annualized_return", "hit_rate"]}})
    print(f"  Bayes V2: Sharpe={bayes_result.get('sharpe_ratio',0):.4f}  NAV={bayes_result.get('final_nav',0):.4f}")

    # ── Save ──
    results_df = pd.DataFrame(rows)
    results_df = results_df.rename(columns={
        "sharpe_ratio": "Sharpe", "final_nav": "NAV",
        "max_drawdown": "MaxDD", "mean_turnover": "Turnover",
        "annualized_return": "AnnualRet", "hit_rate": "WinRate",
    })
    out_path = cfg.output_dir / f"comparison_{cfg.tag}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(results_df.sort_values("Sharpe", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
