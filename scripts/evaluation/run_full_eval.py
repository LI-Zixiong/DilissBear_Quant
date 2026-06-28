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
    BayesBlenderConfig, build_bayes_scores, apply_online_gate,
    _clean_industry, _weights_for, _sector_for,
)
from src.backtest.ensemble_utils import normalize_keys, backtest_score, smooth_predictions
from src.experiment.config import ExperimentConfig
from src.backtest.portfolio import PortfolioConfig
from src.backtest.real_backtest import RealBacktestConfig, run_real_backtest

# ── Config ──────────────────────────────────────────────────

MODELS = ["lightgbm", "xgboost", "dlinear", "gated_dwtcn"]
SUMMARY_KEYS = ["sharpe_ratio", "final_nav", "max_drawdown", "mean_turnover",
                "annualized_return", "hit_rate"]

@dataclass
class EvalConfig:
    eval_split: str = "test"
    mode: str = "full"  # "full" | "bayes" | "real"
    real_capital: float = 200_000.0  # capital for real backtest
    smooth_window: int = 3
    top_n: int = 50
    buffer_n: int = 80
    portfolio_strategy: str = "bin_weighted"  # "top_n" | "score_weighted" | "bin_weighted" | "top_n_buffer" | "bin_weighted_buffer"
    bayes_window: int = 63
    bayes_clip: float = 2.0
    output_dir: Path = Path("reports/strategy_v1/evidence")
    exp_dir: Path = Path(ExperimentConfig().output_dir)
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
    """Load open-to-open returns (1d_next_raw mapped to next trading date)."""
    panel = pd.read_parquet(cfg.factor_path, columns=["time", "stock_id", "1d_next_raw"])
    panel = normalize_keys(panel)
    panel["1d_next_raw"] = pd.to_numeric(panel["1d_next_raw"], errors="coerce")
    panel = panel.dropna(subset=["1d_next_raw"])
    panel = panel.sort_values(["time", "stock_id"])

    # Map 1d_next_raw at signal T → return_1d at T+1 (next trading date)
    dates = sorted(panel["time"].drop_duplicates())
    next_date = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    panel["time"] = panel["time"].map(next_date)
    panel = panel.dropna(subset=["time"])
    panel["time"] = pd.to_datetime(panel["time"])
    return panel.rename(columns={"1d_next_raw": "return_1d"})[["time", "stock_id", "return_1d"]].reset_index(drop=True)


def _load_returns_and_prices(cfg: EvalConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load open-to-open returns + open/pre_close prices."""
    panel = pd.read_parquet(cfg.returns_path)
    panel = normalize_keys(panel)

    # Prices from unified daily panel
    prices = panel[["time", "stock_id", "open", "pre_close"]].copy()
    prices[["open", "pre_close"]] = prices[["open", "pre_close"]].astype(float)

    # Returns from factor panel (open-to-open)
    ret = _load_returns(cfg)
    return ret, prices


def _load_industry(cfg: EvalConfig) -> pd.DataFrame:
    ind = pd.read_parquet(cfg.factor_path, columns=["time", "stock_id", "industry_sw"])
    ind = normalize_keys(ind)
    ind["industry_sw"] = _clean_industry(ind["industry_sw"])
    return ind


def _load_prices(cfg: EvalConfig) -> pd.DataFrame:
    """Load open/pre_close for real backtest (open-to-open, no close needed)."""
    prices = pd.read_parquet(cfg.returns_path, columns=["time", "stock_id", "open", "pre_close"])
    prices = normalize_keys(prices)
    prices[["open", "pre_close"]] = prices[["open", "pre_close"]].astype(float)
    return prices


# ── Bayes+Gate runner ──────────────────────────────────────────

def _run_bayes_v2(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    ind_panel: pd.DataFrame,
    cfg: EvalConfig,
    use_gate: bool = True,
    half_life: int = 252,
) -> dict:
    """Bayes hl=252 + OnlineExpertGate.  Uses build_bayes_scores + apply_online_gate."""
    # Need valid warmup for rolling calibration
    valid_preds = _load_valid_predictions(cfg)
    all_preds = pd.concat([valid_preds, predictions], ignore_index=True)
    all_preds = all_preds.sort_values(["time", "stock_id"]).reset_index(drop=True)

    # Smooth + calculate daily ranks
    all_data = smooth_predictions(all_preds, MODELS, cfg.smooth_window)
    for m in MODELS:
        all_data[f"{m}_r"] = all_data.groupby("time")[m].rank(pct=True)
    all_data = all_data.merge(ind_panel, on=["time", "stock_id"], how="left")

    # H = top 5% of 1d_next_raw (open-to-open).  Load from factor panel.
    if "1d_next_raw" not in all_data.columns:
        h_panel = pd.read_parquet(cfg.factor_path, columns=["time", "stock_id", "1d_next_raw"])
        h_panel = normalize_keys(h_panel)
        h_panel["1d_next_raw"] = pd.to_numeric(h_panel["1d_next_raw"], errors="coerce")
        all_data = all_data.merge(h_panel, on=["time", "stock_id"], how="left")
    all_data["H"] = all_data.groupby("time")["1d_next_raw"].transform(
        lambda x: (x.rank(pct=True) >= 0.95).astype(int))

    # Only eval split dates
    test_time_set = set(predictions["time"].unique())
    eval_dates = set(all_data.loc[all_data["time"].isin(test_time_set), "time"].unique())

    bayes_cfg = BayesBlenderConfig(
        models=tuple(MODELS),
        rolling_window=63, half_life=half_life,
        auto_alpha=True, auto_clip=True, auto_weights=True,
        clip_percentile=85.0, use_sector=True,
    )

    # Build Bayes scores
    merged_ranked = all_data[["time", "stock_id", "industry_sw", "H"]
                             + [f"{m}_r" for m in MODELS]].dropna(subset=["H"])
    if use_gate:
        scores_detail = build_bayes_scores(
            merged_ranked=merged_ranked, models=MODELS, config=bayes_cfg,
            eval_dates=eval_dates, eval_burnin=252, detailed=True,
        )
        scores_detail = normalize_keys(scores_detail)
        final_scores, gate_hist = apply_online_gate(
            scores_detail=scores_detail, returns_df=returns,
            models=MODELS, top_n=cfg.top_n, gate_burnin=63, reward_window=1,
            return_col="return_1d",
        )
        gate_out = cfg.output_dir / f"bayes_gate_history_{cfg.eval_split}.csv"
        gate_hist.to_csv(gate_out, index=False)
    else:
        final_scores = build_bayes_scores(
            merged_ranked=merged_ranked, models=MODELS, config=bayes_cfg,
            eval_dates=eval_dates, eval_burnin=252,
        )
        final_scores = normalize_keys(final_scores)

    # Save
    score_out = cfg.output_dir / f"bayes_scores_{cfg.eval_split}.parquet"
    final_scores.to_parquet(score_out, index=False)

    pfolio = PortfolioConfig(
        strategy=cfg.portfolio_strategy, top_n=cfg.top_n, buffer_n=cfg.buffer_n,
        pred_col="y_pred", stock_col="stock_id",
    )
    paper_result = backtest_score(
        final_scores.rename(columns={"bayes_score": "_bayes"}), returns, "_bayes",
        cfg.top_n, portfolio_config=pfolio,
    )
    return paper_result, final_scores


# ── Main ─────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    # All parameters have defaults in EvalConfig.  Only override when needed.
    parser.add_argument("--mode", default=None, choices=["full", "bayes", "real"])
    parser.add_argument("--eval-split", default=None)
    parser.add_argument("--bayes-window", type=int, default=None)
    parser.add_argument("--bayes-clip", type=float, default=None)
    parser.add_argument("--smooth-window", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--buffer-n", type=int, default=None)
    parser.add_argument("--portfolio-strategy", default=None,
                        choices=["top_n", "score_weighted", "bin_weighted", "top_n_buffer", "bin_weighted_buffer"])
    parser.add_argument("--no-gate", action="store_true", default=False)
    parser.add_argument("--half-life", type=int, default=252)
    args = parser.parse_args()

    cfg = EvalConfig()
    if args.mode is not None:             cfg.mode = args.mode
    if args.eval_split is not None:       cfg.eval_split = args.eval_split
    if args.bayes_window is not None:     cfg.bayes_window = args.bayes_window
    if args.bayes_clip is not None:       cfg.bayes_clip = args.bayes_clip
    if args.smooth_window is not None:    cfg.smooth_window = args.smooth_window
    if args.top_n is not None:            cfg.top_n = args.top_n
    if args.buffer_n is not None:         cfg.buffer_n = args.buffer_n
    if args.portfolio_strategy is not None: cfg.portfolio_strategy = args.portfolio_strategy
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Unified evaluation — {cfg.eval_split} (smooth={cfg.smooth_window}d)")
    if cfg.mode == "real":
        print(f"  real capital={cfg.real_capital:,.0f} CNY")

    print("\n[1] Loading data ...")
    preds = _load_predictions(cfg)
    if cfg.mode == "real":
        returns, prices = _load_returns_and_prices(cfg)
    else:
        returns = _load_returns(cfg)
    ind_panel = _load_industry(cfg)
    print(f"  predictions: {len(preds):,} rows, {preds['time'].nunique()} dates")
    print(f"  returns:     {len(returns):,} rows")
    if cfg.mode == "real":
        print(f"  prices:      {len(prices):,} rows")

    rows = []
    _add = lambda name, result: rows.append({"strategy": name, **{k: result.get(k, None) for k in SUMMARY_KEYS}})

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
            _add(m, s)
            print(f"  {m:>12s}: Sharpe={s.get('sharpe_ratio',0):.4f}  NAV={s.get('final_nav',0):.4f}")

        # ── EW ──
        print("\n[3/6] Equal-weight rank 3d ...")
        ew = equal_weight(preds, returns, MODELS, cfg.smooth_window, cfg.top_n)
        _add("EW rank 3d", ew)
        print(f"  EW rank 3d: Sharpe={ew.get('sharpe_ratio',0):.4f}  NAV={ew.get('final_nav',0):.4f}")

        # ── LGBM+XGB ──
        print("\n[4/6] LGBM+XGB rank 3d ...")
        dual = dual_model(preds, returns, MODELS, cfg.smooth_window, cfg.top_n)
        _add("LGBM+XGB rank 3d", dual)
        print(f"  LGBM+XGB: Sharpe={dual.get('sharpe_ratio',0):.4f}  NAV={dual.get('final_nav',0):.4f}")

        # ── Rank-Ridge ──
        print("\n[5/6] Rank-Ridge 3d ...")
        if cfg.eval_split == "test":
            valid_preds = _load_valid_predictions(cfg)
            rr, rr_w = rank_ridge(preds, returns, MODELS, valid_preds, cfg.smooth_window, cfg.top_n)
        else:
            rr, rr_w = rank_ridge(preds, returns, MODELS, preds, cfg.smooth_window, cfg.top_n)
        _add("Rank-Ridge 3d", rr)
        print(f"  RR: Sharpe={rr.get('sharpe_ratio',0):.4f}  NAV={rr.get('final_nav',0):.4f}")
        print(f"  weights: " + " ".join(f"{m}={rr_w[m]:.3f}" for m in MODELS))

    # ── Bayes V3 ──
    use_gate = not args.no_gate
    hl = args.half_life
    step_label = "6/6" if cfg.mode == "full" else "2/2"
    tag = f"Bayes+Gate hl={hl}" if use_gate else f"Bayes no-gate hl={hl}"
    print(f"\n[{step_label}] {tag} ...")
    bayes_paper, bayes_scores = _run_bayes_v2(preds, returns, ind_panel, cfg, use_gate=use_gate, half_life=hl)
    _add(tag, bayes_paper)
    print(f"  {tag}: Sharpe={bayes_paper.get('sharpe_ratio',0):.4f}  NAV={bayes_paper.get('final_nav',0):.4f}")

    # ── Real backtest (only in real mode) ──
    if cfg.mode == "real":
        print(f"\n[3/3] Real backtest (T+1 open, lots, limits) ...")
        rb_config = RealBacktestConfig(
            capital=cfg.real_capital,
            bin_lots=(0, 1, 2),
            max_stocks=cfg.top_n,
        )
        rb_result = run_real_backtest(bayes_scores, prices, rb_config)
        rb_summary = rb_result["summary"]

        paper_label = f"Bayes hl={hl} (paper)" if not use_gate else f"Bayes+Gate hl={hl} (paper)"
        real_label = f"Bayes hl={hl} (real)" if not use_gate else f"Bayes+Gate hl={hl} (real)"
        _add(paper_label, bayes_paper)
        _add(real_label, rb_summary)

        print(f"  {'':>16s}  {'Sharpe':>8s}  {'NAV':>8s}  {'MaxDD':>8s}  {'Turnover':>9s}  {'Cap%':>7s}  {'Stocks':>7s}")
        print(f"  {'Paper (%-wt)':>16s}  {bayes_paper['sharpe_ratio']:>8.4f}  {bayes_paper['final_nav']:>8.4f}  {bayes_paper['max_drawdown']:>7.1%}  {bayes_paper.get('mean_turnover',0):>8.1%}  {'100%':>7s}  {cfg.top_n:>7d}")
        print(f"  {'Real (lots)':>16s}  {rb_summary['sharpe_ratio']:>8.4f}  {rb_summary['final_nav']:>8.4f}  {rb_summary['max_drawdown']:>7.1%}  {'--':>9s}  {rb_summary['mean_capital_used_pct']:>6.1%}  {rb_summary['mean_n_positions']:>7.1f}")
        print(f"  {'':>16s}  Capital={rb_summary['capital']:,.0f} CNY  Filtered={rb_summary['mean_filtered']:.1f}/day  CapUsed={rb_summary['mean_capital_used']:,.0f}/day")

        # Save real backtest daily returns
        rb_out = cfg.output_dir / f"real_backtest_returns_{cfg.eval_split}.parquet"
        rb_result["daily_returns"].to_frame().to_parquet(rb_out)
        pos_out = cfg.output_dir / f"real_backtest_positions_{cfg.eval_split}.parquet"
        rb_result["daily_positions"].to_parquet(pos_out, index=False)
        print(f"  Saved: {rb_out}, {pos_out}")

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
