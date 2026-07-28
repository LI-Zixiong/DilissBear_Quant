"""Unified evaluation runner for the frozen three-model Bayes paths.

The two fusion modes share the *same* causal Layer-1 LLR calibration.  Their
only difference is whether the outer online gate is applied:

    llr  = sum of the three ``{model}_llr_blended`` contributions
    gate = online gate applied to those same three contributions

Examples
--------
Paper/full comparison (default):
    python -m scripts.evaluation.run_full_eval --mode llr
    python -m scripts.evaluation.run_full_eval --mode gate

Only run Bayes and the paper backtest:
    python -m scripts.evaluation.run_full_eval --mode llr --run-kind bayes

Run the complete account backtest (lots, limits, costs and slippage):
    python -m scripts.evaluation.run_full_eval --mode gate --run-kind real

Live scoring is intentionally limited to the diagnostic baseline:
    python -m scripts.evaluation.run_full_eval --mode llr --run-kind live
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
    single_model, equal_weight, rank_ridge,
)
from src.backtest.bayes_blender import (
    BayesBlenderConfig, build_fusion_scores,
    _clean_industry,
)
from src.backtest.ensemble_utils import normalize_keys, backtest_score, smooth_predictions
from src.experiment.config import ExperimentConfig
from src.backtest.portfolio import PortfolioConfig
from src.backtest.real_backtest import RealBacktestConfig, run_real_backtest

# ── Config ──────────────────────────────────────────────────

MODELS = ["lightgbm", "dlinear", "gated_dwtcn"]
SUMMARY_KEYS = ["sharpe_ratio", "final_nav", "max_drawdown", "mean_turnover",
                "annualized_return", "hit_rate"]

@dataclass
class EvalConfig:
    eval_split: str = "test"
    fusion_mode: str = "llr"  # "llr" | "gate"
    run_kind: str = "full"  # "full" | "bayes" | "real" | "live"
    real_capital: float = 200_000.0  # capital for real backtest
    smooth_window: int = 5
    top_n: int = 50
    buffer_n: int = 80
    portfolio_strategy: str = "bin_weighted"  # "top_n" | "score_weighted" | "bin_weighted" | "top_n_buffer" | "bin_weighted_buffer"
    bayes_window: int = 63
    bayes_half_life: int | None = None
    bayes_clip: float = 2.0
    output_dir: Path = Path("reports/strategy_v1/evidence")
    exp_dir: Path = Path(ExperimentConfig().output_dir)
    returns_path: Path = Path("dataset/processed/unified_daily_panel.parquet")
    factor_path: Path = Path("dataset/processed/factor_panel_1500_54_ind.parquet")

    @property
    def tag(self):
        decay = (
            f"hl{self.bayes_half_life}"
            if self.bayes_half_life is not None
            else f"bw{self.bayes_window}"
        )
        return f"{self.eval_split}_{self.fusion_mode}_w{self.smooth_window}_{decay}"


# ── Data loading ────────────────────────────────────────────

def _load_prediction_split(cfg: EvalConfig, split: str) -> pd.DataFrame:
    merged = None
    for m in MODELS:
        p = cfg.exp_dir / f"predictions_{split}_{m}.parquet"
        if not p.exists():
            legacy = cfg.exp_dir / f"predictions_{m}.parquet"
            if split == "test" and legacy.exists():
                p = legacy
            else:
                raise FileNotFoundError(
                    f"Missing {split} predictions for {m}: {p}"
                )
        df = pd.read_parquet(p)
        df = normalize_keys(df)
        keep = ["time", "stock_id", "y_pred"]
        if merged is None and "limit_status" in df.columns:
            keep.append("limit_status")
        df = df[keep].rename(columns={"y_pred": m})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["time", "stock_id"], how="inner")
    # Signal-day limit-up is known at T and blocks the post-close buy.
    # Do not remove limit-up names here. Selection must happen first; the
    # account executor then records a blocked buy instead of backfilling with a
    # lower-ranked stock.
    return merged.sort_values(["time", "stock_id"]).reset_index(drop=True)


def _load_valid_predictions(cfg: EvalConfig) -> pd.DataFrame:
    merged = None
    for m in MODELS:
        p = cfg.exp_dir / f"predictions_valid_{m}.parquet"
        df = pd.read_parquet(p)
        df = normalize_keys(df)
        keep = ["time", "stock_id", "y_pred", "y_true"]
        if merged is None and "limit_status" in df.columns:
            keep.append("limit_status")
        df = df[keep].rename(columns={"y_pred": m})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df[["time", "stock_id", m]], on=["time", "stock_id"], how="inner")
    return merged.sort_values(["time", "stock_id"]).reset_index(drop=True)


def _load_predictions(cfg: EvalConfig) -> pd.DataFrame:
    return _load_prediction_split(cfg, cfg.eval_split)


def _load_returns(cfg: EvalConfig) -> pd.DataFrame:
    """Load close-close returns (1d_next_raw = close(t+1)/close(t)-1, mapped to next date)."""
    panel = pd.read_parquet(cfg.factor_path, columns=["time", "stock_id", "1d_next_raw"])
    panel = normalize_keys(panel)
    panel["1d_next_raw"] = pd.to_numeric(panel["1d_next_raw"], errors="coerce")
    dates = sorted(panel["time"].drop_duplicates())
    panel = panel.dropna(subset=["1d_next_raw"])
    panel = panel.sort_values(["time", "stock_id"])

    # 1d_next_raw at signal T = close(T+1)/close(T)-1 → map to T+1 as return_1d
    next_date = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    panel["time"] = panel["time"].map(next_date)
    panel = panel.dropna(subset=["time"])
    panel["time"] = pd.to_datetime(panel["time"])
    return panel.rename(columns={"1d_next_raw": "return_1d"})[["time", "stock_id", "return_1d"]].reset_index(drop=True)


def _load_returns_and_prices(cfg: EvalConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load close-close returns + close/pre_close prices for real backtest."""
    panel = pd.read_parquet(cfg.returns_path)
    panel = normalize_keys(panel)

    # Prices from unified daily panel
    optional = ["up_limit", "high_limit", "limit_up_price", "down_limit", "low_limit",
                "limit_down_price", "is_paused", "paused", "suspend", "suspended",
                "is_suspended", "trade_status", "is_st", "st", "risk_warning"]
    prices = panel[["time", "stock_id", "close", "pre_close"]
                   + [c for c in optional if c in panel.columns]].copy()
    prices[["close", "pre_close"]] = prices[["close", "pre_close"]].astype(float)

    # Returns from factor panel (close-to-close)
    ret = _load_returns(cfg)
    return ret, prices


def _load_industry(cfg: EvalConfig) -> pd.DataFrame:
    ind = pd.read_parquet(cfg.factor_path, columns=["time", "stock_id", "industry_sw"])
    ind = normalize_keys(ind)
    ind["industry_sw"] = _clean_industry(ind["industry_sw"])
    return ind


def _load_prices(cfg: EvalConfig) -> pd.DataFrame:
    """Load close/pre_close for real backtest (close-close execution)."""
    prices = pd.read_parquet(cfg.returns_path)
    optional = ["up_limit", "high_limit", "limit_up_price", "down_limit", "low_limit",
                "limit_down_price", "is_paused", "paused", "suspend", "suspended",
                "is_suspended", "trade_status", "is_st", "st", "risk_warning"]
    prices = prices[["time", "stock_id", "close", "pre_close"]
                    + [c for c in optional if c in prices.columns]]
    prices = normalize_keys(prices)
    prices[["close", "pre_close"]] = prices[["close", "pre_close"]].astype(float)
    return prices


# ── Shared Layer-1 LLR + optional outer Gate ───────────────────

def _run_bayes(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    ind_panel: pd.DataFrame,
    cfg: EvalConfig,
    fusion_mode: str = "llr",
    half_life: int | None = None,
    rolling_window: int = 63,
) -> tuple[dict | None, pd.DataFrame]:
    """Run one of the two frozen fusion paths.

    Both paths use identical rolling LLR tables, auto alpha, auto clip and
    auto hierarchy/model weights.  ``gate`` adds exactly one operation on top:
    it reweights the already-built per-model ``llr_blended`` contributions
    using causally realised model returns.
    """
    if fusion_mode not in {"llr", "gate"}:
        raise ValueError(f"fusion_mode must be 'llr' or 'gate', got {fusion_mode!r}")
    # Need valid warmup for rolling calibration
    valid_preds = _load_valid_predictions(cfg)
    history_parts = [valid_preds]
    if cfg.eval_split == "live":
        history_parts.append(_load_prediction_split(cfg, "test"))
    all_preds = pd.concat(history_parts + [predictions], ignore_index=True)
    all_preds = all_preds.drop_duplicates(["time", "stock_id"], keep="last")
    all_preds = all_preds.sort_values(["time", "stock_id"]).reset_index(drop=True)

    # Smooth + calculate daily ranks
    all_data = smooth_predictions(all_preds, MODELS, cfg.smooth_window)
    for m in MODELS:
        all_data[f"{m}_r"] = all_data.groupby("time")[m].rank(pct=True)
    all_data = all_data.merge(ind_panel, on=["time", "stock_id"], how="left")

    # H = top 5% of 1d_next_raw (close-close forward return).
    if "1d_next_raw" not in all_data.columns:
        h_panel = pd.read_parquet(cfg.factor_path, columns=["time", "stock_id", "1d_next_raw"])
        h_panel = normalize_keys(h_panel)
        h_panel["1d_next_raw"] = pd.to_numeric(h_panel["1d_next_raw"], errors="coerce")
        all_data = all_data.merge(h_panel, on=["time", "stock_id"], how="left")
    def _head_label(x: pd.Series) -> pd.Series:
        ranks = x.rank(pct=True)
        labels = (ranks >= 0.95).astype(float)
        # A live/unrealised label must stay missing.  Treating it as H=0 would
        # contaminate calibration when that row later enters history.
        labels[x.isna()] = np.nan
        return labels

    all_data["H"] = all_data.groupby("time")["1d_next_raw"].transform(_head_label)

    # Only eval split dates
    test_time_set = set(predictions["time"].unique())
    eval_dates = set(all_data.loc[all_data["time"].isin(test_time_set), "time"].unique())

    bayes_cfg = BayesBlenderConfig(
        models=tuple(MODELS),
        rolling_window=rolling_window, half_life=half_life,
        # Frozen from the final diagnosis.  Do not disable auto_weights in
        # gate mode: that would change Layer 1 and invalidate the comparison.
        auto_alpha=True, auto_clip=True, auto_weights=True,
        llr_clip=cfg.bayes_clip, clip_percentile=85.0, use_sector=True,
    )

    # Build Bayes scores
    merged_ranked = all_data[["time", "stock_id", "industry_sw", "H"]
                             + [f"{m}_r" for m in MODELS]]
    # A live evaluation row has no H yet.  It is still scoreable because the
    # calibrator only consumes labels through t-1; retain those rows while
    # dropping missing labels from calibration history.
    merged_ranked = merged_ranked[
        merged_ranked["H"].notna() | merged_ranked["time"].isin(eval_dates)
    ]
    # The base module owns the branch so both modes are guaranteed to build
    # the same detailed Layer 1 first.  With a 63-day window Valid starts after
    # 63 observations (~471 days), not after the old hard-coded 252-day burn-in.
    final_scores, gate_hist = build_fusion_scores(
        merged_ranked=merged_ranked, models=MODELS, config=bayes_cfg,
        returns_df=returns, mode=fusion_mode,
        eval_dates=eval_dates, eval_burnin=rolling_window,
        top_n=cfg.top_n, gate_burnin=63, reward_window=1,
        return_col="return_1d",
    )
    final_scores = normalize_keys(final_scores)

    if fusion_mode == "gate":
        gate_out = cfg.output_dir / f"bayes_gate_history_{cfg.eval_split}_gate.csv"
        gate_hist.to_csv(gate_out, index=False)

    # Save
    score_out = cfg.output_dir / f"bayes_scores_{cfg.eval_split}_{fusion_mode}.parquet"
    final_scores.to_parquet(score_out, index=False)

    pfolio = PortfolioConfig(
        strategy=cfg.portfolio_strategy, top_n=cfg.top_n, buffer_n=cfg.buffer_n,
        pred_col="y_pred", stock_col="stock_id",
    )
    paper_result = None
    if cfg.eval_split != "live":
        paper_result = backtest_score(
            final_scores.rename(columns={"bayes_score": "_bayes"}), returns, "_bayes",
            cfg.top_n, portfolio_config=pfolio,
        )
    return paper_result, final_scores


# ── Main ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the shared Layer-1 Bayes LLR with or without the outer online gate."
    )
    parser.add_argument(
        "--mode", default="llr", choices=["llr", "gate"],
        help="Fusion path: direct LLR sum, or the same LLR contributions plus the outer gate.",
    )
    parser.add_argument(
        "--run-kind", default="full", choices=["full", "bayes", "real", "live"],
        help="Evaluation scope. 'real' calls the complete account backtest.",
    )
    parser.add_argument("--eval-split", default=None, choices=["valid", "test", "live"])
    parser.add_argument("--real-capital", type=float, default=None)
    parser.add_argument("--bayes-window", type=int, default=None)
    parser.add_argument("--bayes-clip", type=float, default=None)
    parser.add_argument("--smooth-window", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--buffer-n", type=int, default=None)
    parser.add_argument("--portfolio-strategy", default=None,
                        choices=["top_n", "score_weighted", "bin_weighted", "top_n_buffer", "bin_weighted_buffer"])
    parser.add_argument(
        "--half-life", type=int, default=None,
        help="Experimental override. Diagnostic baseline leaves this unset and uses rolling-window=63.",
    )
    parser.add_argument(
        "--rolling-window", type=int, default=63,
        help="Causal LLR calibration window; the frozen diagnostic default is 63.",
    )
    args = parser.parse_args()

    cfg = EvalConfig()
    cfg.fusion_mode = args.mode
    cfg.run_kind = args.run_kind
    if args.eval_split is not None:       cfg.eval_split = args.eval_split
    rolling_window = args.bayes_window if args.bayes_window is not None else args.rolling_window
    cfg.bayes_window = rolling_window
    if args.bayes_clip is not None:       cfg.bayes_clip = args.bayes_clip
    if args.smooth_window is not None:    cfg.smooth_window = args.smooth_window
    if args.top_n is not None:            cfg.top_n = args.top_n
    if args.buffer_n is not None:         cfg.buffer_n = args.buffer_n
    if args.portfolio_strategy is not None: cfg.portfolio_strategy = args.portfolio_strategy
    if args.real_capital is not None:     cfg.real_capital = args.real_capital
    cfg.bayes_half_life = args.half_life
    if cfg.run_kind == "live":
        cfg.eval_split = "live"
        if cfg.fusion_mode == "gate":
            parser.error(
                "--mode gate is not supported with --run-kind live: the attached runner "
                "does not persist/replay enough historical detailed scores to reconstruct "
                "the live gate state safely. Use --mode llr for live scoring."
            )
    elif cfg.eval_split == "live":
        parser.error("--eval-split live requires --run-kind live")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Unified evaluation — mode={cfg.fusion_mode}, run={cfg.run_kind}, "
        f"split={cfg.eval_split} (smooth={cfg.smooth_window}d)"
    )
    if cfg.run_kind == "real":
        print(f"  real capital={cfg.real_capital:,.0f} CNY")

    print("\n[1] Loading data ...")
    preds = _load_predictions(cfg)
    if cfg.run_kind == "real":
        returns, prices = _load_returns_and_prices(cfg)
    else:
        returns = _load_returns(cfg)
    ind_panel = _load_industry(cfg)
    print(f"  predictions: {len(preds):,} rows, {preds['time'].nunique()} dates")
    print(f"  returns:     {len(returns):,} rows")
    if cfg.run_kind == "real":
        print(f"  prices:      {len(prices):,} rows")

    rows = []
    _add = lambda name, result: rows.append({"strategy": name, **{k: result.get(k, None) for k in SUMMARY_KEYS}})

    if cfg.run_kind == "full":
        history_preds = (
            _load_valid_predictions(cfg) if cfg.eval_split == "test" else None
        )
        # ── Single models ──
        print("\n[2/5] Single models ...")
        import time as _t
        sm_results = {}
        for m in MODELS:
            _t0 = _t.perf_counter()
            sm_results[m] = single_model(
                preds, returns, [m], cfg.smooth_window, cfg.top_n,
                history_predictions=history_preds,
            )[m]
            print(f"  {m:>12s}: elapsed={_t.perf_counter()-_t0:.0f}s")
        for m, s in sm_results.items():
            _add(m, s)
            print(f"  {m:>12s}: Sharpe={s.get('sharpe_ratio',0):.4f}  NAV={s.get('final_nav',0):.4f}")

        # ── EW ──
        ew_label = f"EW rank {cfg.smooth_window}d"
        print(f"\n[3/5] Equal-weight rank {cfg.smooth_window}d ...")
        ew = equal_weight(
            preds, returns, MODELS, cfg.smooth_window, cfg.top_n,
            history_predictions=history_preds,
        )
        _add(ew_label, ew)
        print(f"  {ew_label}: Sharpe={ew.get('sharpe_ratio',0):.4f}  NAV={ew.get('final_nav',0):.4f}")

        # ── Rank-Ridge ──
        rr_label = f"Rank-Ridge {cfg.smooth_window}d"
        print(f"\n[4/5] Rank-Ridge {cfg.smooth_window}d ...")
        if cfg.eval_split == "test":
            valid_preds = history_preds
            rr, rr_w = rank_ridge(
                preds, returns, MODELS, valid_preds,
                cfg.smooth_window, cfg.top_n,
                history_predictions=valid_preds,
            )
        else:
            rr, rr_w = rank_ridge(preds, returns, MODELS, preds, cfg.smooth_window, cfg.top_n)
        _add(rr_label, rr)
        print(f"  RR: Sharpe={rr.get('sharpe_ratio',0):.4f}  NAV={rr.get('final_nav',0):.4f}")
        print(f"  weights: " + " ".join(f"{m}={rr_w[m]:.3f}" for m in MODELS))

    # ── Shared Layer-1 LLR, optionally followed by the outer Gate ──
    hl = args.half_life
    step_label = (
        "5/5" if cfg.run_kind == "full"
        else "2/3" if cfg.run_kind == "real"
        else "2/2"
    )
    bayes_span = f"hl={hl}" if hl is not None else f"w={rolling_window}"
    tag = f"Bayes+Gate {bayes_span}" if cfg.fusion_mode == "gate" else f"Bayes LLR {bayes_span}"
    print(f"\n[{step_label}] {tag} ...")
    bayes_paper, bayes_scores = _run_bayes(
        preds, returns, ind_panel, cfg,
        fusion_mode=cfg.fusion_mode, half_life=hl,
        rolling_window=rolling_window,
    )
    if cfg.run_kind == "live":
        if bayes_scores.empty:
            raise RuntimeError("Live Bayes scoring produced no rows")
        live_date = pd.Timestamp(bayes_scores["time"].max())
        print(f"  {tag}: {len(bayes_scores):,} scores for {live_date.date()}")
    else:
        if cfg.run_kind != "real":
            _add(tag, bayes_paper)
        print(f"  {tag}: Sharpe={bayes_paper.get('sharpe_ratio',0):.4f}  NAV={bayes_paper.get('final_nav',0):.4f}")

    # ── Real backtest (only in real mode) ──
    if cfg.run_kind == "real":
        print(f"\n[3/3] Real backtest (T close to T+1 close, lots, limits) ...")
        rb_config = RealBacktestConfig(
            capital=cfg.real_capital,
            position_sizing="budget",
            cash_ratio=0.98,
            max_stocks=cfg.top_n,
            min_commission=5.0,
        )

        # ── Bayes LLR real ──
        rb_bayes = run_real_backtest(bayes_scores, prices, rb_config)
        s_bayes = rb_bayes["summary"]
        bayes_paper_label = f"{tag} (paper)"
        bayes_real_label = f"{tag} (real)"
        _add(bayes_paper_label, bayes_paper)

        # ── Compute real turnover from daily holdings ──
        def _real_turnover_and_metrics(rb_result):
            """Compute notional turnover from executed trades only."""
            trades = rb_result["trade_log"]
            equity = rb_result["daily_equity"]
            if trades.empty or equity.empty:
                return {"mean_turnover": np.nan}
            work = trades.copy()
            work["date"] = pd.to_datetime(work["date"])
            work["gross"] = pd.to_numeric(work["gross"], errors="coerce").fillna(0.0)
            buys = work.loc[work["action"] == "BUY"].groupby("date")["gross"].sum()
            sells = work.loc[work["action"] == "SELL"].groupby("date")["gross"].sum()
            dates = equity.index.union(buys.index).union(sells.index).sort_values()
            b = buys.reindex(dates).fillna(0.0)
            s = sells.reindex(dates).fillna(0.0)
            e = equity.reindex(dates).ffill().bfill()
            turnover = 0.5 * (b + s) / e
            first_trade = work["date"].min()
            if first_trade in turnover.index and s.loc[first_trade] == 0.0:
                turnover.loc[first_trade] = b.loc[first_trade] / e.loc[first_trade]
            return {"mean_turnover": float(turnover.mean())}

        bayes_extra = _real_turnover_and_metrics(rb_bayes)
        s_bayes.update(bayes_extra)
        _add(bayes_real_label, s_bayes)

        # ── EW rank 5d real ──
        valid_preds_raw = _load_valid_predictions(cfg)
        ew_all = pd.concat([valid_preds_raw, preds], ignore_index=True)
        ew_all = ew_all.drop_duplicates(["time", "stock_id"], keep="last")
        ew_all = ew_all.sort_values(["time", "stock_id"]).reset_index(drop=True)
        ew_sm = smooth_predictions(ew_all, MODELS, cfg.smooth_window)
        for m in MODELS:
            ew_sm[f"{m}_r"] = ew_sm.groupby("time")[m].rank(pct=True)
        ew_sm["ew_score"] = ew_sm[[f"{m}_r" for m in MODELS]].mean(axis=1)
        ew_test = ew_sm[ew_sm["time"].isin(preds["time"].unique())].copy()
        ew_test["bayes_score"] = ew_test["ew_score"]
        ew_test = ew_test[["time", "stock_id", "bayes_score"]]

        rb_ew = run_real_backtest(ew_test, prices, rb_config)
        s_ew = rb_ew["summary"]
        ew_extra = _real_turnover_and_metrics(rb_ew)
        s_ew.update(ew_extra)
        ew_label = f"EW rank {cfg.smooth_window}d (real)"
        _add(ew_label, s_ew)

        print(f"  {'':>16s}  {'Sharpe':>8s}  {'NAV':>8s}  {'MaxDD':>8s}  {'AnnRet':>8s}  {'TO':>7s}  {'Hit':>7s}  {'MV%':>6s}  {'Pos':>6s}")
        for label, s in [("Bayes (paper)", bayes_paper), ("Bayes (real)", s_bayes), ("EW 5d (real)", s_ew)]:
            print(f"  {label:>16s}  {s.get('sharpe_ratio',0):>8.4f}  {s.get('final_nav',0):>8.4f}  {s.get('max_drawdown',0):>7.1%}  "
                  f"{s.get('annualized_return',0):>7.2%}  {s.get('mean_turnover',0):>6.2%}  {s.get('hit_rate',0):>6.2%}  "
                  f"{s.get('mean_market_value_pct',1):>5.0%}  {s.get('mean_n_positions',0):>5.0f}")
        print(f"  {'':>16s}  Capital={s_bayes['capital']:,.0f} CNY  "
              f"Bayes_filtered={s_bayes['mean_filtered']:.1f}/d  EW_filtered={s_ew['mean_filtered']:.1f}/d")

        # Save
        for tag_save, rb_result in [("bayes", rb_bayes), ("ew5d", rb_ew)]:
            out = cfg.output_dir / f"real_backtest_returns_{cfg.eval_split}_{tag_save}.parquet"
            rb_result["daily_returns"].to_frame().to_parquet(out)
            pos_out = cfg.output_dir / f"real_backtest_positions_{cfg.eval_split}_{tag_save}.parquet"
            rb_result["daily_positions"].to_parquet(pos_out, index=False)
            trades_out = cfg.output_dir / f"real_backtest_trades_{cfg.eval_split}_{tag_save}.parquet"
            rb_result["trade_log"].to_parquet(trades_out, index=False)
        print(f"  Saved: real_backtest_*_test_bayes.parquet & real_backtest_*_test_ew5d.parquet")

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
