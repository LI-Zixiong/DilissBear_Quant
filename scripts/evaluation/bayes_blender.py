"""
Run Industry-Conditional Rolling Bayes model blending.

Loads existing model predictions, calibrates per-industry LLR tables on a
rolling window, and outputs daily Bayes scores + backtest.

Recommended usage:
    # 1) Tune on valid
    python -m scripts.evaluation.bayes_blender --eval-split valid

    # 2) Final test, using valid as rolling warm-up history
    python -m scripts.evaluation.bayes_blender --eval-split test

Notes:
    In test mode, valid predictions are loaded by default as warm-up history,
    but final reported scores/backtest only use test dates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.backtest.bayes_blender import (
    BayesBlenderConfig,
    build_bayes_scores,
    _clean_industry,
)
from scripts.evaluation.ensemble import (
    add_daily_model_ranks,
    backtest_score,
    normalize_keys,
    smooth_predictions,
)

MODELS = ["lightgbm", "xgboost", "dlinear", "gated_dwtcn"]
EXP_DIR = Path("dataset/output/experiment_003")
RETURNS_PATH = Path("dataset/processed/unified_daily_panel.parquet")
FACTOR_PATH = Path("dataset/processed/factor_panel_1500_54_ind.parquet")


def _load_one_split(split: str, exp_dir: Path) -> pd.DataFrame:
    """Load and inner-join all model predictions for one split."""
    print(f"Loading predictions for split={split} ...")
    merged = None
    for m in MODELS:
        p = exp_dir / f"predictions_{split}_{m}.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing prediction file: {p}\n"
                f"Expected split-specific file predictions_{split}_{m}.parquet."
            )
        df = pd.read_parquet(p)
        df = normalize_keys(df)
        df = df[["time", "stock_id", "y_pred"]].rename(columns={"y_pred": m})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["time", "stock_id"], how="inner")

    if merged is None or merged.empty:
        raise ValueError(f"No predictions loaded for split={split}")

    merged["split"] = split
    merged = merged.sort_values(["time", "stock_id"]).reset_index(drop=True)
    print(f"  {len(merged):,} rows, {merged['time'].nunique()} dates")
    return merged


def _load_predictions(eval_split: str, use_valid_warmup: bool, exp_dir: Path) -> pd.DataFrame:
    if eval_split == "test" and use_valid_warmup:
        splits = ["valid", "test"]
    else:
        splits = [eval_split]

    frames = [_load_one_split(s, exp_dir) for s in splits]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["time", "stock_id"]).reset_index(drop=True)
    print(
        f"Prediction panel: {len(merged):,} rows, "
        f"{merged['time'].nunique()} dates, splits={splits}"
    )
    return merged


def _load_returns_and_industry() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading returns + industry ...")
    ret_panel = pd.read_parquet(RETURNS_PATH)
    ret_panel = normalize_keys(ret_panel)
    returns = ret_panel[["time", "stock_id", "ret_daily"]].rename(
        columns={"ret_daily": "return_1d"}
    )
    returns["return_1d"] = pd.to_numeric(returns["return_1d"], errors="coerce")
    returns = returns.dropna(subset=["return_1d"])

    ind_panel = pd.read_parquet(FACTOR_PATH, columns=["time", "stock_id", "industry_sw"])
    ind_panel = normalize_keys(ind_panel)
    ind_panel["industry_sw"] = _clean_industry(ind_panel["industry_sw"])
    return returns, ind_panel


def _attach_industry_and_forward_return(
    merged: pd.DataFrame,
    returns: pd.DataFrame,
    ind_panel: pd.DataFrame,
    smooth_window: int = 3,
) -> pd.DataFrame:
    print(f"Applying {smooth_window}d causal per-stock smoothing to model predictions ...")
    merged = smooth_predictions(merged, MODELS, smooth_window)
    print("Computing daily ranks + H ...")
    ranked = add_daily_model_ranks(merged, MODELS)
    ranked = ranked.merge(ind_panel, on=["time", "stock_id"], how="left")
    ranked["industry_sw"] = _clean_industry(ranked["industry_sw"])

    # Forward return alignment: score at date t uses return on next prediction date.
    dates = np.sort(ranked["time"].unique())
    next_date = {
        pd.Timestamp(dates[i]): pd.Timestamp(dates[i + 1])
        for i in range(len(dates) - 1)
    }
    ranked["return_date"] = ranked["time"].map(next_date)
    ranked = ranked.dropna(subset=["return_date"])

    fwd = returns.rename(columns={"time": "return_date"})
    ranked = ranked.merge(
        fwd[["return_date", "stock_id", "return_1d"]],
        on=["return_date", "stock_id"],
        how="inner",
    )

    # H = future 1d return in same-date global top 5%.
    ranked["H"] = ranked.groupby("time")["return_1d"].transform(
        lambda x: (x.rank(pct=True) > 0.95).astype(int)
    )

    print(
        f"  calibratable days: {ranked['time'].nunique()}, "
        f"rows: {len(ranked):,}, head_rate={ranked['H'].mean():.4f}"
    )
    return ranked


def _print_industry_exposure(all_scores: pd.DataFrame, ind_panel: pd.DataFrame) -> None:
    print("\nComputing industry exposure ...")
    tmp = all_scores.merge(ind_panel, on=["time", "stock_id"], how="left")
    tmp["industry_sw"] = _clean_industry(tmp["industry_sw"])
    top50 = tmp.sort_values(["time", "bayes_score"], ascending=[True, False])
    top50 = top50.groupby("time").head(50)
    ind_weights = top50.groupby("industry_sw")["stock_id"].count() / len(top50)
    ind_weights = ind_weights.sort_values(ascending=False)
    print("  Top 10 industries by avg weight:")
    for ind, w in ind_weights.head(10).items():
        bar = "#" * int(w * 50)
        print(f"    {ind:>5s}: {w:.3f} {bar}")


def main(
    eval_split: str = "valid",
    use_valid_warmup: bool = True,
    rolling_window: int | None = None,
    llr_clip: float | None = None,
    top_n: int = 50,
    exp_dir: Path = EXP_DIR,
    eval_burnin: int | None = None,
) -> None:
    print(f"Eval split: {eval_split}")
    print(f"Use valid warm-up for test: {use_valid_warmup and eval_split == 'test'}")

    merged = _load_predictions(eval_split, use_valid_warmup, exp_dir)
    returns, ind_panel = _load_returns_and_industry()
    ranked = _attach_industry_and_forward_return(merged, returns, ind_panel)

    eval_dates = set(pd.to_datetime(ranked.loc[ranked["split"] == eval_split, "time"].unique()))
    print(f"Eval dates after forward-return alignment: {len(eval_dates)}")

    cfg = BayesBlenderConfig(models=tuple(MODELS))
    if rolling_window is not None:
        cfg.rolling_window = int(rolling_window)
    if llr_clip is not None:
        cfg.llr_clip = float(llr_clip)

    effective_burnin = cfg.rolling_window if eval_burnin is None else max(cfg.rolling_window, int(eval_burnin))
    print(
        "Bayes config: "
        f"rolling={cfg.rolling_window}, eval_burnin={eval_burnin}, "
        f"effective_burnin={effective_burnin}, clip={cfg.llr_clip}, "
        f"alpha_strong={cfg.laplace_alpha_strong}, alpha_other={cfg.laplace_alpha_other}"
    )

    n_dates_total = ranked["time"].nunique()
    if n_dates_total <= cfg.rolling_window:
        raise ValueError(
            f"Not enough dates ({n_dates_total}) for rolling_window={cfg.rolling_window}. "
            "Use a shorter --rolling-window or load a warm-up split."
        )
    if n_dates_total <= effective_burnin:
        raise ValueError(
            f"Not enough dates ({n_dates_total}) for effective_burnin={effective_burnin}. "
            "Use a smaller --eval-burnin or load more warm-up history."
        )

    print("\nRunning rolling Bayes ...")
    all_scores = build_bayes_scores(
        ranked,
        MODELS,
        config=cfg,
        eval_dates=eval_dates,
        eval_burnin=eval_burnin,
    )
    if all_scores.empty:
        raise ValueError(
            "No Bayes scores produced. Check rolling_window, eval_split, and warm-up history."
        )

    all_scores["stock_id"] = all_scores["stock_id"].astype(str).str.strip().str.zfill(6)
    all_scores["split"] = eval_split
    print(
        f"Scores: {len(all_scores):,} rows, "
        f"{all_scores['time'].nunique()} eval dates"
    )

    _print_industry_exposure(all_scores, ind_panel)

    print("\nBacktesting ...")
    result = backtest_score(
        all_scores.rename(columns={"bayes_score": "_bayes"}),
        returns,
        "_bayes",
        top_n,
    )
    print(f"  Sharpe:  {result['sharpe_ratio']:.4f}")
    print(f"  NAV:     {result['final_nav']:.4f}")
    print(f"  MaxDD:   {result['max_drawdown']:.4f}")
    print(f"  Turnover:{result['mean_turnover']:.4f}")

    out_path = Path(f"reports/strategy_v1/bayes_scores_{eval_split}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_scores.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-split", default="valid", choices=("valid", "test"))
    parser.add_argument(
        "--no-valid-warmup",
        action="store_true",
        help="In test mode, do not prepend valid predictions as rolling calibration history.",
    )
    parser.add_argument("--rolling-window", type=int, default=None)
    # --tau removed: applied post-sum equally to all stocks, no-op for ranking
    parser.add_argument("--llr-clip", type=float, default=None)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument(
        "--eval-burnin",
        type=int,
        default=None,
        help=(
            "Optional common evaluation burn-in. For fair window comparison, "
            "set --eval-burnin 252 so rolling 126/168/252 are evaluated on the same dates."
        ),
    )
    parser.add_argument("--exp-dir", type=Path, default=EXP_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        eval_split=args.eval_split,
        use_valid_warmup=not args.no_valid_warmup,
        rolling_window=args.rolling_window,
        llr_clip=args.llr_clip,
        top_n=args.top_n,
        exp_dir=args.exp_dir,
        eval_burnin=args.eval_burnin,
    )
