from __future__ import annotations

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
from src.experiment.returns import align_predictions_to_returns
from scripts.experiment.run_walkforward import WINDOWS, OUTPUT_DIR, _window_config


MODELS = ["lightgbm", "xgboost", "dlinear", "gated_dwtcn"]
META_COLS = ["industry_sw", "list_date", "ret_daily", "1d_next_raw"]


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"]).dt.normalize()
    out["stock_id"] = out["stock_id"].astype(str).str.strip().str.zfill(6)
    return out


def _meta_from_split(split_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["time", "stock_id"] + [c for c in META_COLS if c in split_df.columns]
    meta = split_df[cols].copy()
    meta = normalize_keys(meta)
    meta = meta.drop_duplicates(["time", "stock_id"], keep="last")
    return meta


def load_prediction_file(fold_dir: Path, split: str, model: str) -> pd.DataFrame:
    path = fold_dir / f"predictions_{split}_{model}.parquet"
    if not path.exists() and split == "test":
        path = fold_dir / f"predictions_{model}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    df = pd.read_parquet(path)
    df = normalize_keys(df)

    required = {"time", "stock_id", "y_pred", "y_true"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    keep = ["time", "stock_id", "y_true", "y_pred"] + [c for c in META_COLS if c in df.columns]
    return df[keep].rename(columns={"y_pred": model})


def load_wide_or_build(
    fold_dir: Path,
    split: str,
    meta_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    wide_path = fold_dir / f"predictions_{split}_wide.parquet"
    if wide_path.exists():
        wide = normalize_keys(pd.read_parquet(wide_path))
        if meta_df is not None:
            meta_df = normalize_keys(meta_df)
            missing_meta = [c for c in META_COLS if c in meta_df.columns and c not in wide.columns]
            if missing_meta:
                wide = wide.merge(
                    meta_df[["time", "stock_id"] + missing_meta],
                    on=["time", "stock_id"],
                    how="left",
                    validate="one_to_one",
                )
        return wide

    parts = [load_prediction_file(fold_dir, split, m) for m in MODELS]

    base = parts[0]
    for part in parts[1:]:
        model_col = [c for c in part.columns if c in MODELS][0]
        base = base.merge(
            part[["time", "stock_id", model_col]],
            on=["time", "stock_id"],
            how="inner",
            validate="one_to_one",
        )

    if meta_df is not None:
        meta_df = normalize_keys(meta_df)
        missing_meta = [c for c in META_COLS if c in meta_df.columns and c not in base.columns]
        if missing_meta:
            base = base.merge(
                meta_df[["time", "stock_id"] + missing_meta],
                on=["time", "stock_id"],
                how="left",
                validate="one_to_one",
            )

    return base.sort_values(["time", "stock_id"]).reset_index(drop=True)


def add_model_ranks_and_H(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-model daily rank_pct columns and forward-return H.

    H = top 5% of the tradeable 1-period forward return.
    Prefers 1d_next_raw (open-to-open: t+1 open buy, t+2 open sell) when
    available — it is already the forward return at the signal date and
    needs no date-shift.  Falls back to ret_daily (close-to-close) with a
    1-day forward shift for backward compatibility.
    """
    out = normalize_keys(df)

    if "industry_sw" not in out.columns:
        raise ValueError("Bayes requires industry_sw column.")

    for model in MODELS:
        if model not in out.columns:
            raise ValueError(f"Missing model prediction column: {model}")
        out[f"{model}_r"] = out.groupby("time")[model].rank(pct=True)

    use_open_to_open = "1d_next_raw" in out.columns
    if use_open_to_open:
        h_source = "1d_next_raw"
        out[h_source] = pd.to_numeric(out[h_source], errors="coerce")
        out["H"] = (
            out.groupby("time")[h_source]
            .rank(pct=True)
            .ge(0.95)
            .astype(int)
        )
    else:
        # Fallback: close-to-close ret_daily, forward-shifted by 1 day.
        h_source = "ret_daily" if "ret_daily" in out.columns else "y_true"
        out[h_source] = pd.to_numeric(out[h_source], errors="coerce")
        out["time_dt"] = pd.to_datetime(out["time"])
        dates = sorted(out["time_dt"].drop_duplicates())
        next_date = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
        out["return_date"] = out["time_dt"].map(next_date)
        fwd = (out[["time_dt", "stock_id", h_source]]
               .rename(columns={"time_dt": "return_date", h_source: "fwd_ret"}))
        out = out.merge(fwd, on=["return_date", "stock_id"], how="inner")
        out["fwd_ret"] = pd.to_numeric(out["fwd_ret"], errors="coerce")
        out["H"] = (
            out.groupby("time")["fwd_ret"]
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


def evaluate_score(
    scores: pd.DataFrame,
    returns_df: pd.DataFrame,
    top_n: int,
    score_col: str = "bayes_score",
) -> dict[str, Any]:
    pred = scores[["time", "stock_id", score_col]].rename(columns={score_col: "y_pred"}).dropna()
    aligned = align_predictions_to_returns(
        pred_df=pred,
        returns_df=returns_df,
        date_col="time",
        stock_col="stock_id",
        pred_col="y_pred",
        return_col="return_1d",
    )
    pfolio = PortfolioConfig(strategy="top_n", top_n=top_n, pred_col="y_pred", stock_col="stock_id")
    return run_backtest(
        pred_df=aligned,
        returns_df=returns_df,
        portfolio_config=pfolio,
        return_col="return_1d",
        date_col="time",
        stock_col="stock_id",
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


def run_fold_bayes(label: str, win: dict, walkforward_dir: Path,
                   rolling_window: int = 63, half_life: int | None = None,
                   use_gate: bool = False, reward_window: int = 1) -> tuple[pd.DataFrame, dict[str, Any]]:
    fold_dir = walkforward_dir / label
    cfg = _window_config(win, label)
    data = prepare_experiment_data(cfg)

    valid = load_wide_or_build(fold_dir, "valid", meta_df=_meta_from_split(data.valid_df))
    test = load_wide_or_build(fold_dir, "test", meta_df=_meta_from_split(data.test_df))

    merged = pd.concat([valid, test], ignore_index=True)
    merged = merged.sort_values(["time", "stock_id"]).reset_index(drop=True)
    merged_ranked = add_model_ranks_and_H(merged)

    test_dates = set(pd.to_datetime(test["time"].unique()))

    bayes_cfg = BayesBlenderConfig(
        models=tuple(MODELS),
        rolling_window=rolling_window,
        half_life=half_life,
        auto_alpha=True,
        auto_clip=True,
        auto_weights=True,
        clip_percentile=85.0,
        use_sector=True,
    )

    if use_gate:
        scores_detail = build_bayes_scores(
            merged_ranked=merged_ranked,
            models=MODELS,
            config=bayes_cfg,
            eval_dates=test_dates,
            eval_burnin=252,
            detailed=True,
        )
        scores_detail = normalize_keys(scores_detail)
        scores, gate_hist = apply_online_gate(
            scores_detail=scores_detail,
            returns_df=data.returns_df,
            models=MODELS,
            top_n=cfg.top_n,
            gate_burnin=63,
            reward_window=reward_window,
            return_col="return_1d",
        )
        gate_hist.to_csv(fold_dir / f"bayes_gate_history_{_suffix(rw=rolling_window, hl=half_life, gate=True, rw_win=reward_window)}.csv", index=False)
    else:
        scores = build_bayes_scores(
            merged_ranked=merged_ranked,
            models=MODELS,
            config=bayes_cfg,
            eval_dates=test_dates,
            eval_burnin=252,
        )

    scores = normalize_keys(scores)
    scores["fold"] = label

    # Attach diagnostics/meta for later real backtest and analysis.
    meta_cols = ["time", "stock_id", "y_true", "industry_sw"]
    for c in ["ret_daily", "list_date"]:
        if c in merged.columns:
            meta_cols.append(c)

    meta = merged[meta_cols].drop_duplicates(["time", "stock_id"])
    meta = normalize_keys(meta)
    scores = scores.merge(meta, on=["time", "stock_id"], how="left")

    out_path = fold_dir / f"bayes_scores_test_{_suffix(rw=rolling_window, hl=half_life, gate=use_gate, rw_win=reward_window)}.parquet"
    scores.to_parquet(out_path, index=False)
    print(
        f"Saved {label} Bayes scores: {out_path} "
        f"rows={len(scores):,}, dates={scores['time'].nunique()}"
    )

    summary = evaluate_score(scores, data.returns_df, top_n=cfg.top_n, score_col="bayes_score")
    return scores, summary


def _suffix(rw: int = 63, hl: int | None = None, gate: bool = False, rw_win: int = 1) -> str:
    base = f"w{rw}" if hl is None else f"w{rw}_hl{hl}"
    if gate:
        base = f"{base}_gate_rw{rw_win}"
    return base


def stitch_bayes(scores_list: list[pd.DataFrame], walkforward_dir: Path, suffix: str = "w63") -> pd.DataFrame:
    all_scores = pd.concat(scores_list, ignore_index=True)
    all_scores = normalize_keys(all_scores)

    all_path = walkforward_dir / f"bayes_scores_oos_all_{suffix}.parquet"
    all_scores.to_parquet(all_path, index=False)
    print(f"Saved: {all_path} rows={len(all_scores):,}")

    fold_order = {"W1": 1, "W2": 2, "W3": 3}
    all_scores["fold_rank"] = all_scores["fold"].map(fold_order).fillna(0).astype(int)

    latest = (
        all_scores.sort_values(["time", "stock_id", "fold_rank"])
        .drop_duplicates(["time", "stock_id"], keep="last")
        .drop(columns=["fold_rank"])
        .reset_index(drop=True)
    )

    latest_path = walkforward_dir / f"bayes_scores_oos_latest_{suffix}.parquet"
    latest.to_parquet(latest_path, index=False)
    print(f"Saved: {latest_path} rows={len(latest):,}, dates={latest['time'].nunique()}")
    print(f"Overlap dropped: {len(all_scores) - len(latest):,}")

    return latest


def main() -> None:
    walkforward_dir = Path(OUTPUT_DIR)
    summary_dir = walkforward_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling-window", type=int, default=63)
    parser.add_argument("--half-life", type=int, default=None)
    parser.add_argument("--gate", action="store_true", default=False)
    parser.add_argument("--reward-window", type=int, default=1)
    args = parser.parse_args()
    rw = args.rolling_window
    hl = args.half_life
    use_gate = args.gate
    reward_window = args.reward_window
    suffix = _suffix(rw=rw, hl=hl, gate=use_gate, rw_win=reward_window)

    mode = "Bayes+Gate" if use_gate else "Bayes"
    print(f"=== Walk-forward {mode} (rolling_window={rw}) ===")
    print(f"walkforward_dir = {walkforward_dir.resolve()}")

    score_frames = []
    records = []

    for i, win in enumerate(WINDOWS):
        label = f"W{i + 1}"
        fold_dir = walkforward_dir / label
        score_path = fold_dir / f"bayes_scores_test_{suffix}.parquet"
        if score_path.exists():
            print(f"\n{label}: SKIP {mode} {suffix} (already done)")
            scores = pd.read_parquet(score_path)
            scores = normalize_keys(scores)
            data = prepare_experiment_data(_window_config(win, label))
            summary = evaluate_score(scores, data.returns_df, top_n=50, score_col="bayes_score")
        else:
            print(f"\nRunning {mode} {suffix} for {label}...")
            scores, summary = run_fold_bayes(label, win, walkforward_dir, rolling_window=rw, half_life=hl, use_gate=use_gate, reward_window=reward_window)
        score_frames.append(scores)

        s = summary_row(summary)
        records.append({
            "fold": label,
            "model": "bayes",
            "test_total_return": s["total_return"],
            "test_sharpe": s["sharpe"],
            "test_nav": s["nav"],
            "test_maxdd": s["maxdd"],
            "test_turnover": s["turnover"],
            "test_hit_rate": s["hit_rate"],
        })

    stitch_bayes(score_frames, walkforward_dir, suffix=suffix)

    bayes_summary = pd.DataFrame(records)
    summary_path = summary_dir / f"bayes_fold_summary_{suffix}.csv"
    bayes_summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    print(f"\n{mode} fold summary:")
    print(bayes_summary)

    if not bayes_summary.empty:
        agg = bayes_summary.agg({
            "test_total_return": ["mean", "median", "min"],
            "test_sharpe": "mean",
            "test_maxdd": "mean",
            "test_turnover": "mean",
        })
        print(f"\n{mode} aggregate:")
        print(agg)

    print("\nDone.")


if __name__ == "__main__":
    main()
