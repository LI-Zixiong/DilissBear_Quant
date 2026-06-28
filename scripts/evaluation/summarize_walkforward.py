"""Walk-forward V1 summary / ensemble evaluation utility.

Purpose
-------
This script reads outputs produced by:

    python -m scripts.experiment.run_walkforward

and then creates fold-level summaries, n-scan summaries, per-fold wide
prediction panels, Equal-Rank / Rank-Ridge ensemble evaluations, and stitched
OOS panels for later Bayes / real-account backtests.

Recommended location in project:

    scripts/evaluation/summarize_walkforward.py

Run:

    python -m scripts.evaluation.summarize_walkforward

Conventions
-----------
1. valid is used for calibration / fitting ensemble weights.
2. test is report-only.
3. stitched_oos_all is diagnostic only.
4. stitched_oos_latest is the only stitched panel used for continuous OOS work.
5. repeated OOS dates keep the latest eligible fold: W3 > W2 > W1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.ensemble_methods import equal_weight, rank_ridge
from src.experiment.data import prepare_experiment_data

# Reuse the exact window definitions and config builder from the training script.
# This prevents evaluation from silently drifting away from walk-forward training.
from scripts.experiment.run_walkforward import WINDOWS, OUTPUT_DIR, _window_config


DEFAULT_MODELS = ("lightgbm", "xgboost", "dlinear", "gated_dwtcn")
META_COLS = ("industry_sw", "list_date", "ret_daily", "1d_next_raw")


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize time and stock_id keys for safe joins."""
    out = df.copy()
    if "time" not in out.columns or "stock_id" not in out.columns:
        raise ValueError("DataFrame must contain 'time' and 'stock_id'.")
    out["time"] = pd.to_datetime(out["time"], errors="raise").dt.normalize()
    out["stock_id"] = out["stock_id"].astype(str).str.strip().str.zfill(6)
    return out


def _metric(summary: dict[str, Any], key: str) -> float:
    return float(summary.get(key, np.nan))


def summary_row(summary: dict[str, Any]) -> dict[str, float]:
    """Standardize backtest summary fields."""
    return {
        "total_return": _metric(summary, "total_return"),
        "sharpe": _metric(summary, "sharpe_ratio"),
        "annualized_return": _metric(summary, "annualized_return"),
        "nav": _metric(summary, "final_nav"),
        "maxdd": _metric(summary, "max_drawdown"),
        "turnover": _metric(summary, "mean_turnover"),
        "hit_rate": _metric(summary, "hit_rate"),
    }


def _meta_from_split(split_df: pd.DataFrame) -> pd.DataFrame:
    """Recover metadata from the prepared split panel if prediction files lack it."""
    cols = ["time", "stock_id"] + [c for c in META_COLS if c in split_df.columns]
    meta = split_df[cols].copy()
    meta = normalize_keys(meta)
    meta = meta.drop_duplicates(["time", "stock_id"], keep="last")
    return meta


def _fold_label(i: int) -> str:
    return f"W{i + 1}"


# ---------------------------------------------------------------------------
# Prediction loading and wide panel creation
# ---------------------------------------------------------------------------


def _prediction_path(fold_dir: Path, split: str, model: str) -> Path:
    path = fold_dir / f"predictions_{split}_{model}.parquet"
    if path.exists():
        return path
    # Backward compatibility: test predictions may also be saved as predictions_<model>.parquet.
    legacy = fold_dir / f"predictions_{model}.parquet"
    if split == "test" and legacy.exists():
        return legacy
    raise FileNotFoundError(
        f"Missing prediction for split={split!r}, model={model!r}. "
        f"Expected {path}" + (f" or {legacy}" if split == "test" else "")
    )


def load_prediction_file(fold_dir: Path, split: str, model: str) -> pd.DataFrame:
    """Load one model prediction file and rename y_pred to model name."""
    path = _prediction_path(fold_dir, split, model)
    df = pd.read_parquet(path)
    df = normalize_keys(df)

    required = {"time", "stock_id", "y_pred", "y_true"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    keep = ["time", "stock_id", "y_true", "y_pred"]
    keep += [c for c in META_COLS if c in df.columns]
    out = df[keep].copy().rename(columns={"y_pred": model})
    return out


def load_split_wide(
    fold_dir: Path,
    split: str,
    models: Iterable[str] = DEFAULT_MODELS,
    meta_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load all model prediction files for one fold/split and inner-join into wide form.

    Output columns:
        time, stock_id, y_true, optional metadata, lightgbm, xgboost, dlinear, gated_dwtcn
    """
    models = list(models)
    if not models:
        raise ValueError("models cannot be empty")

    parts = [load_prediction_file(fold_dir, split, m) for m in models]
    base = parts[0]

    for part in parts[1:]:
        model_col = [c for c in part.columns if c in models][0]
        base = base.merge(
            part[["time", "stock_id", model_col]],
            on=["time", "stock_id"],
            how="inner",
            validate="one_to_one",
        )

    # If prediction files do not carry metadata, recover it from the prepared split panel.
    if meta_df is not None:
        meta_df = normalize_keys(meta_df)
        present_meta = [c for c in META_COLS if c in base.columns]
        missing_meta = [c for c in META_COLS if c in meta_df.columns and c not in present_meta]
        if missing_meta:
            base = base.merge(
                meta_df[["time", "stock_id"] + missing_meta],
                on=["time", "stock_id"],
                how="left",
                validate="one_to_one",
            )

    cols = ["time", "stock_id", "y_true"]
    cols += [c for c in META_COLS if c in base.columns]
    cols += models

    base = base[cols].sort_values(["time", "stock_id"]).reset_index(drop=True)

    missing_models = [m for m in models if m not in base.columns]
    if missing_models:
        raise ValueError(f"Missing model columns after merge: {missing_models}")

    return base


def save_split_wide_files(walkforward_dir: Path, models: Iterable[str]) -> None:
    """Create predictions_valid_wide.parquet and predictions_test_wide.parquet in each fold."""
    models = tuple(models)
    for i, win in enumerate(WINDOWS):
        label = _fold_label(i)
        fold_dir = walkforward_dir / label
        cfg = _window_config(win, label)
        data = prepare_experiment_data(cfg)
        meta_by_split = {
            "valid": _meta_from_split(data.valid_df),
            "test": _meta_from_split(data.test_df),
        }

        for split in ("valid", "test"):
            out_path = fold_dir / f"predictions_{split}_wide.parquet"
            if out_path.exists():
                print(f"Skip: {out_path} (already exists)")
                continue
            wide = load_split_wide(fold_dir, split, models=models, meta_df=meta_by_split[split])
            wide.to_parquet(out_path, index=False)
            print(
                f"Saved: {out_path}  rows={len(wide):,}, "
                f"dates={wide['time'].nunique()}"
            )


# ---------------------------------------------------------------------------
# Aggregation of existing walk-forward outputs
# ---------------------------------------------------------------------------


def aggregate_model_comparison(walkforward_dir: Path, summary_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for i in range(len(WINDOWS)):
        label = _fold_label(i)
        path = walkforward_dir / label / "model_comparison.csv"
        if not path.exists():
            print(f"WARNING: missing {path}")
            continue
        df = pd.read_csv(path)
        df.insert(0, "fold", label)
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out_path = summary_dir / "fold_model_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return out


def aggregate_n_scan(walkforward_dir: Path, summary_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for i in range(len(WINDOWS)):
        label = _fold_label(i)
        fold_dir = walkforward_dir / label

        for model in ("lightgbm", "xgboost"):
            path = fold_dir / f"n_scan_{model}.csv"
            if not path.exists():
                print(f"WARNING: missing {path}")
                continue

            df = pd.read_csv(path)
            df.insert(0, "model", model)
            df.insert(0, "fold", label)

            if "valid_total_return" in df.columns and not df.empty:
                df["selected"] = False
                df.loc[df["valid_total_return"].idxmax(), "selected"] = True

            rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out_path = summary_dir / "n_scan_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return out


# ---------------------------------------------------------------------------
# Ensemble scores and evaluation
# ---------------------------------------------------------------------------


def add_daily_model_ranks(df: pd.DataFrame, model_cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for m in model_cols:
        if m not in out.columns:
            raise ValueError(f"Missing model prediction column: {m}")
        out[f"{m}_r"] = out.groupby("time")[m].rank(pct=True)
    return out


def make_equal_score(df: pd.DataFrame, model_cols: Iterable[str]) -> pd.DataFrame:
    model_cols = list(model_cols)
    out = add_daily_model_ranks(df, model_cols)
    out["ensemble_equal_rank"] = out[[f"{m}_r" for m in model_cols]].mean(axis=1)
    return out


def make_rank_ridge_score(
    df: pd.DataFrame,
    model_cols: Iterable[str],
    weights: pd.Series,
) -> pd.DataFrame:
    model_cols = list(model_cols)
    out = add_daily_model_ranks(df, model_cols)
    out["ensemble_rank_ridge"] = 0.0
    for m in model_cols:
        if m not in weights.index:
            raise ValueError(f"Rank-Ridge weight missing model: {m}")
        out["ensemble_rank_ridge"] += float(weights[m]) * out[f"{m}_r"]
    return out


def _save_score_file(scored: pd.DataFrame, score_col: str, out_path: Path) -> None:
    keep = ["time", "stock_id", "y_true", score_col]
    keep += [c for c in META_COLS if c in scored.columns]
    out = scored[keep].copy().rename(columns={score_col: "y_pred"})
    out.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")


def evaluate_ensembles(
    walkforward_dir: Path,
    summary_dir: Path,
    models: Iterable[str],
    smooth_window: int = 1,
) -> pd.DataFrame:
    """Evaluate Equal-Rank and Rank-Ridge per fold.

    Rank-Ridge is fitted on each fold's valid split and applied to its test split.
    Test never participates in weight fitting.
    """
    models = list(models)
    records: list[dict[str, Any]] = []

    for i, win in enumerate(WINDOWS):
        label = _fold_label(i)
        fold_dir = walkforward_dir / label
        print(f"\nEvaluating ensembles for {label}...")

        cfg = _window_config(win, label)
        data = prepare_experiment_data(cfg)
        returns_df = data.returns_df

        valid = load_split_wide(fold_dir, "valid", models=models, meta_df=_meta_from_split(data.valid_df))
        test = load_split_wide(fold_dir, "test", models=models, meta_df=_meta_from_split(data.test_df))

        # Existing pure-method summaries.
        ew_valid_summary = equal_weight(
            predictions=valid,
            returns=returns_df,
            model_cols=models,
            smooth_window=smooth_window,
            top_n=cfg.top_n,
        )
        ew_test_summary = equal_weight(
            predictions=test,
            returns=returns_df,
            model_cols=models,
            smooth_window=smooth_window,
            top_n=cfg.top_n,
        )

        rr_valid_summary, rr_weights = rank_ridge(
            predictions=valid,
            returns=returns_df,
            model_cols=models,
            valid_predictions=valid,
            smooth_window=smooth_window,
            top_n=cfg.top_n,
        )
        rr_test_summary, _ = rank_ridge(
            predictions=test,
            returns=returns_df,
            model_cols=models,
            valid_predictions=valid,
            smooth_window=smooth_window,
            top_n=cfg.top_n,
        )

        # Save scored predictions for later diagnostics / real backtest.
        valid_equal = make_equal_score(valid, models)
        test_equal = make_equal_score(test, models)
        valid_rr = make_rank_ridge_score(valid, models, rr_weights)
        test_rr = make_rank_ridge_score(test, models, rr_weights)

        _save_score_file(
            valid_equal,
            "ensemble_equal_rank",
            fold_dir / "predictions_valid_ensemble_equal_rank.parquet",
        )
        _save_score_file(
            test_equal,
            "ensemble_equal_rank",
            fold_dir / "predictions_test_ensemble_equal_rank.parquet",
        )
        _save_score_file(
            valid_rr,
            "ensemble_rank_ridge",
            fold_dir / "predictions_valid_ensemble_rank_ridge.parquet",
        )
        _save_score_file(
            test_rr,
            "ensemble_rank_ridge",
            fold_dir / "predictions_test_ensemble_rank_ridge.parquet",
        )

        ev = summary_row(ew_valid_summary)
        et = summary_row(ew_test_summary)
        records.append({
            "fold": label,
            "ensemble": "equal_rank",
            "smooth_window": smooth_window,
            "valid_total_return": ev["total_return"],
            "valid_sharpe": ev["sharpe"],
            "valid_nav": ev["nav"],
            "valid_maxdd": ev["maxdd"],
            "valid_turnover": ev["turnover"],
            "test_total_return": et["total_return"],
            "test_sharpe": et["sharpe"],
            "test_nav": et["nav"],
            "test_maxdd": et["maxdd"],
            "test_turnover": et["turnover"],
            **{f"weight_{m}": 1.0 / len(models) for m in models},
        })

        rv = summary_row(rr_valid_summary)
        rt = summary_row(rr_test_summary)
        records.append({
            "fold": label,
            "ensemble": "rank_ridge",
            "smooth_window": smooth_window,
            "valid_total_return": rv["total_return"],
            "valid_sharpe": rv["sharpe"],
            "valid_nav": rv["nav"],
            "valid_maxdd": rv["maxdd"],
            "valid_turnover": rv["turnover"],
            "test_total_return": rt["total_return"],
            "test_sharpe": rt["sharpe"],
            "test_nav": rt["nav"],
            "test_maxdd": rt["maxdd"],
            "test_turnover": rt["turnover"],
            **{f"weight_{m}": float(rr_weights[m]) for m in models},
        })

    out = pd.DataFrame(records)
    out_path = summary_dir / "ensemble_fold_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return out


# ---------------------------------------------------------------------------
# Stitched latest panels
# ---------------------------------------------------------------------------


def build_stitched_wide(
    walkforward_dir: Path,
    summary_dir: Path,
    models: Iterable[str],
) -> pd.DataFrame:
    """Build wide model-prediction panel from stitched_oos_latest.parquet."""
    models = list(models)
    path = walkforward_dir / "stitched_oos_latest.parquet"
    if not path.exists():
        print(f"WARNING: missing {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df = normalize_keys(df)

    model_col = "model" if "model" in df.columns else "model_name" if "model_name" in df.columns else None
    if model_col is None:
        raise ValueError("stitched_oos_latest.parquet must contain 'model' or 'model_name'.")

    required = {"time", "stock_id", model_col, "y_pred", "y_true"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    index_cols = ["time", "stock_id", "y_true"]
    index_cols += [c for c in META_COLS if c in df.columns]

    wide = df.pivot_table(
        index=index_cols,
        columns=model_col,
        values="y_pred",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    missing_models = [m for m in models if m not in wide.columns]
    if missing_models:
        print(f"WARNING: stitched wide missing model columns: {missing_models}")

    wide = wide.sort_values(["time", "stock_id"]).reset_index(drop=True)

    out_path = walkforward_dir / "stitched_oos_latest_wide.parquet"
    wide.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}  rows={len(wide):,}, dates={wide['time'].nunique()}")

    coverage = wide.groupby("time").size().rename("n_stocks").reset_index()
    coverage_path = summary_dir / "stitched_oos_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    print(f"Saved: {coverage_path}")

    return wide


def stitch_ensemble_score(
    walkforward_dir: Path,
    ensemble_name: str,
    out_name: str,
) -> pd.DataFrame:
    """Stitch per-fold test ensemble score files with latest-fold-wins rule."""
    frames: list[pd.DataFrame] = []
    for i in range(len(WINDOWS)):
        label = _fold_label(i)
        fold_dir = walkforward_dir / label
        path = fold_dir / f"predictions_test_ensemble_{ensemble_name}.parquet"
        if not path.exists():
            print(f"WARNING: missing {path}")
            continue
        df = pd.read_parquet(path)
        df = normalize_keys(df)
        df["fold"] = label
        df["model_name"] = f"ensemble_{ensemble_name}"
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    all_path = walkforward_dir / f"{out_name}_oos_all.parquet"
    all_df.to_parquet(all_path, index=False)

    fold_order = {"W1": 1, "W2": 2, "W3": 3}
    all_df["fold_rank"] = all_df["fold"].map(fold_order).fillna(0).astype(int)
    latest = (
        all_df.sort_values(["time", "stock_id", "fold_rank"])
        .drop_duplicates(["time", "stock_id"], keep="last")
        .drop(columns=["fold_rank"])
        .reset_index(drop=True)
    )

    latest_path = walkforward_dir / f"{out_name}_oos_latest.parquet"
    latest.to_parquet(latest_path, index=False)
    print(f"Saved: {latest_path}  rows={len(latest):,}, dates={latest['time'].nunique()}")
    print(f"  Overlap dropped for {out_name}: {len(all_df) - len(latest):,}")
    return latest


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(
    summary_dir: Path,
    model_df: pd.DataFrame,
    nscan_df: pd.DataFrame,
    ensemble_df: pd.DataFrame,
    stitched_wide: pd.DataFrame,
) -> None:
    path = summary_dir / "walkforward_summary.md"
    lines: list[str] = []

    lines.append("# Walk-forward V1 Summary")
    lines.append("")
    lines.append("## Generated files")
    lines.append("")
    lines.append("- fold_model_summary.csv")
    lines.append("- n_scan_summary.csv")
    lines.append("- ensemble_fold_summary.csv")
    lines.append("- stitched_oos_coverage.csv")
    lines.append("- stitched_oos_latest_wide.parquet")
    lines.append("- ensemble_equal_rank_oos_latest.parquet")
    lines.append("- ensemble_rank_ridge_oos_latest.parquet")
    lines.append("")

    if not ensemble_df.empty:
        lines.append("## Ensemble fold results")
        lines.append("")
        cols = [
            "fold", "ensemble", "smooth_window",
            "valid_total_return", "valid_sharpe",
            "test_total_return", "test_sharpe",
            "test_maxdd", "test_turnover",
        ]
        cols = [c for c in cols if c in ensemble_df.columns]
        lines.append(ensemble_df[cols].to_markdown(index=False))
        lines.append("")

        agg = ensemble_df.groupby("ensemble").agg(
            mean_test_return=("test_total_return", "mean"),
            median_test_return=("test_total_return", "median"),
            worst_test_return=("test_total_return", "min"),
            positive_folds=("test_total_return", lambda x: int((x > 0).sum())),
            mean_test_sharpe=("test_sharpe", "mean"),
            mean_test_maxdd=("test_maxdd", "mean"),
            mean_test_turnover=("test_turnover", "mean"),
        ).reset_index()

        lines.append("## Ensemble aggregate")
        lines.append("")
        lines.append(agg.to_markdown(index=False))
        lines.append("")

    if not nscan_df.empty:
        lines.append("## Selected n_tree")
        lines.append("")
        if "selected" in nscan_df.columns:
            selected = nscan_df[nscan_df["selected"] == True].copy()
        else:
            idx = nscan_df.groupby(["fold", "model"])["valid_total_return"].idxmax()
            selected = nscan_df.loc[idx].copy()

        show_cols = [c for c in [
            "fold", "model", "n_tree",
            "valid_total_return", "valid_sharpe",
            "test_total_return", "test_sharpe",
        ] if c in selected.columns]
        if show_cols:
            lines.append(selected[show_cols].to_markdown(index=False))
        lines.append("")

    if not model_df.empty:
        lines.append("## Fold model rows")
        lines.append("")
        lines.append(f"- rows: {len(model_df):,}")
        if "model" in model_df.columns:
            lines.append(f"- models: {', '.join(sorted(model_df['model'].astype(str).unique()))}")
        lines.append("")

    if not stitched_wide.empty:
        lines.append("## Stitched OOS latest wide")
        lines.append("")
        lines.append(f"- rows: {len(stitched_wide):,}")
        lines.append(f"- dates: {stitched_wide['time'].nunique():,}")
        lines.append(f"- start: {stitched_wide['time'].min()}")
        lines.append(f"- end: {stitched_wide['time'].max()}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Walk-forward V1 outputs.")
    parser.add_argument(
        "--walkforward-dir",
        type=Path,
        default=Path(OUTPUT_DIR),
        help="Walk-forward output directory. Default imported from run_walkforward."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model names."
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Smoothing window passed to ensemble_methods. Current default is 1."
    )
    parser.add_argument(
        "--skip-ensembles",
        action="store_true",
        help="Only build summaries / wide panels; skip Equal and Rank-Ridge evaluation."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    walkforward_dir = Path(args.walkforward_dir)
    summary_dir = walkforward_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    if not models:
        raise ValueError("No models provided.")

    print("=== Summarize Walk-forward V1 ===")
    print(f"walkforward_dir = {walkforward_dir.resolve()}")
    print(f"summary_dir     = {summary_dir.resolve()}")
    print(f"models          = {models}")

    save_split_wide_files(walkforward_dir, models=models)
    model_df = aggregate_model_comparison(walkforward_dir, summary_dir)
    nscan_df = aggregate_n_scan(walkforward_dir, summary_dir)

    if args.skip_ensembles:
        ensemble_df = pd.DataFrame()
    else:
        ensemble_df = evaluate_ensembles(
            walkforward_dir=walkforward_dir,
            summary_dir=summary_dir,
            models=models,
            smooth_window=int(args.smooth_window),
        )
        stitch_ensemble_score(walkforward_dir, "equal_rank", "ensemble_equal_rank")
        stitch_ensemble_score(walkforward_dir, "rank_ridge", "ensemble_rank_ridge")

    stitched_wide = build_stitched_wide(walkforward_dir, summary_dir, models=models)

    write_report(
        summary_dir=summary_dir,
        model_df=model_df,
        nscan_df=nscan_df,
        ensemble_df=ensemble_df,
        stitched_wide=stitched_wide,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
