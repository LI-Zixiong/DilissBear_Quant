"""Partition-backed adapter around the frozen live predictor."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.experiment.config import ExperimentConfig
from src.predict.live_predictor import (
    MODEL_FEATURE_COLS,
    MODEL_TYPES,
    generate_live_predictions as _legacy_generate,
)


def _read_tail_predictions(
    exp_dir: Path,
    smooth_window: int,
) -> pd.DataFrame:
    """Read only the last *smooth_window* days of live predictions per model."""
    models = tuple(MODEL_TYPES)
    merged = None
    for model in models:
        path = exp_dir / f"predictions_live_{model}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Cannot checkpoint without {path}")
        all_dates = pd.read_parquet(path, columns=["time"])
        all_dates["time"] = pd.to_datetime(all_dates["time"])
        unique_dates = sorted(all_dates["time"].unique())
        cutoff = unique_dates[max(0, len(unique_dates) - smooth_window - 1)]
        part = pd.read_parquet(path, columns=["time", "stock_id", "y_pred"])
        part["time"] = pd.to_datetime(part["time"])
        part = part[part["time"] >= pd.Timestamp(cutoff)]
        part["stock_id"] = part["stock_id"].astype(str).str.strip().str.zfill(6)
        part = part.rename(columns={"y_pred": model})
        merged = part if merged is None else merged.merge(
            part, on=["time", "stock_id"], how="inner",
        )
    if merged is None or merged.empty:
        raise ValueError("No prediction rows available for checkpoint")
    return merged.sort_values(["time", "stock_id"]).reset_index(drop=True)


def export_live_selection_checkpoint(
    exp_dir: str | Path,
    smooth_window: int = 10,
    top_n: int = 50,
) -> Path:
    """Atomically publish the latest three-model EW ranking and Top-N."""
    from src.backtest.ensemble_utils import smooth_predictions

    exp_dir = Path(exp_dir)
    models = tuple(MODEL_TYPES)
    merged = _read_tail_predictions(exp_dir, smooth_window)

    scored = smooth_predictions(merged, models, smooth_window)
    for model in models:
        scored[f"{model}_rank"] = scored.groupby("time")[model].rank(pct=True)
    scored["ew_score"] = scored[
        [f"{model}_rank" for model in models]
    ].mean(axis=1)
    scored["rank"] = scored.groupby("time")["ew_score"].rank(
        method="first", ascending=False,
    )
    latest = pd.Timestamp(scored["time"].max())
    selection = scored[
        (scored["time"] == latest) & (scored["rank"] <= top_n)
    ].sort_values("rank").reset_index(drop=True)
    if len(selection) != top_n:
        raise ValueError(f"Checkpoint has {len(selection)} rows, expected {top_n}")

    output = exp_dir / "live_selection_checkpoint.parquet"
    tmp = output.with_name(f"{output.stem}.tmp{output.suffix}")
    selection.to_parquet(tmp, index=False)
    tmp.replace(output)
    meta = {
        "time": latest.date().isoformat(),
        "top_n": top_n,
        "smooth_window": smooth_window,
        "models": list(models),
        "rows": len(selection),
    }
    meta_path = exp_dir / "live_selection_checkpoint.json"
    meta_tmp = meta_path.with_suffix(".json.tmp")
    meta_tmp.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    meta_tmp.replace(meta_path)
    return output


def generate_live_predictions(
    exp_dir: str | Path,
    live_store_root: str | Path = "dataset/cache/live",
    smooth_window: int = 10,
    track_start: pd.Timestamp = pd.Timestamp("2026-07-20"),
    config: ExperimentConfig | None = None,
) -> Path:
    """Read a projected live tail, run frozen models, then publish checkpoint."""
    from src.pipeline.live_store import LiveStore

    t_total = time.perf_counter()
    config = config or ExperimentConfig()
    store = LiveStore(live_store_root)
    read_start = track_start - pd.Timedelta(days=120)
    columns = {
        config.date_col, config.stock_col, *config.feature_cols,
        *config.meta_cols, config.target_col,
        "industry_sw", "list_date", "ret_daily", "1d_next_raw",
    }
    for values in MODEL_FEATURE_COLS.values():
        columns.update(values)
    raw = store.read(
        "factor_panel",
        columns=sorted(columns),
        start=read_start,
    )
    if raw.empty:
        raise ValueError(f"No committed factor rows on/after {read_start.date()}")
    print(
        f"Prediction input: generation {store.pointer().generation}, "
        f"{raw['time'].nunique()} dates, {len(raw):,} rows"
    )
    print(f"  [store read] {time.perf_counter() - t_total:.1f}s")

    _legacy_generate(
        exp_dir=exp_dir,
        data_path=None,
        smooth_window=smooth_window,
        track_start=track_start,
        config=config,
        _live_factor_df=raw,
    )
    print(f"  [predict done] {time.perf_counter() - t_total:.1f}s")

    checkpoint = export_live_selection_checkpoint(
        exp_dir=exp_dir,
        smooth_window=smooth_window,
        top_n=50,
    )
    print(f"  [checkpoint] {time.perf_counter() - t_total:.1f}s total")
    print(f"Prediction checkpoint ready: {checkpoint}")
    return checkpoint
