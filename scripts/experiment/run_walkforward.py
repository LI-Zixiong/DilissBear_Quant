"""
Walk-forward experiment runner.

3 windows, 1-year step, half-open intervals with purge.
Trains & evaluates each window independently, then stitches OOS predictions.

Usage:
    python -m scripts.experiment.run_walkforward
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.experiment.config import ExperimentConfig
from src.experiment.runner import run_experiment

# ── Walk-forward windows (half-open [start,end)) ────────────────────────
SMOKE = "--smoke" in sys.argv

WINDOWS = [
    {  # W1
        "train": ("2015-06-01", "2021-06-01"),
        "valid": ("2021-06-01", "2022-12-01"),
        "test":  ("2022-12-01", "2024-06-01"),
    },
    {  # W2
        "train": ("2016-06-01", "2022-06-01"),
        "valid": ("2022-06-01", "2023-12-01"),
        "test":  ("2023-12-01", "2025-06-01"),
    },
    {  # W3
        "train": ("2017-06-01", "2023-06-01"),
        "valid": ("2023-06-01", "2024-12-01"),
        "test":  ("2024-12-01", "2026-06-01"),
    },
]

OUTPUT_DIR = Path(ExperimentConfig().output_dir) / "walkforward"


def _boundaries(win: dict) -> tuple[str, str, str, str, str, str]:
    return (
        win["train"][0], win["train"][1],
        win["valid"][0], win["valid"][1],
        win["test"][0],  win["test"][1],
    )


def _window_config(win: dict, label: str) -> ExperimentConfig:
    """Build config for one window. Reads production defaults, overrides boundaries + output."""
    cfg = ExperimentConfig(
        data_path="dataset/processed/factor_panel_1500_54_ind.parquet",
        date_col="time",
        stock_col="stock_id",
        target_col="5d_next_raw",
        return_col="return_1d",
        backtest_return_mode="column",
        backtest_return_source="ret_daily",
        feature_cols=(
            "F001SIZE", "F002SIZENL", "F003LIQUIDITY", "F004BETA",
            "F005RESVOL", "F006MOMENTUM", "F007LTREV", "F008STREV",
            "F009LEVERAGE", "F010VALUE", "F011EARNYLD", "F012GROWTH",
            "F013REV5", "F014MOM120_20", "F015VOLREV", "F016MAXRET",
            "F017IVOL", "F018AMIHUD", "F019COSTDEV", "F020BP",
            "F021CFP", "F022GPTA", "F023ACCRUAL", "F024ASSETGR",
            "F025GAP", "F026KLEN", "F027KUP", "F028KLOW", "F029KSFT",
            "F030RSV20", "F031RSV60", "F032RANGEZ20", "F033GAPREV5",
            "F034HIGHDEV20", "F035LOWDEV20", "F036VOLSHOCK5", "F037VOLSHOCK20",
            "F038TURNZ20", "F039VSTD20", "F040PVCORR20", "F041RETVOLCORR20",
            "F042AMTCORR20", "F043SLOPE20", "F044RSQR20", "F045RESI20",
            "F046LIMITUP20", "F047LIMITDN20", "F048LIMITSTREAKUP",
            "F049ROE", "F050ROA", "F051GPM", "F052CFOA",
            "F053RD_INTENSITY", "F054RECEIVABLE_RATIO",
            "F055IND",
        ),
        meta_cols=("industry_sw", "list_date", "ret_daily"),
        model_names=("lightgbm", "xgboost", "dlinear", "gated_dwtcn"),
        model_feature_cols={
            "lightgbm": (
                "F002SIZENL", "F003LIQUIDITY", "F004BETA",
                "F005RESVOL", "F007LTREV", "F008STREV",
                "F010VALUE", "F011EARNYLD", "F012GROWTH",
                "F013REV5", "F015VOLREV", "F016MAXRET",
                "F017IVOL", "F018AMIHUD", "F019COSTDEV", "F020BP",
                "F021CFP", "F023ACCRUAL", "F024ASSETGR",
                "F025GAP", "F026KLEN", "F027KUP", "F028KLOW", "F029KSFT",
                "F030RSV20", "F031RSV60", "F032RANGEZ20", "F033GAPREV5",
                "F035LOWDEV20", "F036VOLSHOCK5", "F037VOLSHOCK20",
                "F038TURNZ20", "F039VSTD20", "F040PVCORR20", "F041RETVOLCORR20",
                "F042AMTCORR20", "F043SLOPE20", "F044RSQR20", "F045RESI20",
                "F051GPM", "F052CFOA", "F054RECEIVABLE_RATIO",
                "F055IND",
            ),
            "xgboost": (
                "F002SIZENL", "F003LIQUIDITY", "F004BETA",
                "F005RESVOL", "F007LTREV", "F008STREV",
                "F010VALUE", "F011EARNYLD", "F012GROWTH",
                "F013REV5", "F015VOLREV", "F016MAXRET",
                "F017IVOL", "F018AMIHUD", "F019COSTDEV", "F020BP",
                "F021CFP", "F023ACCRUAL", "F024ASSETGR",
                "F025GAP", "F026KLEN", "F027KUP", "F028KLOW", "F029KSFT",
                "F030RSV20", "F031RSV60", "F032RANGEZ20", "F033GAPREV5",
                "F035LOWDEV20", "F036VOLSHOCK5", "F037VOLSHOCK20",
                "F038TURNZ20", "F039VSTD20", "F040PVCORR20", "F041RETVOLCORR20",
                "F042AMTCORR20", "F043SLOPE20", "F044RSQR20", "F045RESI20",
                "F046LIMITUP20", "F047LIMITDN20", "F048LIMITSTREAKUP",
                "F051GPM", "F052CFOA", "F054RECEIVABLE_RATIO",
                "F055IND",
            ),
            "dlinear": (
                "F039VSTD20", "F041RETVOLCORR20",
                "F016MAXRET",
                "F017IVOL", "F018AMIHUD", "F002SIZENL",
                "F026KLEN", "F028KLOW", "F035LOWDEV20",
                "F008STREV", "F042AMTCORR20",
                "F031RSV60",
                "F005RESVOL", "F003LIQUIDITY", "F025GAP",
                "F027KUP", "F010VALUE",
                "F033GAPREV5",
                "F021CFP", "F020BP", "F023ACCRUAL",
                "F030RSV20", "F015VOLREV",
                "F007LTREV", "F052CFOA",
            ),
            "gated_dwtcn": (
                "F002SIZENL", "F003LIQUIDITY", "F004BETA",
                "F005RESVOL", "F007LTREV", "F008STREV",
                "F010VALUE", "F011EARNYLD", "F012GROWTH",
                "F013REV5", "F015VOLREV", "F016MAXRET",
                "F017IVOL", "F018AMIHUD", "F019COSTDEV", "F020BP",
                "F021CFP", "F023ACCRUAL", "F024ASSETGR",
                "F025GAP", "F026KLEN", "F027KUP", "F028KLOW", "F029KSFT",
                "F030RSV20", "F031RSV60", "F032RANGEZ20", "F033GAPREV5",
                "F035LOWDEV20", "F036VOLSHOCK5", "F037VOLSHOCK20",
                "F038TURNZ20", "F039VSTD20", "F040PVCORR20", "F041RETVOLCORR20",
                "F042AMTCORR20", "F043SLOPE20", "F044RSQR20", "F045RESI20",
                "F046LIMITUP20", "F047LIMITDN20", "F048LIMITSTREAKUP",
                "F051GPM", "F052CFOA", "F054RECEIVABLE_RATIO",
                "F055IND",
            ),
        },
        model_feature_signs={},
        seed=1713627,
        top_n=50,
        periods_per_year=252,
        seq_len=20,
        torch_epochs=10,
        torch_patience=2,
        torch_batch_size=4096,
        torch_learning_rate=7.5e-4,
        torch_weight_decay=0.0,
        torch_device="auto",
        predict_batch_size=8192,
        model_params={
            "lightgbm": {
                "n_estimators": 4000,
                "learning_rate": 0.01,
                "num_leaves": 31,
                "early_stopping_rounds": None,
                "feature_fraction": 0.8,
            },
            "xgboost": {
                "n_estimators": 4000,
                "max_depth": 5,
                "learning_rate": 0.01,
                "subsample": 0.8,
                "colsample_bytree": 0.12,
                "early_stopping_rounds": None,
            },
            "dlinear": {
                "seq_len": 20, "epochs": 80,
                "lr": 2e-4, "wd": 5e-5, "patience": 10,
            },
            "gated_dwtcn": {
                "seq_len": 20, "epochs": 25,
                "lr": 2e-4, "wd": 0.0, "patience": 5,
                "kernel_size": 3, "dilations": (1, 2, 4),
                "hidden_dim": 32, "gate_rank": 8,
            },
        },
        date_boundaries=_boundaries(win),
        purge_days=6,
        target_horizon_days=5,
        output_dir=str(OUTPUT_DIR / label),
        report_path=str(OUTPUT_DIR / label / "report.md"),
    )
    if SMOKE:
        cfg.model_params["lightgbm"]["n_estimators"] = 1
        cfg.model_params["xgboost"]["n_estimators"] = 1
        cfg.model_params["dlinear"]["epochs"] = 1
        cfg.model_params["gated_dwtcn"]["epochs"] = 1
    return cfg


def _stitch_oos(fold_specs: list[dict], output_dir: Path):
    """Stitch test predictions from all folds.

    fold_specs: [{"label": "W1", "dir": Path, "models": ["lightgbm", ...]}, ...]

    Produces:
        stitched_oos_all.parquet — all fold predictions, allows duplicate dates
        stitched_oos_latest.parquet — overlapping dates keep latest fold (W3 > W2 > W1)
    """
    MODELS = ["lightgbm", "xgboost", "dlinear", "gated_dwtcn"]
    all_frames = []

    for spec in fold_specs:
        label = spec["label"]
        exp_dir = spec["dir"]
        for m in MODELS:
            p = exp_dir / f"predictions_test_{m}.parquet"
            if not p.exists():
                p = exp_dir / f"predictions_{m}.parquet"
            if not p.exists():
                print(f"  WARNING: {p} not found — skipping {label}/{m}")
                continue
            df = pd.read_parquet(p)
            df["fold"] = label
            df["model"] = m
            all_frames.append(df)

    if not all_frames:
        print("No prediction files found to stitch.")
        return

    all_df = pd.concat(all_frames, ignore_index=True)
    all_df["time"] = pd.to_datetime(all_df["time"])
    all_df["stock_id"] = all_df["stock_id"].astype(str).str.strip().str.zfill(6)
    all_out = output_dir / "stitched_oos_all.parquet"
    all_df.to_parquet(all_out, index=False)
    print(f"  Stitched all: {all_out}  ({len(all_df):,} rows)")

    # Latest fold per (time, stock_id, model)
    fold_order = {f"W{i+1}": i for i in range(len(fold_specs))}
    all_df["fold_rank"] = all_df["fold"].map(fold_order)
    latest = all_df.sort_values(["time", "stock_id", "model", "fold_rank"]) \
        .drop_duplicates(["time", "stock_id", "model"], keep="last") \
        .drop(columns=["fold_rank"]) \
        .reset_index(drop=True)

    latest_out = output_dir / "stitched_oos_latest.parquet"
    latest.to_parquet(latest_out, index=False)
    print(f"  Stitched latest: {latest_out}  ({len(latest):,} rows)")
    print(f"  Overlap dropped: {len(all_df) - len(latest):,} rows")

    # Wide format for ensemble / Bayes
    wide = latest.pivot_table(
        index=["time", "stock_id"], columns="model", values="y_pred",
    ).reset_index()
    wide.columns.name = None
    wide_path = output_dir / "stitched_oos_latest_wide.parquet"
    wide.to_parquet(wide_path, index=False)
    print(f"  Stitched wide: {wide_path}  ({len(wide):,} rows)")

    for m in MODELS:
        sub = latest[latest["model"] == m]
        print(f"  {m:>12s}: {sub['time'].nunique()} dates, {len(sub):,} rows")


def main():
    print("=== Walk-forward V1 ===\n")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fold_specs = []

    for i, win in enumerate(WINDOWS):
        label = f"W{i+1}"
        fold_dir = OUTPUT_DIR / label

        # Checkpoint: skip if fold already completed.
        done_marker = fold_dir / "model_comparison.csv"
        if done_marker.exists():
            print(f"\n=== {label}: SKIP (already complete) ===")
            fold_specs.append({
                "label": label,
                "dir": fold_dir,
                "train_end": "", "valid_end": "",
            })
            continue

        b = _boundaries(win)
        print(f"\n{'='*60}")
        print(f"{label}: train={b[0]}..{b[1]}  valid={b[2]}..{b[3]}  test={b[4]}..{b[5]}")
        print(f"{'='*60}")

        cfg = _window_config(win, label)
        result = run_experiment(cfg)

        fold_specs.append({
            "label": label,
            "dir": Path(cfg.output_dir),
            "train_end": cfg.train_end,
            "valid_end": cfg.valid_end,
        })

    # Stitch
    print(f"\n{'='*60}")
    print("Stitching OOS predictions ...")
    _stitch_oos(fold_specs, OUTPUT_DIR)


if __name__ == "__main__":
    main()
