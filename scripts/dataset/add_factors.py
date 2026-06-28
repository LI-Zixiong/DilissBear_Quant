"""
Column-extension entry point: add new factor columns to existing factor panel.

  Batch 1-3: TECH24 + FIN6 (original)
  Batch 4-9: V2 expansion F056GAP_UP_FAIL-F100INV_MINUS_REV (45 factors, ≤10 per batch)

Usage:
    python -m scripts.dataset.add_factors
    python -m scripts.dataset.add_factors --batch 4
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.pipeline.factor_panel import (
    FactorPanelConfig,
    TECH24_FACTORS, FIN6_FACTORS, NEW45_FACTORS,
    merge_factor_panel_columns,
)

# -- Config --
INPUT_FACTOR_PANEL = "dataset/processed/factor_panel_1500_54_ind.parquet"
OUTPUT_FACTOR_PANEL = "dataset/processed/factor_panel_1500_54_ind.parquet"

V2_BATCHES = {
    4: NEW45_FACTORS[:10],   # F056GAP_UP_FAIL-F065MAXDD20
    5: NEW45_FACTORS[10:20], # F066EFFICIENCY20-F075VOLUME_RATIO
    6: NEW45_FACTORS[20:30], # F076SIGNED_AMT20-F085SP_TTM
    7: NEW45_FACTORS[30:40], # F086DIV_TTM-F095NET_FIN
    8: NEW45_FACTORS[40:45], # F096DILUTION-F100INV_MINUS_REV
}

ALL_BATCHES = {
    1: TECH24_FACTORS[:18],
    2: TECH24_FACTORS[18:],
    3: FIN6_FACTORS,
    **V2_BATCHES,
}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--factor", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output_path = args.output or OUTPUT_FACTOR_PANEL

    if args.factor:
        print(f"Single factor: {args.factor} -> {output_path}")
        existing = pd.read_parquet(output_path)
        # Only load base rows within factor panel's date range + 4y warmup,
        # and only stocks in the factor panel (1500 ZZ500+1000 universe).
        warmup = pd.Timestamp(existing["time"].min()) - pd.DateOffset(years=4)
        universe = sorted(existing["stock_id"].unique())
        base = pd.read_parquet(
            "dataset/processed/unified_daily_panel.parquet",
            filters=[("time", ">=", warmup), ("stock_id", "in", universe)],
        )
        financial = pd.read_parquet("dataset/processed/financial_quarterly_panel.parquet")
        config = FactorPanelConfig()
        updated = merge_factor_panel_columns(
            existing_factor_panel=existing,
            base_panel=base,
            financial_quarterly=financial,
            factor_names=[args.factor],
            config=config,
            overwrite=True,
        )
        updated.to_parquet(output_path, index=False)
        print(f"  Saved: {output_path} ({len(updated):,} rows, {len(updated.columns)} cols)")
        sys.exit(0)

    if args.batch == 0:
        batches = list(V2_BATCHES.keys())
        print(f"V2 expansion: batches {batches}")
    else:
        batches = [args.batch]

    for b in batches:
        names = ALL_BATCHES[b]
        print(f"\nBatch {b}: {len(names)} factors: {names[:3]}... -> {output_path}")
        existing = pd.read_parquet(output_path)
        # Only load base rows within factor panel's date range + 4y warmup,
        # and only stocks in the factor panel (1500 ZZ500+1000 universe).
        warmup = pd.Timestamp(existing["time"].min()) - pd.DateOffset(years=4)
        universe = sorted(existing["stock_id"].unique())
        base = pd.read_parquet(
            "dataset/processed/unified_daily_panel.parquet",
            filters=[("time", ">=", warmup), ("stock_id", "in", universe)],
        )
        financial = pd.read_parquet("dataset/processed/financial_quarterly_panel.parquet")

        config = FactorPanelConfig()
        updated = merge_factor_panel_columns(
            existing_factor_panel=existing,
            base_panel=base,
            financial_quarterly=financial,
            factor_names=names,
            config=config,
            overwrite=True,
        )
        updated.to_parquet(output_path, index=False)
        print(f"  Saved: {output_path} ({len(updated):,} rows, {len(updated.columns)} cols)")

    print("\nDone.")
