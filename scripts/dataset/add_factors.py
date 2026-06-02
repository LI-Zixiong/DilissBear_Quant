"""
Column-extension entry point: add new factor columns to existing factor panel.
Does NOT recompute existing factors. Set BATCH below, run, then switch.

Batch 1: F025-F042 (18 factors, ~15 min)
Batch 2: F043-F048 (6 factors, slower: polyfit regressions)
"""
import pandas as pd
from src.pipeline.factor_panel import (
    FactorPanelConfig,
    FIN6_FACTORS,
    TECH24_FACTORS,
    merge_factor_panel_columns,
)

# -- Config --
BATCH = 3  # 1/2 = TECH24, 3 = FIN6, 0 = ALL
INPUT_FACTOR_PANEL = "dataset/processed/factor_panel_54_ind.parquet"
OUTPUT_FACTOR_PANEL = "dataset/processed/factor_panel_54_ind.parquet"

BATCH_MAP = {
    1: TECH24_FACTORS[:18],   # F025-F042
    2: TECH24_FACTORS[18:],   # F043-F048
    3: FIN6_FACTORS,          # F049-F054
    0: TECH24_FACTORS + FIN6_FACTORS,
}
# Fresh start from 24: override INPUT to 24 panel
if BATCH == 0:
    INPUT_FACTOR_PANEL = "dataset/processed/factor_panel_24_ind.parquet"
NEW_FACTORS = BATCH_MAP[BATCH]

if __name__ == "__main__":
    print(f"Batch {BATCH}: {len(NEW_FACTORS)} factors: {NEW_FACTORS[:3]}...")
    existing = pd.read_parquet(INPUT_FACTOR_PANEL)
    base = pd.read_parquet("dataset/processed/unified_daily_panel.parquet")
    financial = pd.read_parquet("dataset/processed/financial_quarterly_panel.parquet")

    config = FactorPanelConfig()
    updated = merge_factor_panel_columns(
        existing_factor_panel=existing,
        base_panel=base,
        financial_quarterly=financial,
        factor_names=NEW_FACTORS,
        config=config,
    )
    updated.to_parquet(OUTPUT_FACTOR_PANEL, index=False)
    print(
        f"Saved {OUTPUT_FACTOR_PANEL}: "
        f"{len(updated):,} rows, {len(updated.columns)} cols"
    )
