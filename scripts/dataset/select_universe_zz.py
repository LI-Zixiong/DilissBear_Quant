"""
Select stock universe from ZZ500 + ZZ1000 index constituents.

Replaces the scoring-based selection with a simple index-constituent universe
for experiments that need an external benchmark universe.
"""

from pathlib import Path

import pandas as pd
import numpy as np

ZZ500_PATH = Path("dataset/input/zz500.xls")
ZZ1000_PATH = Path("dataset/input/zz1000.xls")
FACTOR_PATH = Path("dataset/processed/factor_panel_54_ind.parquet")
OUTPUT_PATH = Path("dataset/processed/factor_panel_1500_54_ind.parquet")


def load_index_stocks(path: Path) -> set[str]:
    df = pd.read_excel(path)
    stock_col = df.columns[4]
    return set(df[stock_col].astype(str).str.strip().str.zfill(6))


def main() -> None:
    print("Loading index constituents...")
    s500 = load_index_stocks(ZZ500_PATH)
    s1000 = load_index_stocks(ZZ1000_PATH)
    universe = s500 | s1000
    overlap = s500 & s1000
    print(f"  zz500:     {len(s500)} stocks")
    print(f"  zz1000:    {len(s1000)} stocks")
    print(f"  overlap:   {len(overlap)}")
    print(f"  universe:  {len(universe)} stocks")

    print("\nLoading factor panel...")
    panel = pd.read_parquet(FACTOR_PATH)
    panel["stock_id"] = panel["stock_id"].astype(str).str.strip().str.zfill(6)

    total_stocks = panel["stock_id"].nunique()
    panel_filtered = panel[panel["stock_id"].isin(universe)].copy()
    filtered_stocks = panel_filtered["stock_id"].nunique()

    in_500 = len(s500 & set(panel_filtered["stock_id"].unique()))
    in_1000 = len(s1000 & set(panel_filtered["stock_id"].unique()))

    print(f"  panel total:       {len(panel):,} rows, {total_stocks} stocks")
    print(f"  panel filtered:    {len(panel_filtered):,} rows, {filtered_stocks} stocks")
    print(f"  zz500 coverage:    {in_500}/{len(s500)} ({in_500/len(s500)*100:.1f}%)")
    print(f"  zz1000 coverage:   {in_1000}/{len(s1000)} ({in_1000/len(s1000)*100:.1f}%)")

    print(f"\nSaving to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel_filtered.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH} ({len(panel_filtered):,} rows, {filtered_stocks} stocks)")


if __name__ == "__main__":
    main()
