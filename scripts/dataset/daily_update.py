"""
Daily update pipeline: pull Tushare → append base panel → compute new factors.
Safe: backs up key files before modifying. Dry-run mode available.
"""
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DRY_RUN = False  # Set True to see what would happen without writing

UNIFIED = Path("dataset/processed/unified_daily_panel.parquet")
FACTOR = Path("dataset/processed/factor_panel_54_ind.parquet")
POOL = Path("dataset/processed/factor_panel_1500_54_ind.parquet")
BACKUP_DIR = Path("dataset/processed/_backup")

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from src.pipeline.base_panel import BasePanelConfig, append_base_panel_rows
    from src.pipeline.factor_panel import FactorPanelConfig, append_factor_panel_rows

    # ── 1. Pull Tushare daily ──
    print(f"=== Daily Update {datetime.now():%Y-%m-%d %H:%M} ===\n")
    print("Step 1: Pulling Tushare daily data...")
    from temp.tushare_daily_update import pull_trade_dates, pull_one_date

    u = pd.read_parquet(UNIFIED)
    u["time"] = pd.to_datetime(u["time"])
    last_date = u["time"].max().strftime("%Y%m%d")
    today = datetime.now().strftime("%Y%m%d")

    all_dates = pull_trade_dates(last_date, today)
    have_dates = {d.strftime("%Y%m%d") for d in u["time"].dt.date.unique()}
    new_dates = [d for d in all_dates if d not in have_dates]
    print(f"  Existing dates: {len(have_dates)}, New: {len(new_dates)}")

    if not new_dates:
        print("  Already up to date. Done.")
        exit()

    parts = []
    for td in new_dates:
        df = pull_one_date(td)
        if df is not None and len(df) > 0:
            parts.append(df)
            print(f"  {td}: {len(df)} rows")
    if not parts:
        print("  No data pulled. Done.")
        exit()

    new_daily = pd.concat(parts, ignore_index=True)
    print(f"  Pulled {len(new_daily):,} rows\n")

    if DRY_RUN:
        print("DRY_RUN=True — stopping here.")
        exit()

    # ── 2. Append base panel ──
    print("Step 2: Appending to unified base panel...")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if not (BACKUP_DIR / UNIFIED.name).exists():
        import shutil
        shutil.copy2(UNIFIED, BACKUP_DIR / UNIFIED.name)

    financial = pd.read_parquet("dataset/processed/financial_quarterly_panel.parquet")
    base_cfg = BasePanelConfig()
    base = pd.read_parquet(UNIFIED)
    base["time"] = pd.to_datetime(base["time"])
    base["stock_id"] = base["stock_id"].astype(str).str.strip().str.zfill(6)
    updated_base = append_base_panel_rows(
        base_panel=base, new_daily_rows=new_daily,
        financial_quarterly=financial, config=base_cfg,
    )

    if DRY_RUN: exit()
    updated_base.to_parquet(UNIFIED, index=False)
    print(f"  Unified panel: {len(updated_base):,} rows\n")

    # ── 3. Append factor rows ──
    print("Step 3: Appending new factor rows...")
    if not (BACKUP_DIR / FACTOR.name).exists():
        import shutil
        shutil.copy2(FACTOR, BACKUP_DIR / FACTOR.name)

    factor_cfg = FactorPanelConfig(mode="live")
    existing = pd.read_parquet(FACTOR)
    existing["time"] = pd.to_datetime(existing["time"])
    existing["stock_id"] = existing["stock_id"].astype(str).str.strip().str.zfill(6)
    updated_factor = append_factor_panel_rows(
        existing_factor_panel=existing,
        updated_base_panel=updated_base,
        financial_quarterly=financial,
        config=factor_cfg,
    )
    updated_factor.to_parquet(FACTOR, index=False)
    print(f"  Factor panel: {len(updated_factor):,} rows")

    print("\nDone. Next: run select_universe + experiment.")
