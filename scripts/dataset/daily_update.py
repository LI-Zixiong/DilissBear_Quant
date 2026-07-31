"""Transactional daily market/base/factor update.

Normal runs touch only committed monthly partitions and one dependency-sized
tail. Legacy monolithic parquet files are migration inputs and maintenance
exports; they are never rewritten on the prediction critical path.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class StepTimer:
    label: str = ""
    started: float = field(default_factory=time.perf_counter)

    def step(self, label: str) -> float:
        elapsed = time.perf_counter() - self.started
        print(f"  [{elapsed:6.1f}s] {label}")
        return elapsed

    def done(self) -> float:
        return self.step(self.label or "done")

UNIFIED = Path("dataset/processed/unified_daily_panel.parquet")
FACTOR = Path("dataset/processed/factor_panel_1500_54_ind.parquet")
FINANCIAL = Path("dataset/processed/financial_quarterly_panel.parquet")
LIVE_ROOT = Path("dataset/cache/live")


def migrate_legacy(store_root: Path = LIVE_ROOT) -> None:
    """One-time conversion of monolithic panels into monthly generations."""
    from src.pipeline.live_factor_schema import state_schema
    from src.pipeline.live_store import LiveStore

    store = LiveStore(store_root)
    if store.exists():
        raise FileExistsError(f"Live store is already initialized: {store.pointer_path}")
    for path in (UNIFIED, FACTOR):
        if not path.exists():
            raise FileNotFoundError(f"Migration input is missing: {path}")

    print("Migrating legacy panels to monthly immutable partitions...")
    with store.begin() as txn:
        print(f"  Reading {UNIFIED} once")
        base = pd.read_parquet(UNIFIED)
        txn.replace_dates("base_panel", base)
        del base

        print(f"  Reading {FACTOR} once")
        factor = pd.read_parquet(FACTOR)
        txn.replace_dates("factor_panel", factor)
        del factor

        manifest = txn.commit(
            metadata={
                "operation": "legacy_migration",
                "factor_state_schema": state_schema(),
            }
        )
    print(f"Migration committed as generation {manifest.generation}.")


def _inherit_static_metadata(
    new_panel: pd.DataFrame,
    latest_base_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Carry static/security metadata without reading the historical panel."""
    result = new_panel.copy()
    if latest_base_rows.empty:
        return result
    latest = (
        latest_base_rows.sort_values(["stock_id", "time"])
        .drop_duplicates("stock_id", keep="last")
        .set_index("stock_id")
    )
    # Only genuinely static metadata is inherited.  Dynamic fields
    # (trade_status, limit_status, price, volume, etc.) are never copied.
    static_cols = {"industry_sw", "list_date", "stk_name"}
    for col in static_cols:
        if col not in result.columns and col in latest.columns:
            result[col] = result["stock_id"].map(latest[col])
    return result


def _last_n_dates(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(frame["time"]).dropna().unique()).sort_values()
    if len(dates) <= count:
        return frame
    return frame[pd.to_datetime(frame["time"]) >= dates[-count]].copy()


def update_live_store(
    *,
    store_root: Path = LIVE_ROOT,
    dry_run: bool = False,
    end_date: str | None = None,
) -> dict[str, object]:
    """Fetch missing dates and atomically commit base plus factor partitions."""
    t = StepTimer(label="data_update total")

    from src.pipeline.base_panel import BasePanelConfig, forward_fill_financial_to_daily
    from src.pipeline.factor_panel import FactorPanelConfig, needed_columns_for_factors
    from src.pipeline.live_factor_engine import compute_live_factor_rows
    from src.pipeline.live_factor_schema import required_tail_rows, state_schema
    from src.pipeline.live_store import LiveStore
    from src.pipeline.tushare_client import pull_one_date, pull_trade_dates

    store = LiveStore(store_root)
    store.clean_orphans()
    base_last = store.latest_date("base_panel")
    factor_last = store.latest_date("factor_panel")
    if base_last is None or factor_last is None:
        raise ValueError("Committed base/factor watermarks are required")

    today = end_date or datetime.now().strftime("%Y%m%d")
    all_dates = pull_trade_dates(base_last.strftime("%Y%m%d"), today)
    missing_trade_dates = [d for d in all_dates if pd.Timestamp(d) > base_last]
    print(f"Committed base through {base_last.date()}; missing dates: {len(missing_trade_dates)}")
    t.step("store init + orphan check + Tushare query")

    pulled: list[pd.DataFrame] = []
    for trade_date in missing_trade_dates:
        frame = pull_one_date(trade_date)
        if frame is not None and not frame.empty:
            pulled.append(frame)
            print(f"  {trade_date}: {len(frame):,} rows")

    financial: pd.DataFrame | None = None
    if pulled:
        new_daily = pd.concat(pulled, ignore_index=True)
        new_daily["time"] = pd.to_datetime(new_daily["time"]).dt.normalize()
        financial = pd.read_parquet(FINANCIAL)
        new_base_panel = forward_fill_financial_to_daily(
            daily=new_daily,
            financial=financial,
            config=BasePanelConfig(),
        )
        if "adj_factor" not in new_base_panel.columns:
            new_base_panel["adj_factor"] = (
                new_base_panel["close_adj"]
                / new_base_panel["close"].clip(lower=1e-12)
            )
        latest_base = store.read("base_panel", tail_dates=1)
        new_base_panel = _inherit_static_metadata(new_base_panel, latest_base)
        output_dates = sorted(pd.Timestamp(v) for v in new_base_panel["time"].unique())
        t.step("Tushare pull + base panel ffill")
    else:
        new_base_panel = pd.DataFrame()
        if factor_last >= base_last:
            print("Base and factor partitions are already current.")
            return {
                "updated": False,
                "generation": store.pointer().generation,
                "latest_date": factor_last,
            }
        catchup = store.read("base_panel", start=factor_last)
        output_dates = sorted(
            pd.Timestamp(v) for v in catchup["time"].unique()
            if pd.Timestamp(v) > factor_last
        )

    if not output_dates:
        print("No factor dates require computation.")
        return {
            "updated": False,
            "generation": store.pointer().generation,
            "latest_date": factor_last,
        }

    factor_cfg = FactorPanelConfig(mode="live")
    tail_count = required_tail_rows(
        factor_cfg.factor_names,
        new_dates=len(output_dates),
    )
    projected_columns_all = sorted(
        needed_columns_for_factors(factor_cfg.factor_names, factor_cfg)
    )
    # Strip columns not physically stored in base_panel partitions.
    # adj_factor is always computed from close_adj / close below — never
    # request it from the store, even if a recent partition happens to have it
    # (column-check reads tail_dates=1 which may differ from older partitions).
    col_check = store.read("base_panel", tail_dates=1).columns
    projected_columns = [c for c in projected_columns_all
                         if c in col_check and c != "adj_factor"]
    base_tail = store.read(
        "base_panel",
        columns=projected_columns,
        tail_dates=tail_count,
    )
    if not new_base_panel.empty:
        projected_new = new_base_panel[
            [col for col in projected_columns if col in new_base_panel.columns]
        ]
        base_tail = pd.concat([base_tail, projected_new], ignore_index=True)
        base_tail = base_tail.sort_values(["time", "stock_id"]).drop_duplicates(
            ["time", "stock_id"], keep="last"
        )
    base_tail = _last_n_dates(base_tail, tail_count)

    # adj_factor is computed internally by compute_factor_columns from
    # close_adj / close; the base panel does not store it as a column.
    if "adj_factor" not in base_tail.columns and "close_adj" in base_tail.columns and "close" in base_tail.columns:
        base_tail["adj_factor"] = base_tail["close_adj"] / base_tail["close"].clip(lower=1e-12)

    t.step(f"read projected tail ({base_tail['time'].nunique()} dates, {base_tail.shape[1]} cols)")

    if financial is None:
        financial = pd.read_parquet(FINANCIAL)
    factor_latest = store.read(
        "factor_panel",
        columns=["time", "stock_id"],
        tail_dates=5,
    )
    universe = factor_latest["stock_id"].unique()
    print(
        f"Computing {len(output_dates)} new date(s) from one "
        f"{base_tail['time'].nunique()}-date projected tail"
    )
    new_factor_rows = compute_live_factor_rows(
        base_tail=base_tail,
        financial_quarterly=financial,
        output_dates=output_dates,
        config=factor_cfg,
        universe_stocks=universe,
    )
    t.step(f"factor computation done ({len(new_factor_rows):,} rows)")

    if dry_run:
        print(
            f"Dry run: base={len(new_base_panel):,} rows, "
            f"factor={len(new_factor_rows):,} rows; nothing committed."
        )
        return {
            "updated": False,
            "generation": store.pointer().generation,
            "latest_date": max(output_dates),
            "factor_rows": new_factor_rows,
        }

    with store.begin() as txn:
        if not new_base_panel.empty:
            txn.replace_dates("base_panel", new_base_panel)
        txn.replace_dates("factor_panel", new_factor_rows)
        committed = txn.commit(
            metadata={
                "operation": "daily_update",
                "output_dates": [value.date().isoformat() for value in output_dates],
                "factor_state_schema": state_schema(),
                "financial_policy": "point_in_time_no_history_rewrite",
            }
        )
    print(
        f"Committed generation {committed.generation}: "
        f"{len(new_base_panel):,} base rows, {len(new_factor_rows):,} factor rows"
    )
    t.done()
    update_indices(store_root=store_root, end_date=end_date)
    return {
        "updated": True,
        "generation": committed.generation,
        "latest_date": max(output_dates),
        "factor_rows": new_factor_rows,
    }


def export_legacy(store_root: Path = LIVE_ROOT) -> None:
    """Maintenance-only compatibility snapshot generation."""
    from src.pipeline.live_store import LiveStore

    store = LiveStore(store_root)
    for dataset, path in (("base_panel", UNIFIED), ("factor_panel", FACTOR)):
        print(f"Exporting {dataset} -> {path}")
        frame = store.read(dataset)
        tmp = path.with_name(f"{path.name}.tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=LIVE_ROOT)
    parser.add_argument("--migrate", action="store_true")
    parser.add_argument("--export-legacy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--end-date", help="Inclusive YYYYMMDD override")
    parser.add_argument("--dump-schema", type=Path)
    parser.add_argument("--update-indices", action="store_true",
                        help="Pull latest ZZ500 + ZZ1000 index daily data")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object] | None:
    args = _parser().parse_args(argv)
    if args.update_indices:
        update_indices(end_date=args.end_date)
        return {"updated": True}
    if args.dump_schema:
        from src.pipeline.live_factor_schema import state_schema

        args.dump_schema.parent.mkdir(parents=True, exist_ok=True)
        args.dump_schema.write_text(
            json.dumps(state_schema(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return None
    if args.migrate:
        migrate_legacy(args.store_root)
        return None
    if args.export_legacy:
        export_legacy(args.store_root)
        return None
    return update_live_store(
        store_root=args.store_root,
        dry_run=args.dry_run,
        end_date=args.end_date,
    )


def update_indices(store_root: str | Path = "dataset/cache/live",
                   input_dir: str | Path = "dataset/input",
                   end_date: str | None = None) -> None:
    """Pull latest ZZ500 + ZZ1000 index daily data from Tushare and append."""
    from src.pipeline.tushare_client import pull_index_daily, pull_trade_dates

    indices = {
        "zz500_daily.parquet": "000905.SH",
        "zz1000_daily.parquet": "000852.SH",
    }
    today = pd.Timestamp.now().strftime("%Y%m%d")
    end = end_date or today
    input_dir = Path(input_dir)

    for fname, ts_code in indices.items():
        path = input_dir / fname
        if not path.exists():
            print(f"  {fname}: not found, skipping")
            continue
        existing = pd.read_parquet(path)
        last_date = existing["time"].max()
        last_str = last_date.strftime("%Y%m%d")
        if last_str >= end:
            print(f"  {fname}: up to date ({last_str})")
            continue

        # Start from next day to avoid re-pulling existing data
        start_str = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
        new = pull_index_daily(ts_code, start_str, end)
        if new is None or new.empty:
            print(f"  {fname}: no new data ({last_str} → {end})")
            continue

        # Deduplicate and append
        combined = pd.concat([existing, new], ignore_index=True)
        combined = combined.drop_duplicates(["time"], keep="last").sort_values("time")
        combined.to_parquet(path, index=False)
        print(f"  {fname}: {last_str} → {combined['time'].max().strftime('%Y-%m-%d')}  "
              f"(+{len(new)} rows, {len(combined)} total)")


if __name__ == "__main__":
    main()
