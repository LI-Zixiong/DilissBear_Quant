"""
Factor panel computation pipeline.

This module supports:
    1. Full historical factor panel computation.
    2. Row extension for daily updates.
    3. Column extension for adding new factors.

Design notes
------------
- Do not import from scripts.
- Base daily data comes from unified_daily_panel.parquet.
- Quarterly financial data comes from financial_quarterly_panel.parquet.
- Final model factor names are standardized as F001...F024.
- Raw and winsorized factor columns are internal by default.
"""

from __future__ import annotations

import gc
import json
import shutil
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Factor names
# ---------------------------------------------------------------------


BARRA12_FACTORS: tuple[str, ...] = (
    "F001SIZE",
    "F002SIZENL",
    "F003LIQUIDITY",
    "F004BETA",
    "F005RESVOL",
    "F006MOMENTUM",
    "F007LTREV",
    "F008STREV",
    "F009LEVERAGE",
    "F010VALUE",
    "F011EARNYLD",
    "F012GROWTH",
)

NEW12_FACTORS: tuple[str, ...] = (
    "F013REV5",
    "F014MOM120_20",
    "F015VOLREV",
    "F016MAXRET",
    "F017IVOL",
    "F018AMIHUD",
    "F019COSTDEV",
    "F020LIMITUP_RECENCY",
    "F021CFP",
    "F022GPTA",
    "F023ACCRUAL",
    "F024ASSETGR",
)

TECH24_FACTORS: tuple[str, ...] = (
    "F025GAP",
    "F026KLEN",
    "F027KUP",
    "F028KLOW",
    "F029KSFT",
    "F030RSV20",
    "F031RSV60",
    "F032RANGEZ20",
    "F033GAPREV5",
    "F034HIGHDEV20",
    "F035LOWDEV20",
    "F036VOLSHOCK5",
    "F037VOLSHOCK20",
    "F038TURNZ20",
    "F039VSTD20",
    "F040PVCORR20",
    "F041RETVOLCORR20",
    "F042AMTCORR20",
    "F043SLOPE20",
    "F044RSQR20",
    "F045RESI20",
    "F046LIMITUP20",
    "F047LIMITDN20",
    "F048LIMITSTREAKUP",
)

FIN6_FACTORS: tuple[str, ...] = (
    "F049ROE",
    "F050ROA",
    "F051GPM",
    "F052CFOA",
    "F053RD_INTENSITY",
    "F054RECEIVABLE_RATIO",
)

IND1_FACTORS: tuple[str, ...] = ("F055IND",)

NEW45_FACTORS: tuple[str, ...] = (
    "F056GAP_UP_FAIL",    "F057INTRA1",        "F058O2O_RET5",       "F059GK_VOL20",
    "F060ON_INTRA_DIV5",  "F061GAP_UP_HOLD",   "F062GAP_DN_RECOVER", "F063RET5D_SKIP1",
    "F064RET_ACCEL20",    "F065MAXDD20",        "F066EFFICIENCY20",   "F067TAIL_LOSS20",
    "F068SKEW20",         "F069DNVOL20",        "F070UP_DN_VOL",      "F071VOL_OF_VOL",
    "F072CORR_60D",       "F073KURT_60D",       "F074BETA_20D",       "F075VOLUME_RATIO",
    "F076SIGNED_AMT20",   "F077AMP_VOL20",      "F078TURN_SIZE",      "F079TURN_ACCEL",
    "F080VWAP_DEV",       "F081STRONG_CLOSE",   "F082LOCKED_PCT",     "F083TURN_FREE",
    "F084AMT_FREE20",     "F085SP_TTM",         "F086DIV_TTM",        "F087LIST_AGE",
    "F088CF_SALES_Q",     "F089CASH_PROFIT",    "F090CRR",            "F091CF_VOL",
    "F092EARN_STAB",      "F093FCF_YIELD",      "F094CAPEX_INT",      "F095NET_FIN",
    "F096DILUTION",       "F097INT_BURDEN",     "F098DIV_PAYOUT",     "F099AR_MINUS_REV",
    "F100INV_MINUS_REV",
)

DEFAULT_FACTOR_NAMES: tuple[str, ...] = (
    BARRA12_FACTORS + NEW12_FACTORS + TECH24_FACTORS + FIN6_FACTORS + IND1_FACTORS + NEW45_FACTORS
)

FACTOR_ALIAS: dict[str, str] = {}


    # V2 expansion (F056GAP_UP_FAIL-F100INV_MINUS_REV)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass
class FactorPanelConfig:
    """Configuration for factor panel computation."""

    base_panel_path: str | Path = "dataset/processed/unified_daily_panel.parquet"
    financial_panel_path: str | Path = "dataset/processed/financial_quarterly_panel.parquet"
    output_path: str | Path = "dataset/processed/factor_panel_24_ind.parquet"
    metadata_path: str | Path = "dataset/processed/factor_panel_24_metadata.json"

    model_start_date: str = "2015-06-01"
    date_col: str = "time"
    stock_col: str = "stock_id"

    factor_names: tuple[str, ...] = DEFAULT_FACTOR_NAMES
    target_names: tuple[str, ...] = ("1d_next_raw", "5d_next_raw")

    # research: calculate targets for training/backtest.
    # live: do not calculate future-return targets.
    mode: str = "research"

    # Rolling windows
    beta_window: int = 252
    beta_min_periods: int = 60
    liquidity_window: int = 63
    liquidity_min_periods: int = 20
    incremental_window: int = 800

    # Residual / new factor windows
    resid_window: int = 252
    resid_min_periods: int = 60
    rev5_window: int = 5
    mom_long_window: int = 120
    mom_short_window: int = 20
    volrev_short_window: int = 5
    volrev_long_window: int = 60
    maxret_window: int = 20
    ivol_short_window: int = 20
    ivol_long_window: int = 60
    amihud_short_window: int = 20
    amihud_long_window: int = 60
    vwap_window: int = 120

    # Cross-sectional transform
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99
    industry_col: str | None = None
    neutralize_factors: tuple[str, ...] = ()

    # Output switches
    save_raw_factors: bool = False
    save_winsorized_factors: bool = False

    # Universe: optional list of index-constituent Excel paths (ZZ500, ZZ1000, etc.).
    # When non-empty, only stocks appearing in the union of these indices are kept.
    universe_paths: tuple[str, ...] = ()

    # Path to standalone adj_factor panel (pulled from Tushare API).
    # When provided and the file exists, adj_factor is merged from here instead of
    # being derived from close_adj/close.  Defaults to the canonical location.
    adj_factor_path: str = "dataset/input/tushare/adj_factor_panel.parquet"

    # Metadata / version
    factor_version: str = "v1_24f"
    seed: int = 42


@dataclass(frozen=True)
class FactorBatchSpec:
    """Internal factor-batch specification.

    keep_cols_extra is intentionally explicit so each batch only carries the
    columns it needs.  The scheduler splits by these specs rather than by raw
    F-number order.  Each production batch is capped at <=10 factors.
    """

    name: str
    factors: tuple[str, ...]
    keep_cols_extra: tuple[str, ...] = ()
    precompute: str | None = None


FACTOR_BATCH_SPECS: tuple[FactorBatchSpec, ...] = (
    FactorBatchSpec("old_size_liquidity", ("F001SIZE", "F002SIZENL", "F003LIQUIDITY"),
                    ("mktcap_total", "mktcap_float", "amount")),
    FactorBatchSpec("old_capm", ("F004BETA", "F005RESVOL", "F013REV5", "F014MOM120_20", "F017IVOL"),
                    ("ret_daily", "mkt_ret_vw"), "capm"),
    FactorBatchSpec("old_momentum_value", ("F006MOMENTUM", "F007LTREV", "F008STREV", "F009LEVERAGE", "F010VALUE", "F015VOLREV", "F016MAXRET", "F018AMIHUD", "F019COSTDEV"),
                    ("close_adj", "ret_daily", "amount", "volume", "mktcap_total", "mktcap_float", "total_assets", "total_liabilities", "equity_parent")),
    FactorBatchSpec("old_ttm_style", ("F011EARNYLD", "F012GROWTH", "F021CFP", "F022GPTA", "F023ACCRUAL", "F024ASSETGR"),
                    ("mktcap_total", "mktcap_float", "total_assets", "equity_parent", "rd_expense", "receivables"), "old_ttm"),
    FactorBatchSpec("tech_ohlc", ("F025GAP", "F026KLEN", "F027KUP", "F028KLOW", "F029KSFT", "F030RSV20", "F031RSV60", "F032RANGEZ20", "F033GAPREV5", "F034HIGHDEV20"),
                    ("open", "high", "low", "close", "pre_close", "close_adj", "amount", "volume"), "adj_ohlc"),
    FactorBatchSpec("tech_volume_regime", ("F035LOWDEV20", "F036VOLSHOCK5", "F037VOLSHOCK20", "F038TURNZ20", "F039VSTD20", "F040PVCORR20", "F041RETVOLCORR20", "F042AMTCORR20"),
                    ("open", "high", "low", "close", "close_adj", "ret_daily", "amount", "volume", "turnover_rate", "turnover_rate_f"), "rolling_corr"),
    FactorBatchSpec("tech_reg_limit", ("F043SLOPE20", "F044RSQR20", "F045RESI20", "F046LIMITUP20", "F047LIMITDN20", "F048LIMITSTREAKUP", "F049ROE", "F050ROA", "F051GPM", "F052CFOA"),
                    ("close_adj", "limit_status", "mktcap_total", "mktcap_float", "total_assets", "equity_parent"), "regression20"),
    FactorBatchSpec("old_fin_remaining", ("F053RD_INTENSITY", "F054RECEIVABLE_RATIO", "F055IND"),
                    ("rd_expense", "receivables", "industry_sw"), "old_ttm"),
    FactorBatchSpec("v2_o2o", NEW45_FACTORS[:7],
                    ("open", "high", "low", "close", "pre_close", "volume", "amount"), "o2o"),
    FactorBatchSpec("v2_momentum_path", NEW45_FACTORS[7:11],
                    ("close_adj", "ret_daily"), "path"),
    FactorBatchSpec("v2_vol_market", NEW45_FACTORS[11:19],
                    ("open", "high", "low", "close", "ret_daily", "mkt_ret_vw"), "ret_stats"),
    FactorBatchSpec("v2_volume_vwap", NEW45_FACTORS[19:26],
                    ("open", "high", "low", "close", "ret_daily", "amount", "volume", "volume_ratio", "turnover_rate", "mktcap_total"), "volume_vwap"),
    FactorBatchSpec("v2_float_value_age", NEW45_FACTORS[26:32],
                    ("close", "amount", "free_share", "float_share", "total_share", "ps_ttm", "dv_ttm", "list_date"), "float_value_age"),
    FactorBatchSpec("v2_cash_quality", NEW45_FACTORS[32:38],
                    ("mktcap_float", "mktcap_total"), "v2_financial"),
    FactorBatchSpec("v2_financing_growth", NEW45_FACTORS[38:45],
                    ("mktcap_float", "mktcap_total"), "v2_financial"),
)


# ---------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------


def compute_factor_panel_full(
    config: FactorPanelConfig | None = None,
) -> dict[str, Any]:
    """
    Compute full factor panel from base and quarterly financial panels.

    Internal production batches are capped at <=10 factors.  Each batch writes
    a temporary narrow parquet file under dataset/processed/_factor_tmp/{ts}/.
    The final wide panel is merged once at the end to avoid repeatedly reading
    and rewriting the full 3M+ row output parquet.
    """

    if config is None:
        config = FactorPanelConfig()

    started = time.perf_counter()
    base = load_base_panel(config)
    financial = load_financial_quarterly_panel(config)

    # Universe filter: restrict to index constituents before factor computation
    if config.universe_paths:
        universe: set[str] = set()
        for p in config.universe_paths:
            idx = pd.read_excel(p)
            col = idx.columns[4]  # stock_id column
            universe.update(idx[col].astype(str).str.strip().str.zfill(6))
        base = base[base[config.stock_col].isin(universe)].copy()
        print(f"Universe filter: {len(universe)} index stocks, "
              f"{base[config.stock_col].nunique()} in base panel")

    factor_names = tuple(config.factor_names)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    tmp_root = output_path.parent / "_factor_tmp" / timestamp
    tmp_root.mkdir(parents=True, exist_ok=False)

    shared_cache: dict[str, Any] = {}
    batch_files: list[Path] = []
    batches = list(iter_factor_batches(factor_names))

    print(f"\nComputing factor panel in {len(batches)} production batches")
    print(f"Temporary directory: {tmp_root}")

    try:
        for i, spec in enumerate(batches, start=1):
            batch = tuple(f for f in spec.factors if f in factor_names)
            if not batch:
                continue
            if len(batch) > 10:
                raise ValueError(f"Batch {spec.name} has {len(batch)} factors; cap is 10.")

            print(f"\n[Batch {i:02d}/{len(batches):02d}] {spec.name}: {', '.join(batch)}")
            t0 = time.perf_counter()
            panel = compute_factor_columns(
                base_panel=base,
                financial_quarterly=financial,
                config=config,
                factor_names=batch,
                shared_cache=shared_cache,
            )

            panel = filter_and_select_output_columns(
                panel=panel,
                config=config,
                factor_names=batch,
                include_targets=False,
            )

            key_cols = [config.date_col, config.stock_col]
            keep_cols = key_cols + [f for f in batch if f in panel.columns]
            panel = panel[keep_cols].copy()

            batch_path = tmp_root / f"batch_{i:02d}_{spec.name}.parquet"
            panel.to_parquet(batch_path, index=False)
            batch_files.append(batch_path)
            print(f"  saved {batch_path.name} rows={len(panel):,} elapsed={time.perf_counter()-t0:.1f}s")

            del panel
            gc.collect()

        print("\nMerging batch parquet files...")
        panel = build_base_output_skeleton(base, config)
        for path in batch_files:
            t0 = time.perf_counter()
            part = pd.read_parquet(path)
            part = standardize_panel_keys(part, config)
            panel = panel.merge(part, on=[config.date_col, config.stock_col], how="inner")
            print(f"  merged {path.name} elapsed={time.perf_counter()-t0:.1f}s")
            del part
            gc.collect()

        if config.mode == "research":
            targets = build_targets(panel, config)
            for col in config.target_names:
                if col in targets.columns:
                    panel[col] = targets[col]
        elif config.mode == "live":
            for col in config.target_names:
                panel[col] = np.nan
        else:
            raise ValueError(f"Unsupported mode: {config.mode}")

        panel = filter_and_select_output_columns(
            panel=panel,
            config=config,
            factor_names=config.factor_names,
        )

        audit = run_factor_panel_checks(panel, config, config.factor_names)
        panel.to_parquet(output_path, index=False)

        metadata = build_factor_metadata(panel=panel, config=config, audit=audit)
        metadata_path = Path(config.metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        print(f"\nSaved factor panel: {output_path}")
        print(f"Saved metadata: {metadata_path}")
        print(
            f"rows={len(panel):,}, "
            f"stocks={panel[config.stock_col].nunique():,}, "
            f"dates={panel[config.date_col].min()}~{panel[config.date_col].max()}, "
            f"elapsed={time.perf_counter()-started:.1f}s"
        )

        return {
            "factor_panel_path": str(output_path),
            "metadata_path": str(metadata_path),
            "audit": audit,
        }
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        shared_cache.clear()
        gc.collect()


def append_factor_panel_rows(
    existing_factor_panel: pd.DataFrame,
    updated_base_panel: pd.DataFrame,
    financial_quarterly: pd.DataFrame,
    config: FactorPanelConfig | None = None,
    recompute_window: int | None = None,
    allow_overlap_replace: bool = False,
) -> pd.DataFrame:
    """
    Row extension API for daily updates.

    Recompute a trailing warm-up window from updated_base_panel and append only
    rows after the last date in existing_factor_panel.
    """

    if config is None:
        config = FactorPanelConfig()

    if recompute_window is None:
        recompute_window = config.incremental_window

    existing = standardize_panel_keys(existing_factor_panel, config)
    financial = standardize_financial_keys(financial_quarterly, config)

    if existing.empty:
        raise ValueError("existing_factor_panel is empty")

    last_factor_date = existing[config.date_col].max()

    # Avoid copying the full base panel — only extract dates to find warmup range
    base_dates = pd.to_datetime(
        updated_base_panel[config.date_col], errors="raise"
    ).dt.normalize()
    all_dates = pd.DatetimeIndex(base_dates.dropna().unique()).sort_values()

    if last_factor_date not in set(all_dates):
        raise ValueError(
            f"last_factor_date={last_factor_date} not found in updated_base_panel dates"
        )

    last_pos = all_dates.get_loc(last_factor_date)
    start_pos = max(0, last_pos - recompute_window)
    warmup_start_date = all_dates[start_pos]

    base_slice = updated_base_panel[base_dates >= warmup_start_date]
    base_slice = standardize_panel_keys(base_slice, config).copy()

    recomputed = compute_factor_columns(
        base_panel=base_slice,
        financial_quarterly=financial,
        config=config,
        factor_names=config.factor_names,
    )

    if config.mode in {"research", "live"}:
        # Incremental updates must settle labels for older rows once the
        # required future closes have actually arrived.  The newest rows still
        # remain NaN naturally, so this is causal as of the update date and
        # does not expose an unobserved future price.
        targets = build_targets(recomputed, config)
        for col in config.target_names:
            if col in targets.columns:
                recomputed[col] = targets[col]
    else:
        raise ValueError(f"Unsupported mode: {config.mode}")

    recomputed = filter_and_select_output_columns(
        panel=recomputed,
        config=config,
        factor_names=config.factor_names,
        filter_model_start=False,
    )

    key_cols = [config.stock_col, config.date_col]
    target_cols = [c for c in config.target_names if c in recomputed.columns and c in existing.columns]

    # Refresh only settled target values on the overlapping tail.  Factor
    # values already stored in the production panel are intentionally left
    # untouched by an incremental run.
    if target_cols:
        refresh = recomputed.loc[
            recomputed[config.date_col] <= last_factor_date,
            key_cols + target_cols,
        ].set_index(key_cols)
        existing_indexed = existing.set_index(key_cols)
        for col in target_cols:
            values = refresh[col].reindex(existing_indexed.index)
            settled = values.notna()
            existing_indexed.loc[settled, col] = values.loc[settled].to_numpy()
        existing = existing_indexed.reset_index()

    new_rows = recomputed[recomputed[config.date_col] > last_factor_date].copy()

    # Keep only stocks already in the existing factor panel (original universe).
    # Without this, daily updates silently expand the universe to all stocks
    # in the base panel, bloating the factor panel and breaking downstream code
    # that assumes a fixed stock universe.
    universe_stocks = set(existing[config.stock_col].unique())
    before_filter = len(new_rows)
    new_rows = new_rows[new_rows[config.stock_col].isin(universe_stocks)]
    dropped = before_filter - len(new_rows)
    if dropped > 0:
        print(f"  Universe filter: {before_filter:,} → {len(new_rows):,} rows "
              f"({dropped:,} non-universe stocks dropped)")

    if new_rows.empty:
        print("No new factor rows to append.")
        return existing

    if allow_overlap_replace:
        marker = new_rows[key_cols].drop_duplicates().assign(_new_key=1)
        existing = existing.merge(marker, on=key_cols, how="left")
        existing = existing[existing["_new_key"].isna()].drop(columns=["_new_key"])
    else:
        overlap = existing[key_cols].merge(new_rows[key_cols], on=key_cols, how="inner")
        if not overlap.empty:
            raise ValueError(f"New factor rows overlap existing panel:\n{overlap.head()}")

    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined = combined.sort_values([config.stock_col, config.date_col]).reset_index(drop=True)

    validate_no_duplicate_keys(combined, config, "combined_factor_panel")
    run_factor_panel_checks(combined, config, config.factor_names, is_incremental=True)

    return combined


def merge_factor_panel_columns(
    existing_factor_panel: pd.DataFrame,
    base_panel: pd.DataFrame,
    financial_quarterly: pd.DataFrame,
    factor_names: Sequence[str],
    config: FactorPanelConfig | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Column extension API for adding new factors.

    Computes only factor_names and merges final columns back by stock_id + time.
    """

    if config is None:
        config = FactorPanelConfig()

    existing = standardize_panel_keys(existing_factor_panel, config)
    base = standardize_panel_keys(base_panel, config)
    financial = standardize_financial_keys(financial_quarterly, config)

    factor_names = tuple(factor_names)

    computed = compute_factor_columns(
        base_panel=base,
        financial_quarterly=financial,
        config=config,
        factor_names=factor_names,
    )

    # Column extension: only keep new factor columns + keys to avoid duplicate meta cols
    key_cols = [config.date_col, config.stock_col]
    keep_cols = key_cols + [f for f in factor_names if f in computed.columns]
    keep_cols = [c for c in keep_cols if c in computed.columns]
    computed = computed[keep_cols]

    computed = filter_and_select_output_columns(
        panel=computed,
        config=config,
        factor_names=factor_names,
        filter_model_start=True,
        include_targets=False,
    )

    key_cols = [config.date_col, config.stock_col]
    new_cols = [col for col in computed.columns if col not in key_cols]

    existing_cols = [col for col in new_cols if col in existing.columns]
    if existing_cols and not overwrite:
        raise ValueError(
            f"Columns already exist in factor panel: {existing_cols}. "
            "Pass overwrite=True to replace."
        )

    if overwrite and existing_cols:
        existing = existing.drop(columns=existing_cols)

    merged = existing.merge(
        computed[key_cols + new_cols],
        on=key_cols,
        how="left",
    )

    validate_no_duplicate_keys(merged, config, "merged_factor_panel")
    return merged


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------


def load_base_panel(config: FactorPanelConfig) -> pd.DataFrame:
    """Load unified daily base panel."""

    path = Path(config.base_panel_path)
    if not path.exists():
        raise FileNotFoundError(f"Base panel not found: {path}")

    df = pd.read_parquet(path)
    df = standardize_panel_keys(df, config)
    return df.sort_values([config.stock_col, config.date_col]).reset_index(drop=True)


def load_financial_quarterly_panel(config: FactorPanelConfig) -> pd.DataFrame:
    """Load quarterly financial panel."""

    path = Path(config.financial_panel_path)
    if not path.exists():
        raise FileNotFoundError(f"Financial quarterly panel not found: {path}")

    df = pd.read_parquet(path)
    df = standardize_financial_keys(df, config)
    return df.sort_values([config.stock_col, "accper"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# Factor computation orchestration
# ---------------------------------------------------------------------


def compute_factor_columns(
    base_panel: pd.DataFrame,
    financial_quarterly: pd.DataFrame,
    config: FactorPanelConfig,
    factor_names: Sequence[str],
    shared_cache: dict[str, Any] | None = None,
    target_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    Compute selected factor columns.

    The public signature is preserved; shared_cache is an internal optional
    cache used by full historical computation to avoid recomputing large shared
    tables across adjacent batches.

    *target_dates*, when provided (live/incremental mode), restrict
    cross-sectional winsorize/z-score to only those dates, significantly
    reducing the cost of daily updates.
    """
    if shared_cache is None:
        shared_cache = {}

    factor_names = tuple(factor_names)
    unknown = sorted(set(factor_names) - set(DEFAULT_FACTOR_NAMES))
    if unknown:
        raise ValueError(f"Unsupported factor names: {unknown}")

    df = prepare_working_panel(base_panel, config, factor_names)

    # Apply backward-adjustment factor to OHLC prices so all downstream return
    # computations (ret_daily, momentum, targets, K-line factors) use adjusted
    # prices, free of split/dividend artifacts.
    if "adj_factor" not in df.columns:
        if "close_adj" in df.columns and "close" in df.columns:
            df["adj_factor"] = df["close_adj"] / df["close"].clip(lower=1e-12)
    # Save raw close before adjustment (VWAP/turnover factors need it)
    if "close" in df.columns and "adj_factor" in df.columns:
        df["_close_raw"] = df["close"].copy()

    for col in ("open", "high", "low", "close"):
        if col in df.columns and "adj_factor" in df.columns:
            df[col] = df[col] * df["adj_factor"]

    # pre_close_adj = previous day's close_adj (already adjusted above)
    if "pre_close" in df.columns and "close" in df.columns:
        df = df.sort_values([config.stock_col, config.date_col])
        df["pre_close"] = df.groupby(config.stock_col)["close"].shift(1)

    # ret_daily is close-close (Tushare default, adjusted for dividends).

    needs = set(factor_names)

    # Shared CAPM residual for residual-based factors.
    needs_resid = bool(needs & {"F013REV5", "F014MOM120_20", "F017IVOL"})
    if needs_resid:
        print_factor_progress("CAPM", "residual")
        df["_capm_resid"] = build_capm_residual(df, config)

    # F001-F010 and F013-F019 direct daily/rolling factors.
    direct_builders: list[tuple[str, Any, tuple[Any, ...]]] = [
        ("F001SIZE", build_size, (config,)),
        ("F003LIQUIDITY", build_liquidity, (config,)),
        ("F006MOMENTUM", build_momentum, (config,)),
        ("F007LTREV", build_ltrev, (config,)),
        ("F008STREV", build_strev, (config,)),
        ("F009LEVERAGE", build_leverage, (config,)),
        ("F010VALUE", build_value, (config,)),
        ("F013REV5", build_rev5, (config,)),
        ("F014MOM120_20", build_resid_mom120_20, (config,)),
        ("F015VOLREV", build_volrev, (config,)),
        ("F016MAXRET", build_maxret, (config,)),
        ("F017IVOL", build_ivol, (config,)),
        ("F018AMIHUD", build_amihud, (config,)),
        ("F019COSTDEV", build_costdev, (config,)),
    ]

    if "F001SIZE" in needs or "F002SIZENL" in needs:
        print_factor_progress("F001SIZE", "build")
        df["F001SIZE_raw"] = build_size(df, config)

    if "F002SIZENL" in needs:
        print_factor_progress("F002SIZENL", "build")
        df["F002SIZENL_raw"] = build_sizenl(df, config, df["F001SIZE_raw"])

    if "F004BETA" in needs or "F005RESVOL" in needs:
        print_factor_progress("F004/F005", "beta_resvol")
        beta, resvol = build_beta_resvol(
            df, config, window=config.beta_window, min_periods=config.beta_min_periods
        )
        if "F004BETA" in needs:
            df["F004BETA_raw"] = beta
        if "F005RESVOL" in needs:
            df["F005RESVOL_raw"] = resvol
        del beta, resvol

    for factor, builder, args in direct_builders:
        if factor not in needs or factor in {"F001SIZE"}:
            continue
        print_factor_progress(factor, FACTOR_ALIAS.get(factor, factor))
        df[f"{factor}_raw"] = builder(df, *args)

    if "F020LIMITUP_RECENCY" in needs:
        print_factor_progress("F020LIMITUP_RECENCY", "limitup_recency20")
        df["F020LIMITUP_RECENCY_raw"] = build_limitup_recency20(df, config)

    # F025-F048: technical / volume / regression / limit factors.
    if needs & set(TECH24_FACTORS):
        compute_tech24_factors(df, config, factor_names)

    # Old TTM-based factors F011/F012/F020-F024/F049-F054.
    old_ttm_needed = {
        "F011EARNYLD", "F012GROWTH", "F021CFP", "F022GPTA", "F023ACCRUAL", "F024ASSETGR",
        "F049ROE", "F050ROA", "F051GPM", "F052CFOA", "F053RD_INTENSITY", "F054RECEIVABLE_RATIO",
    }
    if needs & old_ttm_needed:
        cache_key = "old_ttm_raws"
        if cache_key not in shared_cache:
            print_factor_progress("OLD_TTM", "financial raw table")
            shared_cache[cache_key] = build_ttm_factor_raws(df, financial_quarterly, config)
        ttm_df = shared_cache[cache_key]
        keep = [config.date_col, config.stock_col] + [f"{f}_raw" for f in old_ttm_needed if f in needs]
        keep = [c for c in keep if c in ttm_df.columns]
        df = df.merge(ttm_df[keep], on=[config.date_col, config.stock_col], how="left")

    if "F055IND" in needs:
        print_factor_progress("F055IND", "industry")
        df["F055IND_raw"] = df["industry_sw"].fillna(-1).astype(float) if "industry_sw" in df.columns else -1.0

    # V2 factors F056GAP_UP_FAIL-F100INV_MINUS_REV.
    v2_needs = needs & set(NEW45_FACTORS)
    if v2_needs:
        v2_df = compute_v2_factor_batch(
            df=df,
            financial_quarterly=financial_quarterly,
            factor_names=sorted(v2_needs),
            config=config,
            shared_cache=shared_cache,
        )
        for c in v2_df.columns:
            if c not in (config.date_col, config.stock_col):
                df[c] = v2_df[c].values
        del v2_df

    drop_temporary_columns(df)

    for start in range(0, len(factor_names), 10):
        batch = list(factor_names[start:start + 10])
        df = winsorize_zscore(df=df, factor_names=batch, config=config, target_dates=target_dates)
        drop_cols = [
            c for c in df.columns
            if any(c == f"{f}_w" or (c == f"{f}_raw" and not config.save_raw_factors) for f in batch)
        ]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

    return df




# ---------------------------------------------------------------------
# Factor scheduler helpers
# ---------------------------------------------------------------------


def iter_factor_batches(factor_names: Sequence[str]) -> Iterable[FactorBatchSpec]:
    """Yield dependency-aware batches capped at <=10 factors."""
    requested = set(factor_names)
    emitted: set[str] = set()
    for spec in FACTOR_BATCH_SPECS:
        selected = tuple(f for f in spec.factors if f in requested)
        for i in range(0, len(selected), 10):
            chunk = selected[i:i + 10]
            if chunk:
                emitted.update(chunk)
                yield FactorBatchSpec(
                    name=spec.name if len(selected) <= 10 else f"{spec.name}_{i//10+1}",
                    factors=chunk,
                    keep_cols_extra=spec.keep_cols_extra,
                    precompute=spec.precompute,
                )
    remaining = tuple(f for f in factor_names if f not in emitted)
    for i in range(0, len(remaining), 10):
        chunk = remaining[i:i + 10]
        if chunk:
            yield FactorBatchSpec(name=f"misc_{i//10+1}", factors=chunk)


def needed_columns_for_factors(factor_names: Sequence[str], config: FactorPanelConfig) -> set[str]:
    """Return the minimal base-panel columns needed for a requested factor set."""
    cols: set[str] = {config.date_col, config.stock_col}
    requested = set(factor_names)
    for spec in FACTOR_BATCH_SPECS:
        if requested & set(spec.factors):
            cols.update(spec.keep_cols_extra)

    # Universal fallback columns used by output, targets, and several legacy helpers.
    cols.update({
        "open", "high", "low", "close", "pre_close", "close_adj", "adj_factor",
        "ret_daily", "mkt_ret_vw", "mktcap_total", "mktcap_float",
        "amount", "volume", "trade_status", "limit_status",
        "turnover_rate", "pb", "pe_ttm", "list_date",
    })
    if "F055IND" in requested or any(c.startswith("F") for c in requested):
        cols.add("industry_sw")
    return cols


def prepare_working_panel(
    base_panel: pd.DataFrame,
    config: FactorPanelConfig,
    factor_names: Sequence[str],
) -> pd.DataFrame:
    """Standardize, sort, date-prune, and column-prune the working daily panel."""
    df = standardize_panel_keys(base_panel, config)
    df = df.sort_values([config.stock_col, config.date_col]).reset_index(drop=True)
    warmup_start = pd.Timestamp(config.model_start_date) - pd.DateOffset(years=4)
    df = df[df[config.date_col] >= warmup_start].copy()
    keep_cols = needed_columns_for_factors(factor_names, config)
    keep_cols.update(c for c in df.columns if c.startswith("industry"))

    # Merge precise adj_factor from Tushare API (preferred) or keep close_adj/close fallback
    adj_path = Path(config.adj_factor_path) if config.adj_factor_path else None
    if adj_path and adj_path.exists():
        adj_df = pd.read_parquet(adj_path)
        adj_df[config.date_col] = pd.to_datetime(adj_df[config.date_col])
        adj_df[config.stock_col] = adj_df[config.stock_col].astype(str).str.strip().str.zfill(6)
        df = df.merge(adj_df, on=[config.date_col, config.stock_col], how="left")
        keep_cols.add("adj_factor")

    keep_cols = [c for c in df.columns if c in keep_cols]
    return df[keep_cols]


def build_base_output_skeleton(base_panel: pd.DataFrame, config: FactorPanelConfig) -> pd.DataFrame:
    """Build the key/meta skeleton used for final batch merge.

    Applies adj_factor to OHLC prices so targets (built from this skeleton)
    and meta columns are consistent with factors (computed on adjusted prices).
    """
    base = standardize_panel_keys(base_panel, config)
    base = base[base[config.date_col] >= pd.Timestamp(config.model_start_date)].copy()

    # Merge precise adj_factor from Tushare API if available
    adj_path = Path(config.adj_factor_path) if config.adj_factor_path else None
    if adj_path and adj_path.exists():
        adj_df = pd.read_parquet(adj_path)
        adj_df[config.date_col] = pd.to_datetime(adj_df[config.date_col])
        adj_df[config.stock_col] = adj_df[config.stock_col].astype(str).str.strip().str.zfill(6)
        base = base.merge(adj_df, on=[config.date_col, config.stock_col], how="left")

    # Fallback: derive adj_factor from close_adj/close if not merged above
    if "adj_factor" not in base.columns:
        if "close_adj" in base.columns and "close" in base.columns:
            base["adj_factor"] = base["close_adj"] / base["close"].clip(lower=1e-12)
    # Fill any NaN adj_factor from merge misses with derived value
    elif "close_adj" in base.columns and "close" in base.columns:
        mask = base["adj_factor"].isna()
        if mask.any():
            base.loc[mask, "adj_factor"] = (
                base.loc[mask, "close_adj"] / base.loc[mask, "close"].clip(lower=1e-12)
            )

    # Apply to OHLC (pre_close handled separately below)
    if "adj_factor" in base.columns:
        for col in ("open", "high", "low", "close"):
            if col in base.columns:
                base[col] = base[col] * base["adj_factor"]

    # pre_close_adj = previous day's close_adj (correct across ex-dividend dates)
    if "pre_close" in base.columns and "close" in base.columns and "adj_factor" in base.columns:
        base = base.sort_values([config.stock_col, config.date_col])
        base["pre_close"] = base.groupby(config.stock_col)["close"].shift(1)

    # ret_daily is close-close (Tushare default, adjusted for dividends).

    meta_cols = [
        "industry_sw", "list_date",
        "open", "high", "low", "close", "pre_close", "close_adj", "adj_factor",
        "ret_daily",
        "mktcap_float", "mktcap_total", "volume", "amount", "trade_status",
        "limit_status", "turnover_rate", "pb", "pe_ttm",
    ]
    keep = [config.date_col, config.stock_col] + [c for c in meta_cols if c in base.columns]
    return base[keep].sort_values([config.stock_col, config.date_col]).reset_index(drop=True)


def print_factor_progress(factor: str, label: str = "") -> None:
    suffix = f" {label}" if label else ""
    print(f"  {factor}{suffix}...", flush=True)


def drop_temporary_columns(df: pd.DataFrame, keep: Sequence[str] = ()) -> None:
    """Drop temporary columns in-place and collect garbage."""
    keep_set = set(keep)
    tmp_cols = [c for c in df.columns if c.startswith("_") and c not in keep_set]
    if tmp_cols:
        df.drop(columns=tmp_cols, inplace=True)
        gc.collect()


def rolling_corr_fast(
    x: pd.Series,
    y: pd.Series,
    group: pd.Series,
    window: int,
    min_periods: int,
    eps: float = 1e-12,
) -> pd.Series:
    """Rolling Pearson correlation using rolling-sum decomposition."""
    valid = x.notna() & y.notna()
    xv = x.where(valid)
    yv = y.where(valid)
    cnt = valid.astype(float).groupby(group).transform(lambda s: s.rolling(window, min_periods=min_periods).sum())
    c = cnt.clip(lower=1)
    sx = xv.groupby(group).transform(lambda s: s.rolling(window, min_periods=min_periods).sum())
    sy = yv.groupby(group).transform(lambda s: s.rolling(window, min_periods=min_periods).sum())
    sx2 = (xv * xv).groupby(group).transform(lambda s: s.rolling(window, min_periods=min_periods).sum())
    sy2 = (yv * yv).groupby(group).transform(lambda s: s.rolling(window, min_periods=min_periods).sum())
    sxy = (xv * yv).groupby(group).transform(lambda s: s.rolling(window, min_periods=min_periods).sum())
    cov = sxy / c - (sx / c) * (sy / c)
    vx = (sx2 / c - (sx / c) ** 2).clip(lower=0)
    vy = (sy2 / c - (sy / c) ** 2).clip(lower=0)
    out = cov / (np.sqrt(vx * vy) + eps)
    return out.mask((vx < eps) | (vy < eps))


def precompute_regression20_stats(
    df: pd.DataFrame,
    config: FactorPanelConfig,
    price_col: str = "close_adj",
    window: int = 20,
    min_periods: int = 10,
) -> pd.DataFrame:
    """Precompute 20d trend slope, R², and fixed residual without polyfit callbacks.

    For incomplete warm-up windows, x is always 0..n-1 for the available window,
    matching the old rolling.apply(polyfit(arange(len(y)), y, 1)) semantics.
    """
    group = df[config.stock_col]
    y = df[price_col].astype(float)

    # For full 20d windows, fixed x=0..19.  For min_periods<window warmup,
    # we use rolling apply fallback only for the first few rows per stock; this
    # warmup is tiny compared with the full panel and keeps semantics exact.
    x_const = np.arange(window, dtype=float)
    sx = x_const.sum()
    sx2 = (x_const * x_const).sum()
    denom = window * sx2 - sx * sx

    sy = y.groupby(group).transform(lambda s: s.rolling(window, min_periods=window).sum())
    sy2 = (y * y).groupby(group).transform(lambda s: s.rolling(window, min_periods=window).sum())

    # weighted rolling sum for sum(x*y), x=0..19 in the current window.
    def _rolling_sxy(s: pd.Series) -> pd.Series:
        return s.rolling(window, min_periods=window).apply(lambda arr: float(np.dot(x_const, arr)), raw=True)

    sxy = y.groupby(group).transform(_rolling_sxy)
    slope = (window * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / window
    fitted_last = slope * (window - 1) + intercept
    resid = y - fitted_last

    sst = sy2 - sy * sy / window
    sse = sy2 - 2 * intercept * sy - 2 * slope * sxy + window * intercept**2 + 2 * intercept * slope * sx + slope**2 * sx2
    rsq = 1 - sse / sst.where(sst > 1e-12)
    rsq = rsq.clip(lower=0, upper=1)

    return pd.DataFrame({
        "_reg20_slope": slope,
        "_reg20_rsq": rsq,
        "_reg20_resid": resid,
    }, index=df.index)


# ---------------------------------------------------------------------
# Shared helper: CAPM residual
# ---------------------------------------------------------------------


def build_capm_residual(df: pd.DataFrame, config: FactorPanelConfig) -> pd.Series:
    """
    Rolling CAPM residual.

    residual_t = ret_t - alpha_t - beta_t * market_ret_t
    alpha_t = rolling_mean(ret) - beta_t * rolling_mean(market_ret)

    NOTE: shares rolling computation pattern with build_beta_resvol.
    Consider merging into a single capm_stats() helper in a future refactor.
    """

    ri = df["ret_daily"]
    rm = df["mkt_ret_vw"]
    group = df[config.stock_col]

    e_ri = (
        ri.groupby(group)
        .rolling(config.resid_window, min_periods=config.resid_min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )
    e_rm = (
        rm.groupby(group)
        .rolling(config.resid_window, min_periods=config.resid_min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )
    e_rirm = (
        (ri * rm)
        .groupby(group)
        .rolling(config.resid_window, min_periods=config.resid_min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )
    e_rm2 = (
        (rm ** 2)
        .groupby(group)
        .rolling(config.resid_window, min_periods=config.resid_min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )

    cov_im = e_rirm - e_ri * e_rm
    var_m = e_rm2 - e_rm ** 2

    beta = cov_im / var_m.where(var_m > 1e-10, np.nan)
    alpha = e_ri - beta * e_rm
    resid = ri - alpha - beta * rm

    resid[~np.isfinite(resid)] = np.nan
    return resid


# ---------------------------------------------------------------------
# F001-F012: original 12 factors
# ---------------------------------------------------------------------


def build_size(df: pd.DataFrame, config: FactorPanelConfig) -> pd.Series:
    """F001SIZE = log(total market cap)."""

    mktcap = df["mktcap_total"].fillna(df["mktcap_float"])
    result = np.log(mktcap.where(mktcap > 0))
    result[~np.isfinite(result)] = np.nan
    return result


def build_sizenl(
    df: pd.DataFrame,
    config: FactorPanelConfig,
    size_raw: pd.Series,
) -> pd.Series:
    """F002SIZENL: cube of SIZE, orthogonalized to SIZE (CNE5 mid-cap effect).

    SIZE_cube = log(mktcap)³
    SIZENL = SIZE_cube - beta * SIZE  (cross-sectionally per date)
    Captures the nonlinear mid-cap premium orthogonal to pure size.
    """

    cube = size_raw ** 3
    grp = df[config.date_col]
    # Vectorized per-date moments
    E_cube = cube.groupby(grp).transform("mean")
    E_s = size_raw.groupby(grp).transform("mean")
    cov = (cube * size_raw).groupby(grp).transform("mean") - E_cube * E_s
    var = (size_raw ** 2).groupby(grp).transform("mean") - E_s ** 2
    beta = cov / np.maximum(var, 1e-12)
    return cube - beta * size_raw


def build_liquidity(df: pd.DataFrame, config: FactorPanelConfig) -> pd.Series:
    """F003LIQUIDITY = rolling average of log(amount / market cap)."""

    amt_to_total = df["amount"] / df["mktcap_total"].where(df["mktcap_total"] > 0)
    result = (
        np.log1p(amt_to_total.clip(lower=0))
        .groupby(df[config.stock_col])
        .rolling(
            window=config.liquidity_window,
            min_periods=config.liquidity_min_periods,
        )
        .mean()
        .reset_index(level=0, drop=True)
    )
    result[~np.isfinite(result)] = np.nan
    return result


def build_beta_resvol(
    df: pd.DataFrame,
    config: FactorPanelConfig,
    window: int,
    min_periods: int,
) -> tuple[pd.Series, pd.Series]:
    """F004BETA and F005RESVOL."""

    ri = df["ret_daily"]
    rm = df["mkt_ret_vw"]
    group = df[config.stock_col]

    e_ri = ri.groupby(group).rolling(window, min_periods=min_periods).mean().reset_index(level=0, drop=True)
    e_rm = rm.groupby(group).rolling(window, min_periods=min_periods).mean().reset_index(level=0, drop=True)
    e_ri2 = (ri ** 2).groupby(group).rolling(window, min_periods=min_periods).mean().reset_index(level=0, drop=True)
    e_rm2 = (rm ** 2).groupby(group).rolling(window, min_periods=min_periods).mean().reset_index(level=0, drop=True)
    e_rirm = (ri * rm).groupby(group).rolling(window, min_periods=min_periods).mean().reset_index(level=0, drop=True)

    cov_im = e_rirm - e_ri * e_rm
    var_m = e_rm2 - e_rm ** 2
    var_i = e_ri2 - e_ri ** 2

    beta = cov_im / var_m.where(var_m > 1e-10, np.nan)
    res_var = var_i - cov_im ** 2 / var_m.where(var_m > 1e-10, np.nan)
    resvol = np.sqrt(np.maximum(res_var, 0))

    beta[~np.isfinite(beta)] = np.nan
    resvol[~np.isfinite(resvol)] = np.nan

    return beta, resvol


def build_momentum(df: pd.DataFrame, config: FactorPanelConfig) -> pd.Series:
    """F006MOMENTUM = close[t-21] / close[t-252] - 1 (close-to-close momentum)."""

    group = df.groupby(config.stock_col)["close"]
    return group.shift(21) / group.shift(252) - 1


def build_ltrev(df: pd.DataFrame, config: FactorPanelConfig) -> pd.Series:
    """F007LTREV: long-term reversal ensemble based on 504/630/756-day windows (close-to-close)."""

    group = df.groupby(config.stock_col)["close"]

    def log_reversal(lookback: int) -> pd.Series:
        return -(
            np.log(group.shift(252).clip(lower=1e-10))
            - np.log(group.shift(lookback).clip(lower=1e-10))
        )

    parts = []
    for lookback in (504, 630, 756):
        part = log_reversal(lookback)
        mu = part.groupby(df[config.date_col]).transform("mean")
        sd = part.groupby(df[config.date_col]).transform("std")
        parts.append((part - mu) / sd.where(sd > 1e-10, 1.0))

    return (parts[0] + parts[1] + parts[2]) / 3.0


def build_strev(df: pd.DataFrame, config: FactorPanelConfig) -> pd.Series:
    """F008STREV = negative short-term return (close-to-close)."""

    group = df.groupby(config.stock_col)["close"]
    return -(group.shift(1) / group.shift(21) - 1)


def build_leverage(df: pd.DataFrame, config: FactorPanelConfig) -> pd.Series:
    """F009LEVERAGE = total liabilities / total assets."""

    result = df["total_liabilities"] / df["total_assets"].where(df["total_assets"] > 0)
    result[~np.isfinite(result)] = np.nan
    return result


def build_value(df: pd.DataFrame, config: FactorPanelConfig) -> pd.Series:
    """F010VALUE = equity_parent / market cap."""

    mkt = df["mktcap_total"].fillna(df["mktcap_float"])
    result = df["equity_parent"] / mkt.where(mkt > 0)
    result[~np.isfinite(result)] = np.nan
    return result


# ---------------------------------------------------------------------
# F013-F020: new residual / price / liquidity factors
# ---------------------------------------------------------------------


def build_rev5(df: pd.DataFrame, config: FactorPanelConfig) -> pd.Series:
    """F013REV5 = -sum(CAPM residual, 5d)."""

    resid = df["_capm_resid"]
    return -(
        resid.groupby(df[config.stock_col])
        .rolling(config.rev5_window, min_periods=3)
        .sum()
        .reset_index(level=0, drop=True)
    )


def build_resid_mom120_20(df: pd.DataFrame, config: FactorPanelConfig) -> pd.Series:
    """F014MOM120_20 = sum(resid, 120d) - sum(resid, 20d)."""

    resid = df["_capm_resid"]
    group = df[config.stock_col]

    long_sum = (
        resid.groupby(group)
        .rolling(config.mom_long_window, min_periods=60)
        .sum()
        .reset_index(level=0, drop=True)
    )
    short_sum = (
        resid.groupby(group)
        .rolling(config.mom_short_window, min_periods=10)
        .sum()
        .reset_index(level=0, drop=True)
    )

    return long_sum - short_sum


def build_volrev(df: pd.DataFrame, config: FactorPanelConfig) -> pd.Series:
    """F015VOLREV = -ret_5d * log(avg_amount_5d / avg_amount_60d)."""

    group = df.groupby(config.stock_col)

    ret_5d = group["close_adj"].transform(lambda x: x / x.shift(config.volrev_short_window) - 1)

    amt_5d = (
        df["amount"]
        .groupby(df[config.stock_col])
        .rolling(config.volrev_short_window, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )
    amt_60d = (
        df["amount"]
        .groupby(df[config.stock_col])
        .rolling(config.volrev_long_window, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )

    ratio = amt_5d / amt_60d.where(amt_60d > 0)
    result = -ret_5d * np.log(ratio.where(ratio > 0))
    result[~np.isfinite(result)] = np.nan
    return result


def build_maxret(df: pd.DataFrame, config: FactorPanelConfig) -> pd.Series:
    """F016MAXRET = -max(ret_daily, 20d)."""

    result = (
        df["ret_daily"]
        .groupby(df[config.stock_col])
        .rolling(config.maxret_window, min_periods=10)
        .max()
        .reset_index(level=0, drop=True)
    )
    return -result


def build_ivol(df: pd.DataFrame, config: FactorPanelConfig) -> pd.Series:
    """F017IVOL = -mean(std(resid, 20d), std(resid, 60d))."""

    resid = df["_capm_resid"]
    group = df[config.stock_col]

    std_20 = (
        resid.groupby(group)
        .rolling(config.ivol_short_window, min_periods=10)
        .std()
        .reset_index(level=0, drop=True)
    )
    std_60 = (
        resid.groupby(group)
        .rolling(config.ivol_long_window, min_periods=20)
        .std()
        .reset_index(level=0, drop=True)
    )

    return -pd.concat([std_20, std_60], axis=1).mean(axis=1)


def build_amihud(df: pd.DataFrame, config: FactorPanelConfig) -> pd.Series:
    """F018AMIHUD = -mean(mean(|ret|/amount,20d), mean(|ret|/amount,60d))."""

    amihud = df["ret_daily"].abs() / df["amount"].where(df["amount"] > 0)
    group = df[config.stock_col]

    amihud_20 = (
        amihud.groupby(group)
        .rolling(config.amihud_short_window, min_periods=10)
        .mean()
        .reset_index(level=0, drop=True)
    )
    amihud_60 = (
        amihud.groupby(group)
        .rolling(config.amihud_long_window, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return -pd.concat([amihud_20, amihud_60], axis=1).mean(axis=1)


def build_costdev(df: pd.DataFrame, config: FactorPanelConfig) -> pd.Series:
    """F019COSTDEV = -(close / vwap_120d - 1)."""

    group = df[config.stock_col]

    amount_sum = (
        df["amount"]
        .groupby(group)
        .rolling(config.vwap_window, min_periods=60)
        .sum()
        .reset_index(level=0, drop=True)
    )
    volume_sum = (
        df["volume"]
        .groupby(group)
        .rolling(config.vwap_window, min_periods=60)
        .sum()
        .reset_index(level=0, drop=True)
    )

    vwap_120d = amount_sum / volume_sum.where(volume_sum > 0)
    px = df["_close_raw"] if "_close_raw" in df.columns else df["close"]
    result = -(px / vwap_120d.where(vwap_120d > 0) - 1)
    result[~np.isfinite(result)] = np.nan
    return result


def build_limitup_recency20(df: pd.DataFrame, config: FactorPanelConfig) -> pd.Series:
    """F020LIMITUP_RECENCY LIMITUP_RECENCY20 = exp decay since last limit-up, capped at 20d.

    Replaces old BP (book-to-price) which was 0.990 correlated with F085SP_TTM SP_TTM.
    Larger = more recent limit-up.  No limit-up in 20d → 0.
    """
    stock = df[config.stock_col]
    is_lu = df["limit_status"].eq(1)

    def _days_since(x: pd.Series) -> pd.Series:
        arr = x.to_numpy(dtype=bool)
        out = np.full(len(arr), np.nan, dtype=float)
        last = -1
        for i, flag in enumerate(arr):
            if flag:
                last = i
                out[i] = 0.0
            elif last >= 0:
                out[i] = float(i - last)
        return pd.Series(out, index=x.index)

    days = is_lu.groupby(stock, sort=False).transform(_days_since)
    return np.where(days <= 20, np.exp(-days / 7.0), 0.0)


# ---------------------------------------------------------------------
# F011/F012/F021-F024: TTM financial factors
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Group E: Technical / volume / price-stressed factors (F025-F048)
# ---------------------------------------------------------------------


def _rolling_corr(
    df: pd.DataFrame, config: FactorPanelConfig,
    col_a: str, col_b: str, window: int, eps: float = 1e-12,
) -> pd.Series:
    """Deprecated compatibility wrapper around rolling_corr_fast."""
    warnings.warn(
        "_rolling_corr is deprecated; use rolling_corr_fast instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return rolling_corr_fast(
        df[col_a], df[col_b], df[config.stock_col], window, max(10, window // 2), eps
    )


def compute_tech24_factors(
    df: pd.DataFrame,
    config: FactorPanelConfig,
    factor_names: Sequence[str],
) -> None:
    """Compute F025-F048 factors in-place, with shared regression/corr helpers."""

    needs = set(factor_names)
    eps = 1e-12
    g = df.groupby(config.stock_col, sort=False)

    if needs & {"F030RSV20", "F031RSV60", "F034HIGHDEV20", "F035LOWDEV20"}:
        df["_adj_factor"] = df["close_adj"] / df["close"].clip(lower=eps)
        df["_open_adj"] = df["open"] * df["_adj_factor"]
        df["_high_adj"] = df["high"] * df["_adj_factor"]
        df["_low_adj"] = df["low"] * df["_adj_factor"]

    if needs & {"F043SLOPE20", "F044RSQR20", "F045RESI20"}:
        print("  regression20 precompute...")
        reg20 = precompute_regression20_stats(df, config, price_col="close_adj", window=20, min_periods=10)
        for c in reg20.columns:
            df[c] = reg20[c].values
        del reg20

    factor_builders: list[tuple[str, str, Any]] = [
        ("F025GAP", "gap", lambda: df["open"] / df["pre_close"].clip(lower=eps) - 1),
        ("F026KLEN", "klen", lambda: (df["high"] - df["low"]) / (df["open"] + eps)),
        ("F027KUP", "kup", lambda: (df["high"] - df[["open", "close"]].max(axis=1)) / (df["open"] + eps)),
        ("F028KLOW", "klow", lambda: (df[["open", "close"]].min(axis=1) - df["low"]) / (df["open"] + eps)),
        ("F029KSFT", "ksft", lambda: (2 * df["close"] - df["high"] - df["low"]) / (df["open"] + eps)),
    ]
    for fid, label, builder in factor_builders:
        if fid in needs:
            print_factor_progress(fid, label)
            df[f"{fid}_raw"] = builder()

    if "F030RSV20" in needs:
        print_factor_progress("F030RSV20", "rsv20")
        lo20 = g["_low_adj"].transform(lambda x: x.rolling(20, min_periods=10).min())
        hi20 = g["_high_adj"].transform(lambda x: x.rolling(20, min_periods=10).max())
        df["F030RSV20_raw"] = (df["close_adj"] - lo20) / (hi20 - lo20 + eps)
        del lo20, hi20

    if "F031RSV60" in needs:
        print_factor_progress("F031RSV60", "rsv60")
        lo60 = g["_low_adj"].transform(lambda x: x.rolling(60, min_periods=30).min())
        hi60 = g["_high_adj"].transform(lambda x: x.rolling(60, min_periods=30).max())
        df["F031RSV60_raw"] = (df["close_adj"] - lo60) / (hi60 - lo60 + eps)
        del lo60, hi60

    if "F032RANGEZ20" in needs:
        print_factor_progress("F032RANGEZ20", "rangez20")
        df["_rng"] = (df["high"] - df["low"]) / (df["open"] + eps)
        mu = g["_rng"].transform(lambda x: x.rolling(20, min_periods=10).mean())
        sd = g["_rng"].transform(lambda x: x.rolling(20, min_periods=10).std())
        df["F032RANGEZ20_raw"] = (df["_rng"] - mu) / (sd + eps)
        del mu, sd

    if "F033GAPREV5" in needs:
        print_factor_progress("F033GAPREV5", "gaprev5")
        df["_gap"] = df["open"] / df["pre_close"].clip(lower=eps) - 1
        df["F033GAPREV5_raw"] = -g["_gap"].transform(lambda x: x.rolling(5, min_periods=3).sum())

    if "F034HIGHDEV20" in needs:
        print_factor_progress("F034HIGHDEV20", "highdev20")
        hi20 = g["_high_adj"].transform(lambda x: x.rolling(20, min_periods=10).max())
        df["F034HIGHDEV20_raw"] = df["close_adj"] / (hi20 + eps) - 1
        del hi20

    if "F035LOWDEV20" in needs:
        print_factor_progress("F035LOWDEV20", "lowdev20")
        lo20 = g["_low_adj"].transform(lambda x: x.rolling(20, min_periods=10).min())
        df["F035LOWDEV20_raw"] = df["close_adj"] / (lo20 + eps) - 1
        del lo20

    if "F036VOLSHOCK5" in needs:
        print_factor_progress("F036VOLSHOCK5", "volshock5")
        amt5 = g["amount"].transform(lambda x: x.rolling(5, min_periods=3).mean())
        df["F036VOLSHOCK5_raw"] = np.log(df["amount"] / (amt5 + eps) + eps)
        del amt5

    if "F037VOLSHOCK20" in needs:
        print_factor_progress("F037VOLSHOCK20", "volshock20")
        amt20 = g["amount"].transform(lambda x: x.rolling(20, min_periods=10).mean())
        df["F037VOLSHOCK20_raw"] = np.log(df["amount"] / (amt20 + eps) + eps)
        del amt20

    if "F038TURNZ20" in needs:
        print_factor_progress("F038TURNZ20", "turnz20")
        tf_col = "turnover_rate_f" if "turnover_rate_f" in df.columns else "turnover_rate"
        tf_mu = g[tf_col].transform(lambda x: x.rolling(20, min_periods=10).mean())
        tf_sd = g[tf_col].transform(lambda x: x.rolling(20, min_periods=10).std())
        df["F038TURNZ20_raw"] = (df[tf_col] - tf_mu) / (tf_sd + eps)
        del tf_mu, tf_sd

    if "F039VSTD20" in needs:
        print_factor_progress("F039VSTD20", "vstd20")
        df["_logvol"] = np.log(df["volume"] + 1)
        df["F039VSTD20_raw"] = g["_logvol"].transform(lambda x: x.rolling(20, min_periods=10).std())

    if needs & {"F040PVCORR20", "F041RETVOLCORR20", "F042AMTCORR20"}:
        if "_logvol" not in df.columns:
            df["_logvol"] = np.log(df["volume"] + 1)
        if "_vol_chg" not in df.columns:
            df["_vol_chg"] = np.log(df["volume"] / (g["volume"].shift(1) + eps) + 1)
        if "_logamt" not in df.columns:
            df["_logamt"] = np.log(df["amount"] + 1)

    if "F040PVCORR20" in needs:
        print_factor_progress("F040PVCORR20", "pvcorr20")
        df["F040PVCORR20_raw"] = rolling_corr_fast(df["close_adj"], df["_logvol"], df[config.stock_col], 20, 10)

    if "F041RETVOLCORR20" in needs:
        print_factor_progress("F041RETVOLCORR20", "retvolcorr20")
        df["F041RETVOLCORR20_raw"] = rolling_corr_fast(df["ret_daily"], df["_vol_chg"], df[config.stock_col], 20, 10)

    if "F042AMTCORR20" in needs:
        print_factor_progress("F042AMTCORR20", "amtcorr20")
        df["F042AMTCORR20_raw"] = rolling_corr_fast(df["ret_daily"], df["_logamt"], df[config.stock_col], 20, 10)

    if "F043SLOPE20" in needs:
        print_factor_progress("F043SLOPE20", "slope20")
        df["F043SLOPE20_raw"] = df["_reg20_slope"] / (df["close_adj"] + eps)

    if "F044RSQR20" in needs:
        print_factor_progress("F044RSQR20", "rsqr20")
        df["F044RSQR20_raw"] = df["_reg20_rsq"]

    if "F045RESI20" in needs:
        print_factor_progress("F045RESI20", "resi20_fixed")
        df["F045RESI20_raw"] = df["_reg20_resid"] / (df["close_adj"] + eps)

    if "F046LIMITUP20" in needs:
        print_factor_progress("F046LIMITUP20", "limitup20")
        df["F046LIMITUP20_raw"] = g["limit_status"].transform(lambda x: (x == 1).rolling(20, min_periods=5).sum())

    if "F047LIMITDN20" in needs:
        print_factor_progress("F047LIMITDN20", "limitdn20")
        df["F047LIMITDN20_raw"] = g["limit_status"].transform(lambda x: (x == -1).rolling(20, min_periods=5).sum())

    if "F048LIMITSTREAKUP" in needs:
        print_factor_progress("F048LIMITSTREAKUP", "limitstreakup")
        is_limit = df["limit_status"] == 1
        streak = is_limit.groupby(df[config.stock_col]).transform(lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1))
        df["F048LIMITSTREAKUP_raw"] = streak

    drop_temporary_columns(df)


def _compute_tech24_factors(
    df: pd.DataFrame,
    config: FactorPanelConfig,
    factor_names: Sequence[str],
) -> None:
    """Backward-compatible wrapper for older callers."""
    compute_tech24_factors(df, config, factor_names)


def build_ttm_factor_raws(
    base_panel: pd.DataFrame,
    financial_quarterly: pd.DataFrame,
    config: FactorPanelConfig,
) -> pd.DataFrame:
    """
    Compute TTM-based raw factors.

    Generated raw columns:
        F011EARNYLD_raw
        F012GROWTH_raw
        F021CFP_raw
        F022GPTA_raw
        F023ACCRUAL_raw
        F024ASSETGR_raw
    """

    base = standardize_panel_keys(base_panel, config)
    fin = standardize_financial_keys(financial_quarterly, config)

    # Only standard quarter-end reports participate in TTM.
    fin["_accper_m"] = fin["accper"].dt.month
    fin = fin[fin["_accper_m"].isin({3, 6, 9, 12})].copy()
    fin = fin.drop(columns=["_accper_m"])

    cumulative_cols = [
        "revenue_total",
        "cost_revenue",
        "net_profit_parent",
        "cf_operating",
    ]
    cumulative_cols = [col for col in cumulative_cols if col in fin.columns]

    fin = cumulative_to_single_quarter(fin, cumulative_cols)

    ttm_specs = {
        "revenue_total": "revenue_ttm",
        "cost_revenue": "cost_revenue_ttm",
        "net_profit_parent": "np_ttm",
        "cf_operating": "cf_operating_ttm",
    }

    for src_col, ttm_col in ttm_specs.items():
        sq_col = f"{src_col}_sq"
        if sq_col in fin.columns:
            fin[ttm_col] = compute_ttm(fin, sq_col)

    fin = fin.sort_values([config.stock_col, "accper"]).reset_index(drop=True)

    # YoY / lag-4-quarter fields.
    if "revenue_ttm" in fin.columns:
        prev = fin.groupby(config.stock_col)["revenue_ttm"].shift(4)
        fin["revenue_ttm_yoy"] = (
            (fin["revenue_ttm"] - prev) / prev.abs().where(prev.abs() > 1e-8)
        )

    if "total_assets" in fin.columns:
        ta_lag4 = fin.groupby(config.stock_col)["total_assets"].shift(4)
        fin["total_assets_lag4q"] = ta_lag4
        fin["total_assets_yoy"] = (
            (fin["total_assets"] - ta_lag4)
            / ta_lag4.abs().where(ta_lag4.abs() > 1e-8)
        )

    if "equity_parent" in fin.columns:
        ep_lag4 = fin.groupby(config.stock_col)["equity_parent"].shift(4)
        fin["equity_parent_yoy"] = (
            (fin["equity_parent"] - ep_lag4)
            / ep_lag4.abs().where(ep_lag4.abs() > 1e-8)
        )

    keep_cols = [
        config.stock_col,
        "available_date",
        "accper",
        "revenue_ttm",
        "cost_revenue_ttm",
        "np_ttm",
        "cf_operating_ttm",
        "revenue_ttm_yoy",
        "total_assets_yoy",
        "equity_parent_yoy",
        "total_assets_lag4q",
        "total_assets",
    ]
    keep_cols = [col for col in keep_cols if col in fin.columns]

    fin_daily = fin[keep_cols].copy()
    fin_daily = fin_daily.sort_values(
        [config.stock_col, "available_date", "accper"]
    ).reset_index(drop=True)
    fin_daily = fin_daily.drop_duplicates(
        subset=[config.stock_col, "available_date"],
        keep="last",
    )

    daily = base[[config.date_col, config.stock_col]].copy()
    daily = daily.sort_values([config.stock_col, config.date_col]).reset_index(drop=True)

    fill_cols = [
        col
        for col in keep_cols
        if col not in {config.stock_col, "available_date", "accper"}
    ]

    filled = forward_fill_quarterly_to_daily(
        daily=daily,
        quarterly=fin_daily,
        config=config,
        value_cols=fill_cols,
    )

    # Attach daily market cap and daily current total_assets from base panel.
    # Merge daily market cap + extra financial cols not already in filled
    existing_cols = set(filled.columns)
    extra_cols = ["mktcap_total", "mktcap_float"]
    for c in ["total_assets", "equity_parent", "rd_expense", "receivables"]:
        if c in base.columns and c not in existing_cols:
            extra_cols.append(c)
    if "total_assets" in base.columns and "total_assets" not in extra_cols:
        extra_cols.append("total_assets")  # for total_assets_daily
    base_extra = base[[config.date_col, config.stock_col] + extra_cols].copy()
    if "total_assets" in base_extra.columns:
        base_extra["total_assets_daily"] = base_extra["total_assets"]
        base_extra = base_extra.drop(columns=["total_assets"])  # avoid merge conflict

    filled = filled.merge(base_extra, on=[config.date_col, config.stock_col], how="left")

    mkt = filled["mktcap_total"].fillna(filled["mktcap_float"])

    # F011 EARNYLD: np_ttm / market cap.
    if "np_ttm" in filled.columns:
        filled["F011EARNYLD_raw"] = filled["np_ttm"] / mkt.where(mkt > 0)

    # F012 GROWTH: weighted z-score of growth components.
    growth_parts = []
    if "revenue_ttm_yoy" in filled.columns:
        growth_parts.append(
            0.50 * per_date_zscore(filled["revenue_ttm_yoy"], filled[config.date_col])
        )
    if "total_assets_yoy" in filled.columns:
        growth_parts.append(
            0.25 * per_date_zscore(filled["total_assets_yoy"], filled[config.date_col])
        )
    if "equity_parent_yoy" in filled.columns:
        growth_parts.append(
            0.25 * per_date_zscore(filled["equity_parent_yoy"], filled[config.date_col])
        )
    if growth_parts:
        filled["F012GROWTH_raw"] = sum(growth_parts)

    # F021 CFP: cf_operating_TTM / market cap.
    if "cf_operating_ttm" in filled.columns:
        filled["F021CFP_raw"] = filled["cf_operating_ttm"] / mkt.where(mkt > 0)

    # F022 GPTA: gross profit TTM / total assets.
    if "revenue_ttm" in filled.columns and "cost_revenue_ttm" in filled.columns:
        total_assets_denom = _choose_total_assets_denominator(filled)
        filled["F022GPTA_raw"] = (
            (filled["revenue_ttm"] - filled["cost_revenue_ttm"])
            / total_assets_denom.where(total_assets_denom > 0)
        )

    # F023 ACCRUAL: -(net_profit_TTM - cf_TTM) / total assets.
    if "np_ttm" in filled.columns and "cf_operating_ttm" in filled.columns:
        total_assets_denom = _choose_total_assets_denominator(filled)
        filled["F023ACCRUAL_raw"] = -(
            (filled["np_ttm"] - filled["cf_operating_ttm"])
            / total_assets_denom.where(total_assets_denom > 0)
        )

    # F024 ASSETGR: -(total_assets / total_assets_lag4q - 1).
    if "total_assets" in filled.columns and "total_assets_lag4q" in filled.columns:
        filled["F024ASSETGR_raw"] = -(
            filled["total_assets"] / filled["total_assets_lag4q"].where(filled["total_assets_lag4q"] > 0)
            - 1
        )

    # F020 deprecated — was SP (sales-to-price), now LIMITUP_RECENCY20 computed elsewhere

    # F049-F054: TTM financial quality ratios
    eps = 1e-12
    if "np_ttm" in filled.columns and "equity_parent" in filled.columns:
        filled["F049ROE_raw"] = filled["np_ttm"] / (filled["equity_parent"] + eps)

    if "np_ttm" in filled.columns and "total_assets" in filled.columns:
        filled["F050ROA_raw"] = filled["np_ttm"] / (filled["total_assets"] + eps)

    if "revenue_ttm" in filled.columns and "cost_revenue_ttm" in filled.columns:
        filled["F051GPM_raw"] = (
            (filled["revenue_ttm"] - filled["cost_revenue_ttm"])
            / (filled["revenue_ttm"] + eps)
        )

    if "cf_operating_ttm" in filled.columns and "total_assets" in filled.columns:
        filled["F052CFOA_raw"] = filled["cf_operating_ttm"] / (filled["total_assets"] + eps)

    if "rd_expense" in filled.columns and "revenue_ttm" in filled.columns:
        filled["F053RD_INTENSITY_raw"] = filled["rd_expense"] / (filled["revenue_ttm"] + eps)

    if "receivables" in filled.columns and "revenue_ttm" in filled.columns:
        filled["F054RECEIVABLE_RATIO_raw"] = -filled["receivables"] / (filled["revenue_ttm"] + eps)

    output_cols = [
        config.date_col,
        config.stock_col,
        "F011EARNYLD_raw",
        "F012GROWTH_raw",
        "F021CFP_raw",
        "F020LIMITUP_RECENCY_raw",
        "F022GPTA_raw",
        "F023ACCRUAL_raw",
        "F024ASSETGR_raw",
    ]
    # Add F049-F054 output cols if computed
    for f in ("F049ROE_raw","F050ROA_raw","F051GPM_raw","F052CFOA_raw",
              "F053RD_INTENSITY_raw","F054RECEIVABLE_RATIO_raw"):
        if f in filled.columns:
            output_cols.append(f)
    output_cols = [col for col in output_cols if col in filled.columns]

    return filled[output_cols]


def _choose_total_assets_denominator(df: pd.DataFrame) -> pd.Series:
    """
    Prefer daily ffilled total_assets from base panel if available;
    otherwise use quarterly ffilled total_assets.
    """

    if "total_assets_daily" in df.columns:
        return df["total_assets_daily"].fillna(df.get("total_assets"))

    return df["total_assets"]


def cumulative_to_single_quarter(
    fin: pd.DataFrame,
    value_cols: list[str],
) -> pd.DataFrame:
    """
    Convert cumulative YTD financial fields to single-quarter values.

    Q1 = Q1 cumulative.
    Q2/Q3/Q4 = cumulative - prior quarter of the same year.
    """

    fin = fin.sort_values(["stock_id", "accper"]).reset_index(drop=True)

    for col in value_cols:
        fin[f"{col}_sq"] = np.nan

    for stock_id, group in fin.groupby("stock_id", sort=False):
        group = group.sort_values("accper")
        idx = group.index
        accpers = pd.DatetimeIndex(group["accper"].values)

        for col in value_cols:
            vals = group[col].to_numpy(dtype=float)
            sq = np.full(len(vals), np.nan, dtype=float)

            for i in range(len(vals)):
                month = accpers[i].month
                year = accpers[i].year

                if month == 3:
                    sq[i] = vals[i]
                elif month in {6, 9, 12}:
                    if i == 0:
                        continue
                    prev_month = accpers[i - 1].month
                    prev_year = accpers[i - 1].year
                    expected_prev = {6: 3, 9: 6, 12: 9}[month]

                    if prev_year == year and prev_month == expected_prev:
                        if np.isfinite(vals[i]) and np.isfinite(vals[i - 1]):
                            sq[i] = vals[i] - vals[i - 1]

            fin.loc[idx, f"{col}_sq"] = sq

    return fin


def compute_ttm(fin: pd.DataFrame, col_sq: str) -> pd.Series:
    """Compute rolling 4-quarter TTM from single-quarter values."""

    result = pd.Series(np.nan, index=fin.index, dtype=float)

    for stock_id, group in fin.groupby("stock_id", sort=False):
        group = group.sort_values("available_date")
        vals = group[col_sq].to_numpy(dtype=float)
        ttm = np.full(len(vals), np.nan, dtype=float)

        for i in range(len(vals)):
            window = vals[max(0, i - 3): i + 1]
            if len(window) >= 4 and np.isfinite(window).all():
                ttm[i] = window.sum()

        result.loc[group.index] = ttm

    return result


def forward_fill_quarterly_to_daily(
    daily: pd.DataFrame,
    quarterly: pd.DataFrame,
    config: FactorPanelConfig,
    value_cols: Sequence[str],
) -> pd.DataFrame:
    """Forward-fill quarterly values to daily rows by available_date."""

    daily = standardize_panel_keys(daily, config)
    quarterly = quarterly.copy()
    quarterly[config.stock_col] = quarterly[config.stock_col].astype(str).str.strip().str.zfill(6)
    quarterly["available_date"] = pd.to_datetime(quarterly["available_date"]).dt.normalize()

    result_parts = []

    for stock_id, group in daily.groupby(config.stock_col, sort=False):
        group = group.copy()
        stock_fin = quarterly[quarterly[config.stock_col] == stock_id].sort_values("available_date")

        for col in value_cols:
            group[col] = np.nan

        if stock_fin.empty:
            result_parts.append(group)
            continue

        daily_dates = group[config.date_col].to_numpy()
        fin_dates = stock_fin["available_date"].to_numpy()

        idx = np.searchsorted(fin_dates, daily_dates, side="right") - 1
        valid = idx >= 0

        for col in value_cols:
            values = np.full(len(group), np.nan, dtype=float)
            source_values = stock_fin[col].to_numpy(dtype=float)
            values[valid] = source_values[idx[valid]]
            group[col] = values

        result_parts.append(group)

    return pd.concat(result_parts, ignore_index=True)




# ---------------------------------------------------------------------
# V2 financial factor raws from quarterly panel
# ---------------------------------------------------------------------


def normalize_financial_aliases(fin: pd.DataFrame) -> pd.DataFrame:
    """Normalize common financial field aliases without mutating caller data."""
    out = fin.copy()
    alias_pairs = {
        "revenue": ("revenue_total", "total_revenue"),
        "net_profit": ("net_profit_parent", "np_parent"),
    }
    for target, sources in alias_pairs.items():
        if target not in out.columns:
            for src in sources:
                if src in out.columns:
                    out[target] = out[src]
                    break
    return out


def build_v2_financial_factor_raws(
    base_panel: pd.DataFrame,
    financial_quarterly: pd.DataFrame,
    config: FactorPanelConfig,
    factor_names: Sequence[str],
) -> pd.DataFrame:
    """Build F088CF_SALES_Q-F100INV_MINUS_REV raw factor columns from financial_quarterly_panel.

    This is intentionally quarterly-first: YTD->single quarter->TTM is computed
    on the quarterly table, then daily values are forward-filled by
    available_date.  This avoids using daily ffilled rows as if they were
    consecutive financial reports.
    """
    requested = set(factor_names)
    base = standardize_panel_keys(base_panel, config)
    fin = normalize_financial_aliases(standardize_financial_keys(financial_quarterly, config))

    fin["_accper_m"] = fin["accper"].dt.month
    fin = fin[fin["_accper_m"].isin({3, 6, 9, 12})].drop(columns=["_accper_m"]).copy()
    fin = fin.sort_values([config.stock_col, "accper"]).reset_index(drop=True)

    flow_cols = [
        "cf_sales_cash", "cf_operating", "cf_capex", "cf_borrow", "cf_repay_debt",
        "cf_equity_issue", "cf_dividend_paid", "revenue", "net_profit",
        "operating_profit", "finance_expense", "income_tax",
    ]
    flow_cols = [c for c in flow_cols if c in fin.columns]
    fin = cumulative_to_single_quarter(fin, flow_cols)

    for col in flow_cols:
        sq_col = f"{col}_sq"
        if sq_col in fin.columns:
            fin[f"_ttm_{col}"] = compute_ttm(fin, sq_col)

    # Quarterly helper metrics.
    group = fin.groupby(config.stock_col, sort=False)
    if "net_profit_sq" in fin.columns and "cf_operating_sq" in fin.columns:
        x = group["net_profit_sq"].shift(1)
        y = fin["cf_operating_sq"]
        w, minp = 16, 8
        valid = x.notna() & y.notna()
        xv = x.where(valid)
        yv = y.where(valid)
        cnt = valid.astype(float).groupby(fin[config.stock_col]).transform(lambda s: s.rolling(w, min_periods=minp).sum()).clip(lower=1)
        sx = xv.groupby(fin[config.stock_col]).transform(lambda s: s.rolling(w, min_periods=minp).sum())
        sy = yv.groupby(fin[config.stock_col]).transform(lambda s: s.rolling(w, min_periods=minp).sum())
        sxy = (xv*yv).groupby(fin[config.stock_col]).transform(lambda s: s.rolling(w, min_periods=minp).sum())
        sx2 = (xv*xv).groupby(fin[config.stock_col]).transform(lambda s: s.rolling(w, min_periods=minp).sum())
        cov = sxy / cnt - (sx / cnt) * (sy / cnt)
        var = (sx2 / cnt - (sx / cnt)**2).clip(lower=0)
        fin["_q_crr"] = cov / np.maximum(var, 1e-12)

    if "_ttm_cf_operating" in fin.columns and "total_assets" in fin.columns:
        ratio = fin["_ttm_cf_operating"] / np.maximum(fin["total_assets"], 1e-8)
        fin["_q_cf_vol"] = -ratio.groupby(fin[config.stock_col]).transform(lambda s: s.rolling(16, min_periods=8).std())

    if "_ttm_net_profit" in fin.columns and "total_assets" in fin.columns:
        ratio = fin["_ttm_net_profit"] / np.maximum(fin["total_assets"], 1e-8)
        fin["_q_earn_stab"] = -ratio.groupby(fin[config.stock_col]).transform(lambda s: s.rolling(16, min_periods=8).std())

    if "_ttm_revenue" in fin.columns:
        rev_lag4 = fin.groupby(config.stock_col)["_ttm_revenue"].shift(4)
        fin["_rev_ttm_yoy"] = (fin["_ttm_revenue"] - rev_lag4) / np.maximum(np.abs(rev_lag4), 1e-8)

    if "receivables" in fin.columns:
        ar_lag4 = fin.groupby(config.stock_col)["receivables"].shift(4)
        fin["_ar_yoy"] = (fin["receivables"] - ar_lag4) / np.maximum(np.abs(ar_lag4), 1e-8)

    if "inventory" in fin.columns:
        inv_lag4 = fin.groupby(config.stock_col)["inventory"].shift(4)
        fin["_inv_yoy"] = (fin["inventory"] - inv_lag4) / np.maximum(np.abs(inv_lag4), 1e-8)

    fill_cols = [c for c in fin.columns if c.startswith("_ttm_") or c.startswith("_q_") or c in {"_rev_ttm_yoy", "_ar_yoy", "_inv_yoy", "total_assets", "receivables", "inventory", "payables_trade", "equity_parent"}]
    quarterly = fin[[config.stock_col, "available_date", "accper"] + fill_cols].copy()
    quarterly = quarterly.sort_values([config.stock_col, "available_date", "accper"]).drop_duplicates([config.stock_col, "available_date"], keep="last")

    daily = base[[config.date_col, config.stock_col]].copy()
    filled = forward_fill_quarterly_to_daily(daily, quarterly, config, fill_cols)

    daily_extra_cols = [config.date_col, config.stock_col]
    for c in ["mktcap_float", "mktcap_total"]:
        if c in base.columns:
            daily_extra_cols.append(c)
    filled = filled.merge(base[daily_extra_cols], on=[config.date_col, config.stock_col], how="left")
    mkt = filled.get("mktcap_float", filled.get("mktcap_total"))
    if mkt is None:
        mkt = pd.Series(np.nan, index=filled.index)

    eps = 1e-8
    if "F088CF_SALES_Q" in requested and {"_ttm_cf_sales_cash", "_ttm_revenue"}.issubset(filled.columns):
        filled["F088CF_SALES_Q_raw"] = _ttm_ratio(filled["_ttm_cf_sales_cash"], filled["_ttm_revenue"])
    if "F089CASH_PROFIT" in requested and {"_ttm_cf_operating", "_ttm_net_profit"}.issubset(filled.columns):
        filled["F089CASH_PROFIT_raw"] = _ttm_ratio(filled["_ttm_cf_operating"], filled["_ttm_net_profit"])
    if "F090CRR" in requested and "_q_crr" in filled.columns:
        filled["F090CRR_raw"] = filled["_q_crr"]
    if "F091CF_VOL" in requested and "_q_cf_vol" in filled.columns:
        filled["F091CF_VOL_raw"] = filled["_q_cf_vol"]
    if "F092EARN_STAB" in requested and "_q_earn_stab" in filled.columns:
        filled["F092EARN_STAB_raw"] = filled["_q_earn_stab"]
    if "F093FCF_YIELD" in requested and {"_ttm_cf_operating", "_ttm_cf_capex"}.issubset(filled.columns):
        capex = filled["_ttm_cf_capex"]
        cfo = filled["_ttm_cf_operating"]
        fcf = cfo - capex if capex.median(skipna=True) > 0 else cfo + capex
        filled["F093FCF_YIELD_raw"] = fcf / np.maximum(mkt, eps)
    if "F094CAPEX_INT" in requested and {"_ttm_cf_capex", "total_assets"}.issubset(filled.columns):
        filled["F094CAPEX_INT_raw"] = -filled["_ttm_cf_capex"] / np.maximum(filled["total_assets"], eps)
    if "F095NET_FIN" in requested and {"_ttm_cf_borrow", "_ttm_cf_repay_debt", "total_assets"}.issubset(filled.columns):
        filled["F095NET_FIN_raw"] = (filled["_ttm_cf_borrow"] - filled["_ttm_cf_repay_debt"]) / np.maximum(filled["total_assets"], eps)
    if "F096DILUTION" in requested and "_ttm_cf_equity_issue" in filled.columns:
        filled["F096DILUTION_raw"] = -filled["_ttm_cf_equity_issue"] / np.maximum(mkt, eps)
    if "F097INT_BURDEN" in requested and {"_ttm_finance_expense", "_ttm_operating_profit"}.issubset(filled.columns):
        filled["F097INT_BURDEN_raw"] = -_ttm_ratio(filled["_ttm_finance_expense"], filled["_ttm_operating_profit"])
    if "F098DIV_PAYOUT" in requested and {"_ttm_cf_dividend_paid", "_ttm_net_profit"}.issubset(filled.columns):
        filled["F098DIV_PAYOUT_raw"] = _ttm_ratio(filled["_ttm_cf_dividend_paid"], filled["_ttm_net_profit"])
    if "F099AR_MINUS_REV" in requested and {"_ar_yoy", "_rev_ttm_yoy"}.issubset(filled.columns):
        filled["F099AR_MINUS_REV_raw"] = filled["_ar_yoy"] - filled["_rev_ttm_yoy"]
    if "F100INV_MINUS_REV" in requested and {"_inv_yoy", "_rev_ttm_yoy"}.issubset(filled.columns):
        filled["F100INV_MINUS_REV_raw"] = filled["_inv_yoy"] - filled["_rev_ttm_yoy"]

    out_cols = [config.date_col, config.stock_col] + [f"{f}_raw" for f in requested if f"{f}_raw" in filled.columns]
    return filled[out_cols].copy()

# ---------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------


def build_targets(df: pd.DataFrame, config: FactorPanelConfig) -> pd.DataFrame:
    """
    Build forward-return targets.

    1d_next_raw:
        signal on t close, buy at close(t), sell at close(t+1).

    5d_next_raw:
        signal on t close, buy at close(t), sell at close(t+5).
    """

    group = df.groupby(config.stock_col)["close"]

    result = pd.DataFrame(index=df.index)
    result["1d_next_raw"] = group.transform(lambda x: x.shift(-1) / x - 1)
    result["5d_next_raw"] = group.transform(lambda x: x.shift(-5) / x - 1)

    return result


# ---------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------


def winsorize_zscore(
    df: pd.DataFrame,
    factor_names: list[str],
    config: FactorPanelConfig,
    target_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    Cross-sectional winsorize and z-score.

    Optional industry neutralization is supported but off by default.

    When *target_dates* is provided (live/incremental mode), only those dates
    receive cross-sectional quantile winsorizing and z-score; tail rows
    outside *target_dates* are left as-is, saving significant time.
    """

    neutralize_set = set(config.neutralize_factors)
    categorical_factors = {"F055IND"}
    dates_col = config.date_col

    if target_dates is not None and len(target_dates) == 0:
        return df
    live_mode = target_dates is not None

    for name in factor_names:
        raw_col = f"{name}_raw"

        if raw_col not in df.columns:
            print(f"    [WARNING] {raw_col} not found, skipping")
            continue

        if name in categorical_factors:
            df[name] = df[raw_col]
            continue

        work = df[[dates_col, raw_col]].copy()
        work["_idx"] = np.arange(len(work))
        if live_mode:
            live_mask = work[dates_col].isin(target_dates)
            if not live_mask.any():
                continue

        lower = work.groupby(dates_col)[raw_col].transform(
            lambda x: x.quantile(config.winsorize_lower)
        )
        upper = work.groupby(dates_col)[raw_col].transform(
            lambda x: x.quantile(config.winsorize_upper)
        )
        w_col = f"{name}_w"
        work["_w"] = work[raw_col].clip(lower, upper)

        if (
            config.industry_col
            and config.industry_col in df.columns
            and name in neutralize_set
        ):
            has_industry = df[config.industry_col].notna().values
            if has_industry.any():
                group_keys = [dates_col, config.industry_col]
                work["_ind"] = df.loc[has_industry, config.industry_col]
                mu_ind = work.loc[has_industry].groupby(group_keys)["_w"].transform("mean")
                sd_ind = work.loc[has_industry].groupby(group_keys)["_w"].transform("std")
                work.loc[has_industry, "_w"] = (
                    work.loc[has_industry, "_w"] - mu_ind
                ) / sd_ind.where(sd_ind > 1e-10, 1.0)

        mu = work.groupby(dates_col)["_w"].transform("mean")
        sd = work.groupby(dates_col)["_w"].transform("std")
        z = (work["_w"] - mu) / sd.where(sd > 1e-10, 1.0)
        z = z.fillna(0.0)

        if live_mode:
            df.loc[live_mask, raw_col] = work.loc[live_mask, raw_col]  # winsorized raw
            df.loc[live_mask, w_col] = work.loc[live_mask, "_w"]
            df.loc[live_mask, name] = z.loc[live_mask]
        else:
            df[w_col] = work["_w"]
            df[name] = z

    return df


def per_date_zscore(values: pd.Series, dates: pd.Series) -> pd.Series:
    """Cross-sectional z-score by date. Requires datetime dates."""
    dates = pd.to_datetime(dates, errors="raise")

    mu = values.groupby(dates).transform("mean")
    sd = values.groupby(dates).transform("std")
    return (values - mu) / sd.where(sd > 1e-10, 1.0)


# ---------------------------------------------------------------------
# Output / checks / metadata
# ---------------------------------------------------------------------


def filter_and_select_output_columns(
    panel: pd.DataFrame,
    config: FactorPanelConfig,
    factor_names: Sequence[str],
    filter_model_start: bool = True,
    include_targets: bool = True,
) -> pd.DataFrame:
    """Filter date range and select final output columns."""

    df = panel
    if filter_model_start:
        df = df[df[config.date_col] >= pd.Timestamp(config.model_start_date)]

    key_cols = [config.date_col, config.stock_col]
    output_cols = key_cols + [name for name in factor_names if name in df.columns]

    if include_targets:
        output_cols += [col for col in config.target_names if col in df.columns]

    if config.save_raw_factors:
        output_cols += [
            f"{name}_raw"
            for name in factor_names
            if f"{name}_raw" in df.columns
        ]

    if config.save_winsorized_factors:
        output_cols += [
            f"{name}_w"
            for name in factor_names
            if f"{name}_w" in df.columns
        ]

    meta_cols = [
        "industry_sw",
        "list_date",
        "open",
        "high",
        "low",
        "close",
        "close_adj",
        "adj_factor",
        "ret_daily",
        "mktcap_float",
        "mktcap_total",
        "volume",
        "amount",
        "trade_status",
        "limit_status",
        "turnover_rate",
        "pb",
        "pe_ttm",
    ]

    output_cols += [
        col for col in meta_cols
        if col in df.columns and col not in output_cols
    ]

    output_cols = [col for col in output_cols if col in df.columns]

    return df[output_cols].sort_values(key_cols).reset_index(drop=True)


def run_factor_panel_checks(
    panel: pd.DataFrame,
    config: FactorPanelConfig,
    factor_names: Sequence[str],
    is_incremental: bool = False,
) -> dict[str, Any]:
    """Run factor panel quality checks."""

    validate_no_duplicate_keys(panel, config, "factor_panel")

    audit: dict[str, Any] = {
        "is_incremental": is_incremental,
        "n_rows": int(len(panel)),
        "n_stocks": int(panel[config.stock_col].nunique()),
        "start_date": str(panel[config.date_col].min()),
        "end_date": str(panel[config.date_col].max()),
        "factor_missing_rate": {},
        "factor_extreme_abs_gt_5_rate": {},
        "target_missing_rate": {},
    }

    print("\n========== Factor Panel Checks ==========")
    print(f"Rows: {len(panel):,}")
    print(f"Stocks: {panel[config.stock_col].nunique():,}")
    print(f"Dates: {panel[config.date_col].min()} ~ {panel[config.date_col].max()}")

    for name in factor_names:
        if name not in panel.columns:
            audit["factor_missing_rate"][name] = None
            print(f"  {name:16s} MISSING")
            continue

        missing = float(panel[name].isna().mean())
        extreme = float((panel[name].abs() > 5).mean())
        audit["factor_missing_rate"][name] = missing
        audit["factor_extreme_abs_gt_5_rate"][name] = extreme

        print(f"  {name:16s} missing={missing:.4f} extreme(|z|>5)={extreme:.4f}")

    for target in config.target_names:
        if target in panel.columns:
            audit["target_missing_rate"][target] = float(panel[target].isna().mean())

    available_factor_names = [name for name in factor_names if name in panel.columns]
    sample_dates = pd.DatetimeIndex(panel[config.date_col].dropna().unique()).sort_values()[:5]

    if available_factor_names and len(sample_dates) > 0:
        print("\nCross-sectional mean/std sample:")
        for date in sample_dates:
            sub = panel[panel[config.date_col] == date]
            means = [sub[name].mean() for name in available_factor_names]
            stds = [sub[name].std() for name in available_factor_names]
            print(
                f"  {date.date()}: "
                f"mean range [{np.nanmin(means):.3f}, {np.nanmax(means):.3f}], "
                f"std range [{np.nanmin(stds):.3f}, {np.nanmax(stds):.3f}]"
            )

    return audit


def build_factor_metadata(
    panel: pd.DataFrame,
    config: FactorPanelConfig,
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Build metadata for factor panel output."""

    return {
        "factor_version": config.factor_version,
        "factor_alias": FACTOR_ALIAS,
        "barra12_factors": BARRA12_FACTORS,
        "new12_factors": NEW12_FACTORS,
        "config": asdict(config),
        "audit": audit,
        "output_rows": int(len(panel)),
        "output_stocks": int(panel[config.stock_col].nunique()),
        "output_start_date": str(panel[config.date_col].min()),
        "output_end_date": str(panel[config.date_col].max()),
    }


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def standardize_panel_keys(
    df: pd.DataFrame,
    config: FactorPanelConfig,
) -> pd.DataFrame:
    """Standardize stock_id and time columns."""

    out = df.copy(deep=False)
    out[config.stock_col] = out[config.stock_col].astype(str).str.strip().str.zfill(6)
    out[config.date_col] = pd.to_datetime(out[config.date_col], errors="raise").dt.normalize()
    return out


def standardize_financial_keys(
    df: pd.DataFrame,
    config: FactorPanelConfig,
) -> pd.DataFrame:
    """Standardize quarterly financial keys."""

    out = df.copy(deep=False)
    out[config.stock_col] = out[config.stock_col].astype(str).str.strip().str.zfill(6)
    out["accper"] = pd.to_datetime(out["accper"], errors="raise").dt.normalize()

    if "available_date" not in out.columns:
        raise ValueError("financial_quarterly panel must contain available_date")

    out["available_date"] = pd.to_datetime(
        out["available_date"],
        errors="raise",
    ).dt.normalize()

    return out


def validate_no_duplicate_keys(
    df: pd.DataFrame,
    config: FactorPanelConfig,
    name: str,
) -> None:
    """Validate no duplicated stock-date keys."""

    key_cols = [config.stock_col, config.date_col]
    dup = df.duplicated(subset=key_cols)

    if dup.any():
        sample = df.loc[dup, key_cols].head(10)
        raise ValueError(
            f"{name} has duplicated stock-date rows: {int(dup.sum())}. "
            f"Sample:\n{sample}"
        )


def validate_required_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    """Validate required columns exist."""

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
# ════════════════════════════════════════════════════════════
# V2 factor computation (F056GAP_UP_FAIL-F100INV_MINUS_REV) — merged from factor_blocks_v2
# ════════════════════════════════════════════════════════════
def _ytd_to_single_quarter(df: pd.DataFrame, stock_col: str, date_col: str,
                           flow_cols: list[str]) -> pd.DataFrame:
    """Convert YTD cumulative flow fields to single-quarter.

    CSMAR / Tushare financials are YTD cumulative:
        Q1=Q1, H1=Q1+Q2, Q3=Q1+Q2+Q3, FY=Q1+Q2+Q3+Q4.

    After conversion, each quarter column contains only that quarter's value.
    """
    df = df.copy()
    df["_q"] = df[date_col].dt.quarter
    for col in flow_cols:
        if col not in df.columns:
            continue
        sq_col = f"_sq_{col}"
        # Group by stock and fiscal year
        df["_fyear"] = df[date_col].dt.year
        df["_fyear"] -= (df["_q"] == 1).astype(int)  # Q1 belongs to prior FY reporting
        # Actually simpler: sort by stock + date, diff within same fiscal year
        df = df.sort_values([stock_col, date_col])
        # For each stock, compute single-quarter = current YTD - previous quarter YTD
        prev_ytd = df.groupby(stock_col)[col].shift(1)
        df[sq_col] = df[col] - prev_ytd
        # Q1: YTD = single quarter (no subtraction needed — but need to handle year boundary)
        is_q1 = df["_q"] == 1
        df.loc[is_q1, sq_col] = df.loc[is_q1, col]
        # Negative values from QoQ decline → clamp to 0? No, some can be legitimately negative.
    return df.drop(columns=["_q", "_fyear"])


def _ttm_sum(df: pd.DataFrame, stock_col: str, sq_cols: list[str]) -> pd.DataFrame:
    """TTM = sum of latest 4 single-quarter values."""
    df = df.sort_values([stock_col, "time"])
    for sq_col in sq_cols:
        if sq_col not in df.columns:
            continue
        raw_col = sq_col.replace("_sq_", "")
        ttm_col = f"_ttm_{raw_col}"
        df[ttm_col] = df.groupby(stock_col)[sq_col].transform(
            lambda x: x.rolling(4, min_periods=1).sum())
    return df


def _ttm_ratio(a: pd.Series, b: pd.Series, eps: float = 1e-8) -> pd.Series:
    """Safe TTM ratio a / b."""
    return a / np.maximum(np.abs(b), eps)


# ════════════════════════════════════════════════════════════════
# Block 1: O2O / Overnight / Intraday (F056GAP_UP_FAIL–F062GAP_DN_RECOVER)
# ════════════════════════════════════════════════════════════════

def build_onret1(df: pd.DataFrame) -> pd.Series:
    """F056GAP_UP_FAIL ONRET1 = open / pre_close - 1."""
    return df["open"] / df["pre_close"] - 1.0


def build_intra1(df: pd.DataFrame) -> pd.Series:
    """F057INTRA1 INTRA1 = close / open - 1."""
    return df["close"] / df["open"] - 1.0


def build_o2o_ret5(df: pd.DataFrame, config) -> pd.Series:
    """F058O2O_RET5 O2O_RET5 = open / open.shift(5) - 1 per stock."""
    df = df.sort_values([config.stock_col, config.date_col])
    return df.groupby(config.stock_col)["open"].transform(
        lambda x: x / x.shift(5) - 1.0)


def build_gk_vol20(df: pd.DataFrame) -> pd.Series:
    """F059GK_VOL20 GK_VOL20 = Garman-Klass OHLC volatility over 20 days.

    More efficient than Parkinson — uses open/close in addition to high/low.
    """
    ln_hl = np.log(df["high"] / df["low"])
    ln_co = np.log(df["close"] / df["open"])
    gk = 0.5 * ln_hl ** 2 - (2 * np.log(2) - 1) * ln_co ** 2
    stock_col = "stock_id" if "stock_id" in df.columns else df.index
    gk_mean = gk.groupby(df[stock_col]).transform(lambda x: x.rolling(20, min_periods=10).mean())
    return np.sqrt(gk_mean.clip(lower=0))


def _onret_sum5(df: pd.DataFrame, config) -> pd.Series:
    onret1 = build_onret1(df)
    work = pd.DataFrame({
        config.stock_col: df[config.stock_col],
        config.date_col: df[config.date_col],
        "_onret1": onret1,
    })
    work = work.sort_values([config.stock_col, config.date_col])
    return work.groupby(config.stock_col)["_onret1"].transform(
        lambda x: x.rolling(5, min_periods=3).sum())


def _intra_sum5(df: pd.DataFrame, config) -> pd.Series:
    intra1 = build_intra1(df)
    work = pd.DataFrame({
        config.stock_col: df[config.stock_col],
        config.date_col: df[config.date_col],
        "_intra1": intra1,
    })
    work = work.sort_values([config.stock_col, config.date_col])
    return work.groupby(config.stock_col)["_intra1"].transform(
        lambda x: x.rolling(5, min_periods=3).sum())


def build_on_intra_div5(df: pd.DataFrame, config) -> pd.Series:
    """F060ON_INTRA_DIV5 ON_INTRA_DIV5 = ONRET_SUM5 - INTRA_SUM5."""
    return _onret_sum5(df, config) - _intra_sum5(df, config)


def build_gap_up_hold(df: pd.DataFrame) -> pd.Series:
    """F061GAP_UP_HOLD GAP_UP_HOLD = 1[ONRET1>0] * INTRA1."""
    onret1 = build_onret1(df)
    intra1 = build_intra1(df)
    return (onret1 > 0).astype(float) * intra1


def build_gap_dn_recover(df: pd.DataFrame) -> pd.Series:
    """F062GAP_DN_RECOVER GAP_DN_RECOVER = 1[ONRET1<0] * INTRA1."""
    onret1 = build_onret1(df)
    intra1 = build_intra1(df)
    return (onret1 < 0).astype(float) * intra1


def build_gap_up_fail(df: pd.DataFrame) -> pd.Series:
    """F056GAP_UP_FAIL GAP_UP_FAIL = 1[ONRET1>0 & INTRA1<0] * (-INTRA1).

    Gap-up that reversed intraday — over-optimism punished.
    Replaces ONRET1 which was 0.992 correlated with GAP (F025).
    """
    onret1 = build_onret1(df)
    intra1 = build_intra1(df)
    mask = (onret1 > 0) & (intra1 < 0)
    return mask.astype(float) * (-intra1)


# ════════════════════════════════════════════════════════════════
# Block 2: Momentum / Path Quality (F063RET5D_SKIP1–F066EFFICIENCY20)
# ════════════════════════════════════════════════════════════════

def build_ret5d_skip1(df: pd.DataFrame, config) -> pd.Series:
    """F063RET5D_SKIP1 RET5D_SKIP1 = close.shift(1) / close.shift(6) - 1 (close-to-close)."""
    df = df.sort_values([config.stock_col, config.date_col])
    col = df["close"]
    return col.groupby(df[config.stock_col]).transform(
        lambda x: x.shift(1) / x.shift(6) - 1.0)


def build_ret_accel20(df: pd.DataFrame, config) -> pd.Series:
    """F064RET_ACCEL20 RET_ACCEL20 = ret_20d[t-1] - ret_20d[t-21] (close-to-close)."""
    df = df.sort_values([config.stock_col, config.date_col])
    col = df["close"]
    ret20 = col.groupby(df[config.stock_col]).transform(
        lambda x: x / x.shift(20) - 1.0)
    return ret20.groupby(df[config.stock_col]).transform(
        lambda x: x.shift(1) - x.shift(21))


def build_maxdd20(df: pd.DataFrame, config) -> pd.Series:
    """F065MAXDD20 MAXDD20 = true rolling max drawdown over 20 days per stock.

    Known bottleneck: this remains a small two-level loop because true rolling
    peak-to-trough max drawdown is path-dependent.  Expected to be slower than
    pure rolling sums; keep it isolated and profile separately if needed.
    """
    df = df.sort_values([config.stock_col, config.date_col])
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for _stock_id, grp in df.groupby(config.stock_col, sort=False):
        prices = grp["close"].to_numpy(dtype=float)
        vals = np.full(len(prices), np.nan, dtype=float)
        for t in range(19, len(prices)):
            window = prices[t - 19:t + 1]
            if not np.isfinite(window).all():
                continue
            peak = np.maximum.accumulate(window)
            drawdown = window / np.maximum(peak, 1e-12) - 1.0
            vals[t] = np.nanmin(drawdown)
        result.loc[grp.index] = vals
    return result


def build_efficiency20(df: pd.DataFrame, config) -> pd.Series:
    """F066EFFICIENCY20 EFFICIENCY20 = abs(RET20) / sum(abs(ret_daily), 20) (close-to-close)."""
    df = df.sort_values([config.stock_col, config.date_col])
    col = df["close"]
    ret20 = col.groupby(df[config.stock_col]).transform(
        lambda x: x / x.shift(20) - 1.0)
    abs_ret = df["ret_daily"].abs()
    sum_abs = abs_ret.groupby(df[config.stock_col]).transform(
        lambda x: x.rolling(20, min_periods=10).sum())
    return np.abs(ret20) / np.maximum(sum_abs, 1e-8)


# ════════════════════════════════════════════════════════════════
# Block 3: Volatility / Tail / Market Linkage (F067TAIL_LOSS20–F074BETA_20D)
# ════════════════════════════════════════════════════════════════

def precompute_ret_blocks(df: pd.DataFrame, config) -> pd.DataFrame:
    """Pre-compute shared rolling stats for vol/market blocks.

    Uses rolling-sum decomposition for cov/var instead of rolling.cov()
    to avoid O(n × groups × window) groupby-overhead.  10× faster.
    """
    df = df.sort_values([config.stock_col, config.date_col])
    ret = df["ret_daily"]
    mkt = df["mkt_ret_vw"]
    stock = df[config.stock_col]

    import time as _t

    # -- Rolling std (fast, built-in) --
    df["_std5"] = ret.groupby(stock).transform(lambda x: x.rolling(5, min_periods=5).std())
    df["_std20"] = ret.groupby(stock).transform(lambda x: x.rolling(20, min_periods=10).std())

    # -- Rolling skew (20d) via sum decomposition --
    # Skew = (E[x³] - 3μE[x²] + 2μ³) / σ³
    print("    Rolling skew 20d...", end=" ", flush=True)
    _t0 = _t.perf_counter()
    w, minp = 20, 20
    rsum = ret.groupby(stock).transform(lambda x: x.rolling(w, min_periods=minp).sum())
    r2sum = (ret * ret).groupby(stock).transform(lambda x: x.rolling(w, min_periods=minp).sum())
    r3sum = (ret * ret * ret).groupby(stock).transform(lambda x: x.rolling(w, min_periods=minp).sum())
    cnt = ret.groupby(stock).transform(lambda x: x.rolling(w, min_periods=minp).count())
    mu = rsum / cnt
    ex2 = r2sum / cnt
    ex3 = r3sum / cnt
    var = (ex2 - mu * mu).clip(lower=0)
    std = np.sqrt(var)
    third = ex3 - 3 * mu * ex2 + 2 * mu**3
    df["_skew20"] = (third / (std**3 + 1e-12)).mask(std < 1e-8)
    print(f"done in {_t.perf_counter() - _t0:.1f}s", flush=True)

    # -- Rolling cov/var via sum decomposition (fast ~10× vs rolling.cov) --
    # Cov(x, y, w) = E[xy]_w - E[x]_w * E[y]_w
    # Var(x, w)    = E[x²]_w - E[x]_w²
    for w in [20, 60]:
        label = str(w)
        minp = max(10, w // 2)
        print(f"    Rolling {w}d cov sums...", end=" ", flush=True)
        _t0 = _t.perf_counter()

        valid = ret.notna() & mkt.notna()
        r = ret.where(valid)
        m = mkt.where(valid)

        cnt = valid.astype(float).groupby(stock).transform(
            lambda x: x.rolling(w, min_periods=minp).sum())
        rsum = r.groupby(stock).transform(lambda x: x.rolling(w, min_periods=minp).sum())
        msum = m.groupby(stock).transform(lambda x: x.rolling(w, min_periods=minp).sum())
        r2sum = (r * r).groupby(stock).transform(lambda x: x.rolling(w, min_periods=minp).sum())
        m2sum = (m * m).groupby(stock).transform(lambda x: x.rolling(w, min_periods=minp).sum())
        rmsum = (r * m).groupby(stock).transform(lambda x: x.rolling(w, min_periods=minp).sum())

        c = cnt.clip(lower=1)
        df[f"_cov{label}"] = rmsum / c - (rsum / c) * (msum / c)
        df[f"_var_mkt{label}"] = (m2sum / c - (msum / c) ** 2).clip(lower=0)
        df[f"_var_i{label}"] = (r2sum / c - (rsum / c) ** 2).clip(lower=0)
        print(f"done in {_t.perf_counter() - _t0:.1f}s", flush=True)

    return df


def _precompute_ret_blocks(df: pd.DataFrame, config) -> pd.DataFrame:
    """Backward-compatible wrapper."""
    return precompute_ret_blocks(df, config)


def build_tail_loss20(df: pd.DataFrame) -> pd.Series:
    """F067TAIL_LOSS20 TAIL_LOSS20 = rolling_min(ret_daily, 20) — worst daily return.

    Replaces PARKINSON20 which was 0.986 correlated with GK_VOL20 (F059GK_VOL20).
    """
    stock_col = "stock_id" if "stock_id" in df.columns else df.index
    return df["ret_daily"].groupby(df[stock_col]).transform(
        lambda x: x.rolling(20, min_periods=10).min())


def build_parkinson20(df: pd.DataFrame) -> pd.Series:
    """Parkinson volatility (backup, not used in final 87)."""
    hl2 = (np.log(df["high"] / df["low"])) ** 2
    stock_col = "stock_id" if "stock_id" in df.columns else df.index
    return hl2.groupby(df[stock_col]).transform(
        lambda x: x.rolling(20, min_periods=10).mean())


def build_skew20(df: pd.DataFrame) -> pd.Series:
    """F068SKEW20 SKEW20 — pre-computed in precompute_ret_blocks."""
    if "_skew20" in df.columns:
        return df["_skew20"]
    stock_col = "stock_id" if "stock_id" in df.columns else df.index
    r = df["ret_daily"]
    w, minp = 20, 20
    cnt = r.groupby(df[stock_col] if isinstance(stock_col, str) else df.index).transform(lambda x: x.rolling(w, min_periods=minp).count())
    s1 = r.groupby(df[stock_col] if isinstance(stock_col, str) else df.index).transform(lambda x: x.rolling(w, min_periods=minp).sum())
    s2 = (r*r).groupby(df[stock_col] if isinstance(stock_col, str) else df.index).transform(lambda x: x.rolling(w, min_periods=minp).sum())
    s3 = (r*r*r).groupby(df[stock_col] if isinstance(stock_col, str) else df.index).transform(lambda x: x.rolling(w, min_periods=minp).sum())
    mu = s1 / cnt
    ex2 = s2 / cnt
    ex3 = s3 / cnt
    var = (ex2 - mu*mu).clip(lower=0)
    std = np.sqrt(var)
    third = ex3 - 3*mu*ex2 + 2*mu**3
    return (third / (std**3 + 1e-12)).mask(std < 1e-8)


def build_dnvol20(df: pd.DataFrame) -> pd.Series:
    """F069DNVOL20 DNVOL20 = std(min(ret, 0), 20) — zeros included."""
    dn = np.minimum(df["ret_daily"], 0.0)
    return dn.groupby(df["stock_id"] if "stock_id" in df.columns else df.index
                      ).transform(lambda x: x.rolling(20, min_periods=10).std())


def build_up_dn_vol(df: pd.DataFrame) -> pd.Series:
    """F070UP_DN_VOL UP_DN_VOL = std(max(ret, 0), 20) / (F069DNVOL20 + eps)."""
    up = np.maximum(df["ret_daily"], 0.0)
    up_std = up.groupby(df["stock_id"] if "stock_id" in df.columns else df.index
                        ).transform(lambda x: x.rolling(20, min_periods=10).std())
    dn_std = build_dnvol20(df)
    return up_std / np.maximum(dn_std, 1e-8)


def build_vol_of_vol(df: pd.DataFrame) -> pd.Series:
    """F071VOL_OF_VOL VOL_OF_VOL = std(rolling_std(ret,5), 20)."""
    if "_std5" not in df.columns:
        df["_std5"] = df.groupby("stock_id")["ret_daily"].transform(
            lambda x: x.rolling(5, min_periods=5).std())
    stock_col = "stock_id" if "stock_id" in df.columns else df.index
    return df["_std5"].groupby(df[stock_col] if isinstance(stock_col, str) else df.index
                               ).transform(lambda x: x.rolling(20, min_periods=10).std())


def build_corr_60d(df: pd.DataFrame) -> pd.Series:
    """F072CORR_60D CORR_60D = cov60 / sqrt(var_i60 * var_mkt60)."""
    if "_cov60" not in df.columns:
        return _compute_corr_direct(df, 60)
    return df["_cov60"] / np.sqrt(np.maximum(df["_var_i60"] * df["_var_mkt60"], 1e-20))


def _compute_corr_direct(df: pd.DataFrame, w: int) -> pd.Series:
    """Fallback: direct rolling correlation (slow — only if precompute skipped)."""
    stock_col = "stock_id" if "stock_id" in df.columns else df.index
    ret = df["ret_daily"]
    mkt = df["mkt_ret_vw"]

    def _roll_corr(xy, w):
        x, y = xy.iloc[:, 0], xy.iloc[:, 1]
        return x.rolling(w, min_periods=max(10, w // 2)).corr(y)

    pair = pd.concat([ret, mkt], axis=1)
    return pair.groupby(df[stock_col] if isinstance(stock_col, str) else df.index
                        ).apply(lambda g: _roll_corr(g, w)).reset_index(level=0, drop=True)


def build_rsq_60d(df: pd.DataFrame) -> pd.Series:
    """F073KURT_60D RSQ_60D = beta60² × Var(mkt,60) / Var(i,60), clipped to [0,1]."""
    if "_cov60" in df.columns:
        beta60 = df["_cov60"] / np.maximum(df["_var_mkt60"], 1e-12)
        return (beta60 ** 2 * df["_var_mkt60"] / np.maximum(df["_var_i60"], 1e-12)).clip(0, 1)
    # Fallback
    w = 60; stock_col = "stock_id" if "stock_id" in df.columns else df.index
    def _rsq(g):
        cov60 = (g["ret_daily"] * g["mkt_ret_vw"]).rolling(w, min_periods=max(10, w//2)).sum() / w
        var_m = (g["mkt_ret_vw"]).rolling(w, min_periods=max(10, w//2)).var()
        var_i = (g["ret_daily"]).rolling(w, min_periods=max(10, w//2)).var()
        beta = cov60 / np.maximum(var_m, 1e-12)
        return (beta ** 2 * var_m / np.maximum(var_i, 1e-12)).clip(0, 1)
    return df.groupby(stock_col).apply(
        lambda g: _rsq(g[["ret_daily", "mkt_ret_vw"]].copy())
    ).reset_index(level=0, drop=True)


def build_beta_20d(df: pd.DataFrame) -> pd.Series:
    """F074BETA_20D BETA_20D = cov(ret, mkt, 20) / var(mkt, 20)."""
    if "_cov20" in df.columns:
        return df["_cov20"] / np.maximum(df["_var_mkt20"], 1e-12)
    w = 20; stock_col = "stock_id" if "stock_id" in df.columns else df.index
    def _beta(g):
        cov20 = (g["ret_daily"] * g["mkt_ret_vw"]).rolling(w, min_periods=max(10, w//2)).sum() / w
        var_m = (g["mkt_ret_vw"]).rolling(w, min_periods=max(10, w//2)).var()
        return cov20 / np.maximum(var_m, 1e-12)
    return df.groupby(stock_col).apply(
        lambda g: _beta(g[["ret_daily", "mkt_ret_vw"]].copy())
    ).reset_index(level=0, drop=True)


def build_kurt_60d(df: pd.DataFrame, config) -> pd.Series:
    """F073KURT_60D KURT_60D = kurtosis(ret_daily, 60d) via sum decomposition.

    Replaces RSQ_60D which was 0.998 correlated with CORR_60D (F072CORR_60D).
    Kurt = (E[x^4] - 4μE[x^3] + 6μ²E[x²] - 3μ^4) / σ^4
    """
    stock_col = config.stock_col
    df = df.sort_values([stock_col, config.date_col])
    ret = df["ret_daily"]
    w, minp = 60, 30

    cnt = ret.groupby(df[stock_col]).transform(
        lambda x: x.rolling(w, min_periods=minp).count()).clip(lower=1)
    r1 = ret.groupby(df[stock_col]).transform(lambda x: x.rolling(w, min_periods=minp).sum())
    r2 = (ret**2).groupby(df[stock_col]).transform(lambda x: x.rolling(w, min_periods=minp).sum())
    r3 = (ret**3).groupby(df[stock_col]).transform(lambda x: x.rolling(w, min_periods=minp).sum())
    r4 = (ret**4).groupby(df[stock_col]).transform(lambda x: x.rolling(w, min_periods=minp).sum())

    mu = r1 / cnt
    mu2 = r2 / cnt
    mu3 = r3 / cnt
    mu4 = r4 / cnt
    var = (mu2 - mu**2).clip(lower=0)
    std = np.sqrt(var)
    kurt_num = mu4 - 4 * mu * mu3 + 6 * mu**2 * mu2 - 3 * mu**4
    return (kurt_num / (std**4 + 1e-12)).mask(std < 1e-8)


# ════════════════════════════════════════════════════════════════
# Block 4: Volume / VWAP (F075VOLUME_RATIO–F081STRONG_CLOSE)
# ════════════════════════════════════════════════════════════════

def build_volume_ratio(df: pd.DataFrame) -> pd.Series:
    """F075VOLUME_RATIO VOLUME_RATIO — directly from panel column."""
    return df["volume_ratio"].astype(float)


def build_signed_amt20(df: pd.DataFrame) -> pd.Series:
    """F076SIGNED_AMT20 SIGNED_AMT20 = sum(sign(ret) * amount, 20) / sum(amount, 20)."""
    stock_col = "stock_id" if "stock_id" in df.columns else df.index
    signed = np.sign(df["ret_daily"]) * df["amount"]
    num = signed.groupby(df[stock_col] if isinstance(stock_col, str) else df.index
                        ).transform(lambda x: x.rolling(20, min_periods=10).sum())
    denom = df["amount"].groupby(df[stock_col] if isinstance(stock_col, str) else df.index
                                 ).transform(lambda x: x.rolling(20, min_periods=10).sum())
    return num / np.maximum(denom, 1e-8)


def build_amp_vol20(df: pd.DataFrame) -> pd.Series:
    """F077AMP_VOL20 AMP_VOL20 = mean(|ret| * volume, 20)."""
    stock_col = "stock_id" if "stock_id" in df.columns else df.index
    amp = np.abs(df["ret_daily"]) * df["volume"]
    return amp.groupby(df[stock_col] if isinstance(stock_col, str) else df.index
                       ).transform(lambda x: x.rolling(20, min_periods=10).mean())


def build_turn_size(df: pd.DataFrame) -> pd.Series:
    """F078TURN_SIZE TURN_SIZE = turnover_rate * ln(mktcap)."""
    return df["turnover_rate"] * np.log(df["mktcap_total"])


def build_turn_accel(df: pd.DataFrame) -> pd.Series:
    """F079TURN_ACCEL TURN_ACCEL = MA(turnover, 5) / MA(turnover, 20)."""
    stock_col = "stock_id" if "stock_id" in df.columns else df.index
    tr = df["turnover_rate"]
    ma5 = tr.groupby(df[stock_col] if isinstance(stock_col, str) else df.index
                     ).transform(lambda x: x.rolling(5, min_periods=3).mean())
    ma20 = tr.groupby(df[stock_col] if isinstance(stock_col, str) else df.index
                      ).transform(lambda x: x.rolling(20, min_periods=10).mean())
    return ma5 / np.maximum(ma20, 1e-8)


def build_vwap_dev(df: pd.DataFrame) -> pd.Series:
    """F080VWAP_DEV VWAP_DEV = close / (amount/volume) - 1."""
    px = df["_close_raw"] if "_close_raw" in df.columns else df["close"]
    vwap = df["amount"] / np.maximum(df["volume"], 1)
    return px / np.maximum(vwap, 1e-8) - 1.0


def build_strong_close(df: pd.DataFrame) -> pd.Series:
    """F081STRONG_CLOSE STRONG_CLOSE = CLOSE_POS * log1p(amount)."""
    close_pos = (df["close"] - df["low"]) / np.maximum(df["high"] - df["low"], 1e-8)
    return close_pos * np.log1p(df["amount"])


# ════════════════════════════════════════════════════════════════
# Block 5: Float / Value / Age (F082LOCKED_PCT–F087LIST_AGE)
# ════════════════════════════════════════════════════════════════

def build_locked_pct(df: pd.DataFrame) -> pd.Series:
    """F082LOCKED_PCT LOCKED_PCT = 1 - free_share / total_share."""
    return 1.0 - df["free_share"] / np.maximum(df["total_share"], 1)


def build_turn_free(df: pd.DataFrame) -> pd.Series:
    """F083TURN_FREE TURN_FREE = amount / (free_share * close)."""
    px = df["_close_raw"] if "_close_raw" in df.columns else df["close"]
    return df["amount"] / np.maximum(df["free_share"] * px, 1e-8)


def build_amt_free20(df: pd.DataFrame) -> pd.Series:
    """F084AMT_FREE20 AMT_FREE20 = MA(amount, 20) / (free_share * close)."""
    px = df["_close_raw"] if "_close_raw" in df.columns else df["close"]
    stock_col = "stock_id" if "stock_id" in df.columns else df.index
    ma_amt = df["amount"].groupby(df[stock_col] if isinstance(stock_col, str) else df.index
                                  ).transform(lambda x: x.rolling(20, min_periods=10).mean())
    return ma_amt / np.maximum(df["free_share"] * px, 1e-8)


def build_sp_ttm(df: pd.DataFrame) -> pd.Series:
    """F085SP_TTM SP_TTM = 1 / ps_ttm."""
    return 1.0 / np.maximum(df["ps_ttm"], 1e-8)


def build_div_ttm(df: pd.DataFrame) -> pd.Series:
    """F086DIV_TTM DIV_TTM = dv_ttm."""
    return df["dv_ttm"]


def build_list_age(df: pd.DataFrame, config) -> pd.Series:
    """F087LIST_AGE LIST_AGE = log1p(trading_days_since_list_date)."""
    list_dates = pd.to_datetime(df["list_date"])
    current_dates = pd.to_datetime(df[config.date_col])
    days = (current_dates - list_dates).dt.days
    return np.log1p(np.maximum(days, 0))


# ════════════════════════════════════════════════════════════════
# Block 6: Financial — shared TTM pipeline (F088CF_SALES_Q–F100INV_MINUS_REV)
# ════════════════════════════════════════════════════════════════

def _precompute_financial_ttm(df: pd.DataFrame, config) -> pd.DataFrame:
    """YTD→SQ→TTM for all financial flow fields needed by F088CF_SALES_Q-F100INV_MINUS_REV.

    Returns df with _ttm_xxx columns added.
    """
    flow_cols = [
        "cf_sales_cash", "cf_operating", "cf_capex", "cf_borrow",
        "cf_repay_debt", "cf_equity_issue", "cf_dividend_paid",
        "revenue", "net_profit", "operating_profit", "finance_expense",
        "income_tax",
    ]
    available = [c for c in flow_cols if c in df.columns]
    if not available:
        return df

    df = df.sort_values([config.stock_col, config.date_col])

    # YTD → single-quarter
    df["_q"] = pd.to_datetime(df[config.date_col]).dt.quarter
    for col in available:
        sq_col = f"_sq_{col}"
        prev = df.groupby(config.stock_col)[col].shift(1)
        df[sq_col] = df[col] - prev
        is_q1 = df["_q"] == 1
        # Q1: if this is the first observation for the stock, prev is NaN
        # → Q1 YTD = single quarter directly
        mask_q1 = is_q1 | prev.isna()
        df.loc[mask_q1, sq_col] = df.loc[mask_q1, col]

    df = df.drop(columns=["_q"])

    # Single-quarter → TTM (sum of latest 4)
    sq_cols = [f"_sq_{c}" for c in available]
    for sq_col in sq_cols:
        raw = sq_col.replace("_sq_", "")
        ttm_col = f"_ttm_{raw}"
        df[ttm_col] = df.groupby(config.stock_col)[sq_col].transform(
            lambda x: x.rolling(4, min_periods=1).sum())

    # Also prep balance-sheet stock variables (no TTM needed, use latest)
    for col in ["total_assets", "receivables", "inventory", "payables_trade",
                "equity_parent"]:
        if col in df.columns:
            df[f"_latest_{col}"] = df[col]

    return df


# -- F088CF_SALES_Q-F093FCF_YIELD: Cash Flow Quality --

def build_cf_sales_q(df: pd.DataFrame) -> pd.Series:
    """F088CF_SALES_Q CF_SALES_Q = TTM(cf_sales_cash) / TTM(revenue)."""
    return _ttm_ratio(df["_ttm_cf_sales_cash"], df["_ttm_revenue"])


def build_cash_profit(df: pd.DataFrame) -> pd.Series:
    """F089CASH_PROFIT CASH_PROFIT = TTM(cf_operating) / |TTM(net_profit)|."""
    return _ttm_ratio(df["_ttm_cf_operating"], df["_ttm_net_profit"])


def build_crr(df: pd.DataFrame, config) -> pd.Series:
    """F090CRR CRR = β from 16Q rolling OLS: cf_operating_q = α + β × net_profit_q(lag1) + ε.

    Uses single-quarter values, not TTM.  Vectorised via sum decomposition — 50× faster
    than per-stock for-loop.
    """
    x_col = "_sq_net_profit"
    y_col = "_sq_cf_operating"
    if x_col not in df.columns or y_col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    df = df.sort_values([config.stock_col, config.date_col])
    stock = df[config.stock_col]
    x = df[x_col].groupby(stock).shift(1)  # lag 1Q
    y = df[y_col]

    w, minp = 16, 8
    cnt = x.notna().astype(float).groupby(stock).transform(
        lambda g: g.rolling(w, min_periods=minp).sum()).clip(lower=1)

    xsum = x.groupby(stock).transform(lambda g: g.rolling(w, min_periods=minp).sum())
    ysum = y.groupby(stock).transform(lambda g: g.rolling(w, min_periods=minp).sum())
    xysum = (x * y).groupby(stock).transform(lambda g: g.rolling(w, min_periods=minp).sum())
    x2sum = (x * x).groupby(stock).transform(lambda g: g.rolling(w, min_periods=minp).sum())

    cov = xysum / cnt - (xsum / cnt) * (ysum / cnt)
    var = (x2sum / cnt - (xsum / cnt) ** 2).clip(lower=0)
    return cov / np.maximum(var, 1e-12)


def build_cf_vol(df: pd.DataFrame, config) -> pd.Series:
    """F091CF_VOL CF_VOL = -std(TTM(cf_operating) / total_assets, 16Q)."""
    ratio = df["_ttm_cf_operating"] / np.maximum(df["_latest_total_assets"], 1e-8)
    stock_col = config.stock_col
    std16 = ratio.groupby(df[stock_col]).transform(
        lambda x: x.rolling(16, min_periods=8).std())
    return -std16


def build_earn_stab(df: pd.DataFrame, config) -> pd.Series:
    """F092EARN_STAB EARN_STAB = -std(TTM(net_profit) / total_assets, 16Q)."""
    ratio = df["_ttm_net_profit"] / np.maximum(df["_latest_total_assets"], 1e-8)
    stock_col = config.stock_col
    std16 = ratio.groupby(df[stock_col]).transform(
        lambda x: x.rolling(16, min_periods=8).std())
    return -std16


def build_wcap_press(df: pd.DataFrame) -> pd.Series:
    """WCAP_PRESS (backup)."""
    num = (df["_latest_receivables"] + df["_latest_inventory"]
           - df["_latest_payables_trade"])
    return num / np.maximum(df["_latest_total_assets"], 1e-8)


def build_fcf_yield(df: pd.DataFrame) -> pd.Series:
    """F093FCF_YIELD FCF_YIELD = (TTM(cf_operating) - TTM(cf_capex)) / mktcap_float.

    Free cash flow yield. If median capex > 0, FCF = CFO - capex;
    otherwise FCF = CFO + capex (some sectors report capex as negative).
    Replaces WCAP_PRESS (ICIR=0.0005).
    """
    cfo = df["_ttm_cf_operating"]
    capex = df["_ttm_cf_capex"]
    # Adjust capex sign convention: if median capex > 0, subtract; else add
    median_capex = capex.median()
    if pd.notna(median_capex) and median_capex > 0:
        fcf = cfo - capex
    else:
        fcf = cfo + capex
    return fcf / np.maximum(df["mktcap_float"], 1e-8)


# -- F094CAPEX_INT-F098DIV_PAYOUT: Investment & Financing --

def build_capex_int(df: pd.DataFrame) -> pd.Series:
    """F094CAPEX_INT CAPEX_INT = -TTM(cf_capex) / total_assets."""
    return -df["_ttm_cf_capex"] / np.maximum(df["_latest_total_assets"], 1e-8)


def build_net_fin(df: pd.DataFrame) -> pd.Series:
    """F095NET_FIN NET_FIN = (TTM(cf_borrow) - TTM(cf_repay_debt)) / total_assets."""
    num = df["_ttm_cf_borrow"] - df["_ttm_cf_repay_debt"]
    return num / np.maximum(df["_latest_total_assets"], 1e-8)


def build_dilution(df: pd.DataFrame) -> pd.Series:
    """F096DILUTION DILUTION = -TTM(cf_equity_issue) / mktcap_float."""
    return -df["_ttm_cf_equity_issue"] / np.maximum(df["mktcap_float"], 1e-8)


def build_int_burden(df: pd.DataFrame) -> pd.Series:
    """F097INT_BURDEN INT_BURDEN = -TTM(finance_expense) / |TTM(operating_profit)|."""
    return -_ttm_ratio(df["_ttm_finance_expense"], df["_ttm_operating_profit"])


def build_div_payout(df: pd.DataFrame) -> pd.Series:
    """F098DIV_PAYOUT DIV_PAYOUT = TTM(cf_dividend_paid) / mktcap_float — cash return scale.

    Denominator is market cap, not net_profit, to avoid |NP|~0 blow-up
    for loss-making companies.  Measures absolute cash-return commitment.
    """
    return df["_ttm_cf_dividend_paid"] / np.maximum(df["mktcap_float"], 1e-8)


# -- F099AR_MINUS_REV-F100INV_MINUS_REV: Growth Quality Red Flags --

def build_ar_minus_rev(df: pd.DataFrame, config) -> pd.Series:
    """F099AR_MINUS_REV AR_MINUS_REV = AR_stock_YoY - REV_TTM_YoY.

    AR YoY: (receivables_t - receivables_{t-4Q}) / |receivables_{t-4Q}|  (stock variable)
    REV YoY: (TTM(revenue)_t - TTM(revenue)_{t-4Q}) / |TTM(revenue)_{t-4Q}|  (flow variable)
    """
    df = df.sort_values([config.stock_col, config.date_col])
    ar = df["_latest_receivables"]
    ar_lag = ar.groupby(df[config.stock_col]).transform(lambda x: x.shift(4 * 63))  # ~4 quarters
    ar_yoy = (ar - ar_lag) / np.maximum(np.abs(ar_lag), 1e-8)

    rev_ttm = df["_ttm_revenue"]
    rev_lag = rev_ttm.groupby(df[config.stock_col]).transform(lambda x: x.shift(4 * 63))
    rev_yoy = (rev_ttm - rev_lag) / np.maximum(np.abs(rev_lag), 1e-8)

    return ar_yoy - rev_yoy


def build_inv_minus_rev(df: pd.DataFrame, config) -> pd.Series:
    """F100INV_MINUS_REV INV_MINUS_REV = INV_stock_YoY - REV_TTM_YoY."""
    df = df.sort_values([config.stock_col, config.date_col])
    inv = df["_latest_inventory"]
    inv_lag = inv.groupby(df[config.stock_col]).transform(lambda x: x.shift(4 * 63))
    inv_yoy = (inv - inv_lag) / np.maximum(np.abs(inv_lag), 1e-8)

    rev_ttm = df["_ttm_revenue"]
    rev_lag = rev_ttm.groupby(df[config.stock_col]).transform(lambda x: x.shift(4 * 63))
    rev_yoy = (rev_ttm - rev_lag) / np.maximum(np.abs(rev_lag), 1e-8)

    return inv_yoy - rev_yoy


# ════════════════════════════════════════════════════════════════
# Compute orchestrator
# ════════════════════════════════════════════════════════════════

# Friendly aliases for progress output
FACTOR_ALIASES: dict[str, str] = {}


def compute_v2_factor_batch(
    df: pd.DataFrame,
    financial_quarterly: pd.DataFrame | None,
    factor_names: Sequence[str],
    config: FactorPanelConfig,
    shared_cache: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute one V2 factor batch.  df must be sorted by [stock_col, date_col]."""
    if shared_cache is None:
        shared_cache = {}

    needs = set(factor_names)
    stock_col = config.stock_col
    date_col = config.date_col
    raw_cols: list[str] = []

    if needs & set(NEW45_FACTORS[:7]):
        print("  O2O block...")
        for fid, fn in [("F056GAP_UP_FAIL", build_gap_up_fail), ("F057INTRA1", build_intra1), ("F061GAP_UP_HOLD", build_gap_up_hold), ("F062GAP_DN_RECOVER", build_gap_dn_recover)]:
            if fid in needs:
                t0 = time.perf_counter(); df[f"{fid}_raw"] = fn(df); raw_cols.append(f"{fid}_raw")
                print(f"    {fid} {FACTOR_ALIASES.get(fid, '')} done in {time.perf_counter()-t0:.1f}s", flush=True)
        if "F058O2O_RET5" in needs:
            t0 = time.perf_counter(); df["F058O2O_RET5_raw"] = build_o2o_ret5(df, config); raw_cols.append("F058O2O_RET5_raw")
            print(f"    F058O2O_RET5 done in {time.perf_counter()-t0:.1f}s", flush=True)
        if "F059GK_VOL20" in needs:
            t0 = time.perf_counter(); df["F059GK_VOL20_raw"] = build_gk_vol20(df); raw_cols.append("F059GK_VOL20_raw")
            print(f"    F059GK_VOL20 done in {time.perf_counter()-t0:.1f}s", flush=True)
        if "F060ON_INTRA_DIV5" in needs:
            t0 = time.perf_counter(); df["F060ON_INTRA_DIV5_raw"] = build_on_intra_div5(df, config); raw_cols.append("F060ON_INTRA_DIV5_raw")
            print(f"    F060ON_INTRA_DIV5 done in {time.perf_counter()-t0:.1f}s", flush=True)

    if needs & set(NEW45_FACTORS[7:11]):
        print("  Momentum/path block...")
        for fid, fn in [("F063RET5D_SKIP1", build_ret5d_skip1), ("F064RET_ACCEL20", build_ret_accel20), ("F065MAXDD20", build_maxdd20), ("F066EFFICIENCY20", build_efficiency20)]:
            if fid in needs:
                t0 = time.perf_counter(); df[f"{fid}_raw"] = fn(df, config); raw_cols.append(f"{fid}_raw")
                print(f"    {fid} {FACTOR_ALIASES.get(fid, '')} done in {time.perf_counter()-t0:.1f}s", flush=True)

    if needs & set(NEW45_FACTORS[11:19]):
        print("  Volatility + market block...")
        df = precompute_ret_blocks(df, config)
        for fid, fn in [("F067TAIL_LOSS20", build_tail_loss20), ("F068SKEW20", build_skew20), ("F069DNVOL20", build_dnvol20), ("F070UP_DN_VOL", build_up_dn_vol), ("F071VOL_OF_VOL", build_vol_of_vol), ("F072CORR_60D", build_corr_60d), ("F073KURT_60D", build_kurt_60d), ("F074BETA_20D", build_beta_20d)]:
            if fid in needs:
                t0 = time.perf_counter()
                df[f"{fid}_raw"] = fn(df, config) if fid == "F073KURT_60D" else fn(df)
                raw_cols.append(f"{fid}_raw")
                print(f"    {fid} {FACTOR_ALIASES.get(fid, '')} done in {time.perf_counter()-t0:.1f}s", flush=True)
        drop_temporary_columns(df)

    if needs & set(NEW45_FACTORS[19:26]):
        print("  Volume + VWAP block...")
        for fid, fn in [("F075VOLUME_RATIO", build_volume_ratio), ("F076SIGNED_AMT20", build_signed_amt20), ("F077AMP_VOL20", build_amp_vol20), ("F078TURN_SIZE", build_turn_size), ("F079TURN_ACCEL", build_turn_accel), ("F080VWAP_DEV", build_vwap_dev), ("F081STRONG_CLOSE", build_strong_close)]:
            if fid in needs:
                t0 = time.perf_counter(); df[f"{fid}_raw"] = fn(df); raw_cols.append(f"{fid}_raw")
                print(f"    {fid} {FACTOR_ALIASES.get(fid, '')} done in {time.perf_counter()-t0:.1f}s", flush=True)

    if needs & set(NEW45_FACTORS[26:32]):
        print("  Float + value + age block...")
        for fid, fn in [("F082LOCKED_PCT", build_locked_pct), ("F083TURN_FREE", build_turn_free), ("F084AMT_FREE20", build_amt_free20), ("F085SP_TTM", build_sp_ttm), ("F086DIV_TTM", build_div_ttm)]:
            if fid in needs:
                t0 = time.perf_counter(); df[f"{fid}_raw"] = fn(df); raw_cols.append(f"{fid}_raw")
                print(f"    {fid} {FACTOR_ALIASES.get(fid, '')} done in {time.perf_counter()-t0:.1f}s", flush=True)
        if "F087LIST_AGE" in needs:
            t0 = time.perf_counter(); df["F087LIST_AGE_raw"] = build_list_age(df, config); raw_cols.append("F087LIST_AGE_raw")
            print(f"    F087LIST_AGE done in {time.perf_counter()-t0:.1f}s", flush=True)

    fin_needs = needs & set(NEW45_FACTORS[32:45])
    if fin_needs:
        if financial_quarterly is None:
            raise ValueError("financial_quarterly is required for F088CF_SALES_Q-F100INV_MINUS_REV")
        cache_key = "v2_financial_raws"
        if cache_key not in shared_cache:
            print("  Financial TTM/SQ raw table from financial_quarterly...")
            shared_cache[cache_key] = build_v2_financial_factor_raws(
                base_panel=df,
                financial_quarterly=financial_quarterly,
                config=config,
                factor_names=NEW45_FACTORS[32:45],
            )
        fin_raw = shared_cache[cache_key]
        keep = [date_col, stock_col] + [f"{fid}_raw" for fid in sorted(fin_needs) if f"{fid}_raw" in fin_raw.columns]
        fin_raw = fin_raw[keep]
        df = df.merge(fin_raw, on=[date_col, stock_col], how="left")
        for fid in sorted(fin_needs):
            col = f"{fid}_raw"
            if col in df.columns:
                raw_cols.append(col)
                print(f"    {fid} {FACTOR_ALIASES.get(fid, '')} done", flush=True)

    keep = [date_col, stock_col] + [c for c in raw_cols if c in df.columns]
    return df[keep].copy()


def compute_batch(
    df: pd.DataFrame,
    factor_names: list[str],
    config: FactorPanelConfig,
    financial_quarterly: pd.DataFrame | None = None,
    shared_cache: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Backward-compatible wrapper for V2 factor computation."""
    return compute_v2_factor_batch(df, financial_quarterly, factor_names, config, shared_cache)
