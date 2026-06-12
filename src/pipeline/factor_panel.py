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

import json
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
    "F020BP",
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

DEFAULT_FACTOR_NAMES: tuple[str, ...] = BARRA12_FACTORS + NEW12_FACTORS + TECH24_FACTORS + FIN6_FACTORS + IND1_FACTORS

FACTOR_ALIAS: dict[str, str] = {
    "F001SIZE": "SIZE",
    "F002SIZENL": "SIZENL",
    "F003LIQUIDITY": "LIQUIDITY",
    "F004BETA": "BETA",
    "F005RESVOL": "RESVOL",
    "F006MOMENTUM": "MOMENTUM",
    "F007LTREV": "LTREV",
    "F008STREV": "STREV",
    "F009LEVERAGE": "LEVERAGE",
    "F010VALUE": "VALUE",
    "F011EARNYLD": "EARNYLD",
    "F012GROWTH": "GROWTH",
    "F013REV5": "F1_rev5",
    "F014MOM120_20": "F2_mom120_20",
    "F015VOLREV": "F3_volrev",
    "F016MAXRET": "F4_maxret",
    "F017IVOL": "F5_ivol",
    "F018AMIHUD": "F6_amihud",
    "F019COSTDEV": "F7_costdev",
    "F020BP": "F8_bp",
    "F021CFP": "F9_cfp",
    "F022GPTA": "F10_gpta",
    "F023ACCRUAL": "F11_accrual",
    "F024ASSETGR": "F12_assetgr",
    "F025GAP": "F25_gap",
    "F026KLEN": "F26_klen",
    "F027KUP": "F27_kup",
    "F028KLOW": "F28_klow",
    "F029KSFT": "F29_ksft",
    "F030RSV20": "F30_rsv20",
    "F031RSV60": "F31_rsv60",
    "F032RANGEZ20": "F32_rangez20",
    "F033GAPREV5": "F33_gaprev5",
    "F034HIGHDEV20": "F34_highdev20",
    "F035LOWDEV20": "F35_lowdev20",
    "F036VOLSHOCK5": "F36_volshock5",
    "F037VOLSHOCK20": "F37_volshock20",
    "F038TURNZ20": "F38_turnz20",
    "F039VSTD20": "F39_vstd20",
    "F040PVCORR20": "F40_pvcorr20",
    "F041RETVOLCORR20": "F41_retvolcorr20",
    "F042AMTCORR20": "F42_amtcorr20",
    "F043SLOPE20": "F43_slope20",
    "F044RSQR20": "F44_rsqr20",
    "F045RESI20": "F45_resi20",
    "F046LIMITUP20": "F46_limitup20",
    "F047LIMITDN20": "F47_limitdn20",
    "F048LIMITSTREAKUP": "F48_limitstreakup",
    "F049ROE": "F49_roe",
    "F050ROA": "F50_roa",
    "F051GPM": "F51_gpm",
    "F052CFOA": "F52_cfoa",
    "F053RD_INTENSITY": "F53_rd_intensity",
    "F054RECEIVABLE_RATIO": "F54_receivable_ratio",
    "F055IND": "F55_ind",
}


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

    # Metadata / version
    factor_version: str = "v1_24f"
    seed: int = 42


# ---------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------


def compute_factor_panel_full(
    config: FactorPanelConfig | None = None,
) -> dict[str, Any]:
    """
    Compute full factor panel from base panel and quarterly financial panel.

    Factors are computed in batches of 10 to keep peak memory low.
    Each batch is saved as a column extension to the output parquet.
    """

    if config is None:
        config = FactorPanelConfig()

    base = load_base_panel(config)
    financial = load_financial_quarterly_panel(config)

    factor_names = tuple(config.factor_names)
    batch_size = 10
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_raw_cols: list[str] = []
    first_batch = True
    n_batches = (len(factor_names) + batch_size - 1) // batch_size

    for b in range(0, len(factor_names), batch_size):
        batch = factor_names[b:b + batch_size]
        batch_num = b // batch_size + 1
        print(f"\n--- Batch {batch_num}/{n_batches}: {', '.join(batch)} ---")

        panel = compute_factor_columns(
            base_panel=base,
            financial_quarterly=financial,
            config=config,
            factor_names=batch,
        )

        panel = filter_and_select_output_columns(
            panel=panel,
            config=config,
            factor_names=batch,
            include_targets=False,
        )

        if first_batch:
            panel.to_parquet(output_path, index=False)
            first_batch = False
        else:
            new_cols = [config.date_col, config.stock_col] + [f for f in batch if f in panel.columns]
            existing = pd.read_parquet(output_path)
            existing[config.date_col] = pd.to_datetime(existing[config.date_col]).dt.normalize()
            existing[config.stock_col] = existing[config.stock_col].astype(str).str.strip().str.zfill(6)
            merged = existing.merge(
                panel[new_cols],
                on=[config.date_col, config.stock_col],
                how="inner",
            )
            merged.to_parquet(output_path, index=False)
            del existing, merged

        for fn in batch:
            if fn in panel.columns:
                all_raw_cols.append(fn)
        # Free memory
        del panel

    # Reload merged panel, add targets, final save
    panel = pd.read_parquet(output_path)
    panel[config.date_col] = pd.to_datetime(panel[config.date_col]).dt.normalize()
    panel[config.stock_col] = panel[config.stock_col].astype(str).str.strip().str.zfill(6)

    if config.mode == "research":
        targets = build_targets(panel, config)
        for col in config.target_names:
            if col in targets.columns:
                panel[col] = targets[col]
    elif config.mode == "live":
        for col in config.target_names:
            panel[col] = np.nan

    panel = filter_and_select_output_columns(
        panel=panel,
        config=config,
        factor_names=config.factor_names,
    )

    audit = run_factor_panel_checks(panel, config, config.factor_names)

    panel.to_parquet(output_path, index=False)

    metadata = build_factor_metadata(
        panel=panel,
        config=config,
        audit=audit,
    )
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
        f"dates={panel[config.date_col].min()}~{panel[config.date_col].max()}"
    )

    return {
        "factor_panel_path": str(output_path),
        "metadata_path": str(metadata_path),
        "audit": audit,
    }


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

    if config.mode == "research":
        targets = build_targets(recomputed, config)
        for col in config.target_names:
            if col in targets.columns:
                recomputed[col] = targets[col]
    elif config.mode == "live":
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

    new_rows = recomputed[recomputed[config.date_col] > last_factor_date].copy()

    if new_rows.empty:
        print("No new factor rows to append.")
        return existing

    key_cols = [config.stock_col, config.date_col]

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
) -> pd.DataFrame:
    """
    Compute selected factor columns.

    Returns a copy of base_panel with raw/winsor/final factor columns added.
    """

    factor_names = tuple(factor_names)
    unknown = sorted(set(factor_names) - set(DEFAULT_FACTOR_NAMES))
    if unknown:
        raise ValueError(f"Unsupported factor names: {unknown}")

    df = standardize_panel_keys(base_panel, config)
    df = df.sort_values([config.stock_col, config.date_col]).reset_index(drop=True)

    # Prune rows: keep only warmup window (4y before model_start) + model period
    warmup_start = pd.Timestamp(config.model_start_date) - pd.DateOffset(years=4)
    df = df[df[config.date_col] >= warmup_start].copy()

    # Prune columns — only keep what factor computation + downstream needs
    keep_cols = {
        config.date_col, config.stock_col,
        "open", "high", "low", "close", "pre_close", "close_adj",
        "ret_daily", "mkt_ret_vw",
        "mktcap_total", "mktcap_float", "amount", "volume",
        "total_assets", "total_liabilities", "equity_parent",
        "receivables", "rd_expense",
        "trade_status", "limit_status", "turnover_rate",
        "pb", "pe_ttm", "list_date",
    }
    keep_cols |= {c for c in df.columns if c.startswith("industry")}
    keep_cols = {c for c in keep_cols if c in df.columns}
    df = df[list(keep_cols)]

    needs = set(factor_names)

    # Shared CAPM residual for residual-based factors.
    needs_resid = bool(
        needs.intersection(
            {
                "F013REV5",
                "F014MOM120_20",
                "F017IVOL",
            }
        )
    )
    if needs_resid:
        print("  CAPM residual...")
        df["_capm_resid"] = build_capm_residual(df, config)

    # Group A: old price / volume factors.
    if "F001SIZE" in needs or "F002SIZENL" in needs:
        print("  F001SIZE...")
        df["F001SIZE_raw"] = build_size(df, config)

    if "F002SIZENL" in needs:
        print("  F002SIZENL...")
        if "F001SIZE_raw" not in df.columns:
            df["F001SIZE_raw"] = build_size(df, config)
        df["F002SIZENL_raw"] = build_sizenl(df, config, df["F001SIZE_raw"])

    if "F003LIQUIDITY" in needs:
        print("  F003LIQUIDITY...")
        df["F003LIQUIDITY_raw"] = build_liquidity(df, config)

    if "F004BETA" in needs or "F005RESVOL" in needs:
        print("  F004BETA + F005RESVOL...")
        beta, resvol = build_beta_resvol(
            df,
            config,
            window=config.beta_window,
            min_periods=config.beta_min_periods,
        )
        if "F004BETA" in needs:
            df["F004BETA_raw"] = beta
        if "F005RESVOL" in needs:
            df["F005RESVOL_raw"] = resvol

    if "F006MOMENTUM" in needs:
        print("  F006MOMENTUM...")
        df["F006MOMENTUM_raw"] = build_momentum(df, config)

    if "F007LTREV" in needs:
        print("  F007LTREV...")
        df["F007LTREV_raw"] = build_ltrev(df, config)

    if "F008STREV" in needs:
        print("  F008STREV...")
        df["F008STREV_raw"] = build_strev(df, config)

    # Group B: old balance-sheet factors.
    if "F009LEVERAGE" in needs:
        print("  F009LEVERAGE...")
        df["F009LEVERAGE_raw"] = build_leverage(df, config)

    if "F010VALUE" in needs:
        print("  F010VALUE...")
        df["F010VALUE_raw"] = build_value(df, config)

    # New residual / price / volume factors.
    if "F013REV5" in needs:
        print("  F013REV5...")
        df["F013REV5_raw"] = build_rev5(df, config)

    if "F014MOM120_20" in needs:
        print("  F014MOM120_20...")
        df["F014MOM120_20_raw"] = build_resid_mom120_20(df, config)

    if "F015VOLREV" in needs:
        print("  F015VOLREV...")
        df["F015VOLREV_raw"] = build_volrev(df, config)

    if "F016MAXRET" in needs:
        print("  F016MAXRET...")
        df["F016MAXRET_raw"] = build_maxret(df, config)

    if "F017IVOL" in needs:
        print("  F017IVOL...")
        df["F017IVOL_raw"] = build_ivol(df, config)

    if "F018AMIHUD" in needs:
        print("  F018AMIHUD...")
        df["F018AMIHUD_raw"] = build_amihud(df, config)

    if "F019COSTDEV" in needs:
        print("  F019COSTDEV...")
        df["F019COSTDEV_raw"] = build_costdev(df, config)

    if "F020BP" in needs:
        # F020SP = TTM revenue / mktcap, computed in build_ttm_factor_raws
        pass

    # Group E: technical / volume / stressed factors (F025-F048)
    _compute_tech24_factors(df, config, factor_names)

    # TTM-based factors.
    ttm_needed = {
        "F011EARNYLD",
        "F012GROWTH",
        "F020BP",
        "F021CFP",
        "F022GPTA",
        "F023ACCRUAL",
        "F024ASSETGR",
        "F049ROE",
        "F050ROA",
        "F051GPM",
        "F052CFOA",
        "F053RD_INTENSITY",
        "F054RECEIVABLE_RATIO",
    }
    if needs.intersection(ttm_needed):
        print("  TTM financial factors...")
        ttm_df = build_ttm_factor_raws(
            base_panel=df,
            financial_quarterly=financial_quarterly,
            config=config,
        )
        df = df.merge(ttm_df, on=[config.date_col, config.stock_col], how="left")

    if "F055IND" in needs:
        df["F055IND_raw"] = (
            df["industry_sw"].fillna(-1).astype(float)
            if "industry_sw" in df.columns
            else -1.0
        )

    # Drop temporary helper columns to save memory before winsorize
    tmp_cols = [c for c in df.columns if c.startswith("_")]
    if tmp_cols:
        df = df.drop(columns=tmp_cols)

    # Winsorize in batches to keep memory under limit
    # Each batch: process, then drop _w and _raw of this batch, move to next
    batch_size = 24
    fn_list = list(factor_names)
    for start in range(0, len(fn_list), batch_size):
        batch = fn_list[start:start + batch_size]
        df = winsorize_zscore(df=df, factor_names=batch, config=config)
        # Drop intermediate columns for this batch to free memory
        drop_cols = [c for c in df.columns
                     if any(c == f"{f}_w" or (c == f"{f}_raw" and not config.save_raw_factors) for f in batch)]
        if drop_cols:
            df = df.drop(columns=drop_cols)

    return df


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
    """F002SIZENL proxy: cross-sectional rank of SIZE minus 0.5."""

    return size_raw.groupby(df[config.date_col]).rank(pct=True) - 0.5


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
    """F006MOMENTUM = close_adj[t-21] / close_adj[t-252] - 1."""

    group = df.groupby(config.stock_col)["close_adj"]
    return group.shift(21) / group.shift(252) - 1


def build_ltrev(df: pd.DataFrame, config: FactorPanelConfig) -> pd.Series:
    """F007LTREV: long-term reversal ensemble based on 504/630/756-day windows."""

    group = df.groupby(config.stock_col)["close_adj"]

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
    """F008STREV = negative short-term return."""

    group = df.groupby(config.stock_col)["close_adj"]
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
    result = -(df["close"] / vwap_120d.where(vwap_120d > 0) - 1)
    result[~np.isfinite(result)] = np.nan
    return result


def build_bp(df: pd.DataFrame, config: FactorPanelConfig) -> pd.Series:
    """F020BP = equity_parent / market cap."""

    mkt = df["mktcap_total"].fillna(df["mktcap_float"])
    result = df["equity_parent"] / mkt.where(mkt > 0)
    result[~np.isfinite(result)] = np.nan
    return result


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
    """Rolling Pearson correlation between two columns, per stock."""
    result = pd.Series(np.nan, index=df.index, dtype=np.float64)
    for _sid, grp in df.groupby(config.stock_col, sort=False):
        idx = grp.index
        r = grp[col_a].rolling(window, min_periods=max(10, window // 2)).corr(grp[col_b])
        result.loc[idx] = r.values
    return result


def _compute_tech24_factors(
    df: pd.DataFrame,
    config: FactorPanelConfig,
    factor_names: Sequence[str],
) -> None:
    """Compute F025-F048 factors. Modifies df in-place."""
    needs = set(factor_names)
    eps = 1e-12
    g = df.groupby(config.stock_col)

    # Shared: adj open/high/low
    if needs & {"F030RSV20", "F031RSV60", "F034HIGHDEV20", "F035LOWDEV20"}:
        df["_adj_factor"] = df["close_adj"] / df["close"].clip(eps)
        df["_open_adj"] = df["open"] * df["_adj_factor"]
        df["_high_adj"] = df["high"] * df["_adj_factor"]
        df["_low_adj"] = df["low"] * df["_adj_factor"]

    # Shared: time_index for regression factors
    if needs & {"F043SLOPE20", "F044RSQR20", "F045RESI20"}:
        ti = np.arange(20).reshape(1, -1).astype(np.float64)
        ti_mean = ti.mean()
        ti_denom = (ti * ti).sum() - 20 * ti_mean * ti_mean

    # Shared: limit_streak for F048
    if "F048LIMITSTREAKUP" in needs:
        pass  # computed inline

    if "F025GAP" in needs:
        print("  F025GAP...")
        df["F025GAP_raw"] = df["open"] / df["pre_close"].clip(eps) - 1

    if "F026KLEN" in needs:
        print("  F026KLEN...")
        df["F026KLEN_raw"] = (df["high"] - df["low"]) / (df["open"] + eps)

    if "F027KUP" in needs:
        print("  F027KUP...")
        df["F027KUP_raw"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["open"] + eps)

    if "F028KLOW" in needs:
        print("  F028KLOW...")
        df["F028KLOW_raw"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["open"] + eps)

    if "F029KSFT" in needs:
        print("  F029KSFT...")
        df["F029KSFT_raw"] = (2 * df["close"] - df["high"] - df["low"]) / (df["open"] + eps)

    if "F030RSV20" in needs:
        print("  F030RSV20...")
        lo20 = g["_low_adj"].transform(lambda x: x.rolling(20, min_periods=10).min())
        hi20 = g["_high_adj"].transform(lambda x: x.rolling(20, min_periods=10).max())
        df["F030RSV20_raw"] = (df["close_adj"] - lo20) / (hi20 - lo20 + eps)

    if "F031RSV60" in needs:
        print("  F031RSV60...")
        lo60 = g["_low_adj"].transform(lambda x: x.rolling(60, min_periods=30).min())
        hi60 = g["_high_adj"].transform(lambda x: x.rolling(60, min_periods=30).max())
        df["F031RSV60_raw"] = (df["close_adj"] - lo60) / (hi60 - lo60 + eps)

    if "F032RANGEZ20" in needs:
        print("  F032RANGEZ20...")
        df["_rng"] = (df["high"] - df["low"]) / (df["open"] + eps)
        rng20_mu = g["_rng"].transform(lambda x: x.rolling(20, min_periods=10).mean())
        rng20_sd = g["_rng"].transform(lambda x: x.rolling(20, min_periods=10).std())
        df["F032RANGEZ20_raw"] = (df["_rng"] - rng20_mu) / (rng20_sd + eps)

    if "F033GAPREV5" in needs:
        print("  F033GAPREV5...")
        df["_gap"] = df["open"] / df["pre_close"].clip(eps) - 1
        df["F033GAPREV5_raw"] = -g["_gap"].transform(lambda x: x.rolling(5, min_periods=3).sum())

    if "F034HIGHDEV20" in needs:
        print("  F034HIGHDEV20...")
        hi20_max = g["_high_adj"].transform(lambda x: x.rolling(20, min_periods=10).max())
        df["F034HIGHDEV20_raw"] = df["close_adj"] / (hi20_max + eps) - 1

    if "F035LOWDEV20" in needs:
        print("  F035LOWDEV20...")
        lo20_min = g["_low_adj"].transform(lambda x: x.rolling(20, min_periods=10).min())
        df["F035LOWDEV20_raw"] = df["close_adj"] / (lo20_min + eps) - 1

    if "F036VOLSHOCK5" in needs:
        print("  F036VOLSHOCK5...")
        amt5 = g["amount"].transform(lambda x: x.rolling(5, min_periods=3).mean())
        df["F036VOLSHOCK5_raw"] = np.log(df["amount"] / (amt5 + eps) + eps)

    if "F037VOLSHOCK20" in needs:
        print("  F037VOLSHOCK20...")
        amt20 = g["amount"].transform(lambda x: x.rolling(20, min_periods=10).mean())
        df["F037VOLSHOCK20_raw"] = np.log(df["amount"] / (amt20 + eps) + eps)

    if "F038TURNZ20" in needs:
        print("  F038TURNZ20...")
        tf_col = "turnover_rate_f" if "turnover_rate_f" in df.columns else "turnover_rate"
        tf_mu = g[tf_col].transform(lambda x: x.rolling(20, min_periods=10).mean())
        tf_sd = g[tf_col].transform(lambda x: x.rolling(20, min_periods=10).std())
        df["F038TURNZ20_raw"] = (df[tf_col] - tf_mu) / (tf_sd + eps)

    if "F039VSTD20" in needs:
        print("  F039VSTD20...")
        df["_logvol"] = np.log(df["volume"] + 1)
        df["F039VSTD20_raw"] = g["_logvol"].transform(lambda x: x.rolling(20, min_periods=10).std())

    if "F040PVCORR20" in needs:
        print("  F040PVCORR20...")
        df["_logvol"] = np.log(df["volume"] + 1)
        df["F040PVCORR20_raw"] = _rolling_corr(
            df, config, "close_adj", "_logvol", 20, eps
        )

    if "F041RETVOLCORR20" in needs:
        print("  F041RETVOLCORR20...")
        df["_vol_chg"] = np.log(df["volume"] / (g["volume"].shift(1) + eps) + 1)
        df["F041RETVOLCORR20_raw"] = _rolling_corr(
            df, config, "ret_daily", "_vol_chg", 20, eps
        )

    if "F042AMTCORR20" in needs:
        print("  F042AMTCORR20...")
        df["_logamt"] = np.log(df["amount"] + 1)
        df["F042AMTCORR20_raw"] = _rolling_corr(
            df, config, "ret_daily", "_logamt", 20, eps
        )

    if "F043SLOPE20" in needs:
        print("  F043SLOPE20...")
        slope = g["close_adj"].transform(
            lambda x: x.rolling(20, min_periods=10).apply(
                lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] if len(y) >= 10 else np.nan,
                raw=True,
            )
        )
        df["F043SLOPE20_raw"] = slope / (df["close_adj"] + eps)

    if "F044RSQR20" in needs:
        print("  F044RSQR20...")
        df["F044RSQR20_raw"] = g["close_adj"].transform(
            lambda x: x.rolling(20, min_periods=10).apply(
                lambda y: (
                    (np.corrcoef(np.arange(len(y)), y)[0, 1] ** 2)
                    if len(y) >= 10 and np.std(y) > eps
                    else np.nan
                ),
                raw=True,
            )
        )

    if "F045RESI20" in needs:
        print("  F045RESI20...")
        resid = g["close_adj"].transform(
            lambda x: x.rolling(20, min_periods=10).apply(
                lambda y: (
                    np.polyfit(np.arange(len(y)), y, 1)[1]
                    - np.polyval(np.polyfit(np.arange(len(y)), y, 1), len(y) - 1)
                    if len(y) >= 10
                    else np.nan
                ),
                raw=True,
            )
        )
        df["F045RESI20_raw"] = resid / (df["close_adj"] + eps)

    if "F046LIMITUP20" in needs:
        print("  F046LIMITUP20...")
        df["F046LIMITUP20_raw"] = g["limit_status"].transform(
            lambda x: (x == 1).rolling(20, min_periods=5).sum()
        )

    if "F047LIMITDN20" in needs:
        print("  F047LIMITDN20...")
        df["F047LIMITDN20_raw"] = g["limit_status"].transform(
            lambda x: (x == -1).rolling(20, min_periods=5).sum()
        )

    if "F048LIMITSTREAKUP" in needs:
        print("  F048LIMITSTREAKUP...")
        is_limit = df["limit_status"] == 1
        streak = is_limit.groupby(df[config.stock_col]).transform(
            lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
        )
        df["F048LIMITSTREAKUP_raw"] = streak


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

    # F020SP: TTM revenue / market cap (replaces F020BP)
    if "revenue_ttm" in filled.columns:
        filled["F020BP_raw"] = filled["revenue_ttm"] / mkt.where(mkt > 0)

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
        "F020BP_raw",
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
# Targets
# ---------------------------------------------------------------------


def build_targets(df: pd.DataFrame, config: FactorPanelConfig) -> pd.DataFrame:
    """
    Build forward-return targets.

    1d_next_raw:
        signal on t, buy at t+1 open, sell at t+2 open.

    5d_next_raw:
        signal on t, buy at t+1 open, sell around t+7 open.
    """

    group = df.groupby(config.stock_col)["open"]

    result = pd.DataFrame(index=df.index)
    result["1d_next_raw"] = group.shift(-2) / group.shift(-1) - 1
    result["5d_next_raw"] = group.shift(-6) / group.shift(-1) - 1

    return result


# ---------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------


def winsorize_zscore(
    df: pd.DataFrame,
    factor_names: list[str],
    config: FactorPanelConfig,
) -> pd.DataFrame:
    """
    Cross-sectional winsorize and z-score.

    Optional industry neutralization is supported but off by default.
    """

    neutralize_set = set(config.neutralize_factors)

    categorical_factors = {"F055IND"}

    for name in factor_names:
        raw_col = f"{name}_raw"

        if raw_col not in df.columns:
            print(f"    [WARNING] {raw_col} not found, skipping")
            continue

        if name in categorical_factors:
            df[name] = df[raw_col]
            continue

        lower = df.groupby(config.date_col)[raw_col].transform(
            lambda x: x.quantile(config.winsorize_lower)
        )
        upper = df.groupby(config.date_col)[raw_col].transform(
            lambda x: x.quantile(config.winsorize_upper)
        )

        w_col = f"{name}_w"
        df[w_col] = df[raw_col].clip(lower, upper)

        if (
            config.industry_col
            and config.industry_col in df.columns
            and name in neutralize_set
        ):
            has_industry = df[config.industry_col].notna()
            if has_industry.any():
                neutralized = df[w_col].copy()

                group_keys = [config.date_col, config.industry_col]
                mu_ind = df.loc[has_industry].groupby(group_keys)[w_col].transform("mean")
                sd_ind = df.loc[has_industry].groupby(group_keys)[w_col].transform("std")

                neutralized.loc[has_industry] = (
                    df.loc[has_industry, w_col] - mu_ind
                ) / sd_ind.where(sd_ind > 1e-10, 1.0)

                df[w_col] = neutralized

        mu = df.groupby(config.date_col)[w_col].transform("mean")
        sd = df.groupby(config.date_col)[w_col].transform("std")
        df[name] = (df[w_col] - mu) / sd.where(sd > 1e-10, 1.0)

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

    df = panel.copy()

    if filter_model_start:
        df = df[df[config.date_col] >= pd.Timestamp(config.model_start_date)].copy()

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
        "close",
        "close_adj",
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

    out = df.copy()
    out[config.stock_col] = out[config.stock_col].astype(str).str.strip().str.zfill(6)
    out[config.date_col] = pd.to_datetime(out[config.date_col], errors="raise").dt.normalize()
    return out


def standardize_financial_keys(
    df: pd.DataFrame,
    config: FactorPanelConfig,
) -> pd.DataFrame:
    """Standardize quarterly financial keys."""

    out = df.copy()
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