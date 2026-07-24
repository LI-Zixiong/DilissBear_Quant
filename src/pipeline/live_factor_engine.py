"""Memory-bounded factor calculation for newly committed live dates."""
from __future__ import annotations

import gc
import time
from dataclasses import replace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.pipeline.factor_panel import (
    FactorPanelConfig,
    compute_factor_columns,
    filter_and_select_output_columns,
    iter_factor_batches,
    run_factor_panel_checks,
    standardize_financial_keys,
    standardize_panel_keys,
    validate_no_duplicate_keys,
)
from src.pipeline.live_factor_schema import LIVE_FACTOR_REGISTRY


_HISTORY_BANDS = (2, 7, 24, 63, 120, 260, 380, 760)
# Groups whose precompute cache is expensive and should survive band boundaries.
_SHARED_CACHE_BANDS = {"capm_252", "financial_pit", "return_moments"}


def _tail(frame: pd.DataFrame, date_col: str, count: int) -> pd.DataFrame:
    dates = pd.DatetimeIndex(frame[date_col].dropna().unique()).sort_values()
    if len(dates) <= count:
        return frame
    return frame[frame[date_col] >= dates[-count]]


def compute_live_factor_rows(
    base_tail: pd.DataFrame,
    financial_quarterly: pd.DataFrame,
    output_dates: Sequence[pd.Timestamp],
    config: FactorPanelConfig | None = None,
    universe_stocks: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute new rows in dependency/history batches of at most ten factors.

    Each band receives only its required daily history. Cross-sectional
    winsorize/z-score is restricted to *output_dates*, not the full tail.
    """
    config = config or FactorPanelConfig(mode="live")
    if config.mode != "live":
        raise ValueError("compute_live_factor_rows requires mode='live'")
    config = replace(config, adj_factor_path="")

    dates = pd.DatetimeIndex(pd.to_datetime(list(output_dates))).normalize()
    if dates.empty:
        return pd.DataFrame()
    t_total = time.perf_counter()

    base = standardize_panel_keys(base_tail, config)
    financial = standardize_financial_keys(financial_quarterly, config)
    available = set(pd.DatetimeIndex(base[config.date_col].unique()).normalize())
    missing = [date for date in dates if date not in available]
    if missing:
        raise ValueError(f"Output dates missing from base tail: {missing}")

    by_band: dict[int, list[str]] = {band: [] for band in _HISTORY_BANDS}
    for factor in config.factor_names:
        history = LIVE_FACTOR_REGISTRY[factor].daily_history
        band = next(value for value in _HISTORY_BANDS if history <= value)
        by_band[band].append(factor)

    keys = [config.date_col, config.stock_col]
    meta_cols = [
        "industry_sw", "list_date", "open", "high", "low", "close",
        "pre_close", "close_adj", "adj_factor", "ret_daily",
        "mktcap_float", "mktcap_total", "volume", "amount", "trade_status",
        "limit_status", "turnover_rate", "pb", "pe_ttm",
    ]
    result: pd.DataFrame | None = None
    extra_dates = max(0, len(dates) - 1) + 2
    shared_cache: dict[str, object] = {}
    total_factors_done = 0

    for band in _HISTORY_BANDS:
        band_factors = by_band[band]
        if not band_factors:
            continue
        t0 = time.perf_counter()
        band_base = _tail(base, config.date_col, band + extra_dates)
        batches = list(iter_factor_batches(band_factors))
        total_factors_done += len(band_factors)
        for spec in batches:
            factors = tuple(spec.factors)
            if len(factors) > 10:
                raise RuntimeError(f"Live factor batch exceeds memory cap: {spec.name}")
            print(
                f"  live {band:3d}d/{spec.name}: {len(factors)} factors, "
                f"{band_base[config.date_col].nunique()} dates",
                flush=True,
            )
            part = compute_factor_columns(
                base_panel=band_base,
                financial_quarterly=financial,
                config=config,
                factor_names=factors,
                shared_cache=shared_cache,
                target_dates=dates,
            )
            part = part[part[config.date_col].isin(dates)]
            factor_cols = [factor for factor in factors if factor in part.columns]
            if result is None:
                keep_meta = [col for col in meta_cols if col in part.columns]
                result = part[keys + keep_meta + factor_cols].copy()
            else:
                result = result.merge(
                    part[keys + factor_cols],
                    on=keys, how="inner", validate="one_to_one",
                )
            del part
            gc.collect()
        elapsed = time.perf_counter() - t0
        print(
            f"  [band {band:3d}d] {len(band_factors)} factors in {len(batches)} "
            f"batches, {elapsed:.1f}s ({total_factors_done}/100 factors done)",
            flush=True,
        )

    if result is None:
        return pd.DataFrame()
    for target in config.target_names:
        result[target] = np.nan
    result = filter_and_select_output_columns(
        panel=result, config=config, factor_names=config.factor_names,
        filter_model_start=False,
    )
    result = result[result[config.date_col].isin(dates)].copy()
    if universe_stocks is not None:
        universe = {str(value).strip().zfill(6) for value in universe_stocks}
        result = result[result[config.stock_col].isin(universe)]
    validate_no_duplicate_keys(result, config, "live_factor_rows")
    t_check = time.perf_counter()
    run_factor_panel_checks(
        result, config, config.factor_names, is_incremental=True,
    )
    print(f"  [checks] {time.perf_counter() - t_check:.1f}s", flush=True)

    total_elapsed = time.perf_counter() - t_total
    print(f"  [factor_engine_total] {total_elapsed:.1f}s", flush=True)
    return result.sort_values(keys).reset_index(drop=True)
