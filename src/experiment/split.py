"""
Date-based experiment split utilities.

The split is based on unique trading dates, not row counts.
This prevents the same trading day from appearing in multiple subsets.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def split_panel_by_date_ratio(
    df: pd.DataFrame,
    date_col: str,
    stock_col: str,
    split_ratio: Sequence[float] = (0.7, 0.1, 0.2),
    purge: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """
    Split a panel DataFrame into chronological train / valid / test sets.

    Parameters
    ----------
    df:
        Cleaned panel DataFrame.

    date_col:
        Date column name.

    stock_col:
        Stock identifier column name.

    split_ratio:
        Three positive ratios for train, valid, and test.
        Must sum to 1.0.

        Examples:
            (0.7, 0.1, 0.2) -> 70% train, 10% valid, 20% test
            (0.6, 0.1, 0.3) -> 60% train, 10% valid, 30% test
            (0.6, 0.2, 0.2) -> 60% train, 20% valid, 20% test

    purge:
        Number of trailing signal dates removed from train and valid.  This
        prevents a forward-return target near a boundary from using prices in
        the following split.

    Returns
    -------
    train_df, valid_df, test_df, train_end, valid_end
    """

    _validate_split_inputs(
        df=df,
        date_col=date_col,
        stock_col=stock_col,
        split_ratio=split_ratio,
    )
    if purge < 0:
        raise ValueError(f"purge must be >= 0, got {purge}")

    train_ratio, valid_ratio, _ = split_ratio

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="raise")
    work = work.sort_values([date_col, stock_col], kind="mergesort").reset_index(
        drop=True
    )

    unique_dates = pd.DatetimeIndex(work[date_col].drop_duplicates()).sort_values()
    n_dates = len(unique_dates)

    if n_dates < 3:
        raise ValueError(
            f"Need at least 3 unique dates for train/valid/test split, "
            f"got {n_dates}"
        )

    train_end_idx = int(n_dates * train_ratio)
    valid_end_idx = int(n_dates * (train_ratio + valid_ratio))

    if train_end_idx <= 0:
        raise ValueError(
            f"Train split is empty. n_dates={n_dates}, "
            f"split_ratio={split_ratio}, train_end_idx={train_end_idx}"
        )

    if valid_end_idx <= train_end_idx:
        raise ValueError(
            f"Valid split is empty. n_dates={n_dates}, "
            f"split_ratio={split_ratio}, train_end_idx={train_end_idx}, "
            f"valid_end_idx={valid_end_idx}"
        )

    if valid_end_idx >= n_dates:
        raise ValueError(
            f"Test split is empty. n_dates={n_dates}, "
            f"split_ratio={split_ratio}, valid_end_idx={valid_end_idx}"
        )

    train_dates = unique_dates[:train_end_idx]
    valid_dates = unique_dates[train_end_idx:valid_end_idx]
    test_dates = unique_dates[valid_end_idx:]

    if purge > 0:
        if len(train_dates) <= purge or len(valid_dates) <= purge:
            raise ValueError(
                "purge would empty a split. "
                f"train_dates={len(train_dates)}, valid_dates={len(valid_dates)}, "
                f"purge={purge}"
            )
        train_dates = train_dates[:-purge]
        valid_dates = valid_dates[:-purge]

    train_df = work[work[date_col].isin(train_dates)].copy().reset_index(drop=True)
    valid_df = work[work[date_col].isin(valid_dates)].copy().reset_index(drop=True)
    test_df = work[work[date_col].isin(test_dates)].copy().reset_index(drop=True)

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError(
            "Date split produced an empty subset. "
            f"train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}, "
            f"split_ratio={split_ratio}"
        )

    train_end = str(pd.Timestamp(train_dates[-1]))
    valid_end = str(pd.Timestamp(valid_dates[-1]))

    return train_df, valid_df, test_df, train_end, valid_end


def _validate_split_inputs(
    df: pd.DataFrame,
    date_col: str,
    stock_col: str,
    split_ratio: Sequence[float],
) -> None:
    """Validate split inputs before performing the split."""

    missing_cols = [col for col in (date_col, stock_col) if col not in df.columns]
    if missing_cols:
        raise ValueError(f"df missing required columns: {missing_cols}")

    if len(split_ratio) != 3:
        raise ValueError(
            f"split_ratio must have three values: train, valid, test. "
            f"Got {split_ratio}"
        )

    if any(x <= 0 for x in split_ratio):
        raise ValueError(
            f"All split_ratio values must be positive. Got {split_ratio}"
        )

    ratio_sum = float(sum(split_ratio))
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(
            f"split_ratio must sum to 1.0. Got {split_ratio}, sum={ratio_sum}"
        )


def split_panel_by_dates(
    df: pd.DataFrame,
    date_col: str,
    stock_col: str,
    boundaries: tuple[str, str, str, str, str, str],
    purge: int = 6,
    target_horizon: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """Split panel by explicit date boundaries with purge for walk-forward.

    All boundaries are half-open [start, end).

    Purge: Train's last `purge` signal dates are removed (label must not
    overlap with Valid). Likewise Valid's last `purge` signal dates are
    removed (label must not overlap with Test).

    Test right-boundary is NOT purged, but signal dates too close to the
    data end are dropped so that `ret_daily` can be honoured.
    """
    train_start, train_end, valid_start, valid_end, test_start, test_end = boundaries

    # -- validate boundaries --
    if purge < 0:
        raise ValueError(f"purge must be >= 0, got {purge}")
    if target_horizon < 0:
        raise ValueError(f"target_horizon must be >= 0, got {target_horizon}")
    def _ts(s: str) -> pd.Timestamp:
        return pd.Timestamp(s)
    if not (_ts(train_start) < _ts(train_end) <= _ts(valid_start)
            < _ts(valid_end) <= _ts(test_start) < _ts(test_end)):
        raise ValueError(
            "Boundaries must satisfy: "
            "train_start < train_end <= valid_start < valid_end <= test_start < test_end. "
            f"Got: train=[{train_start},{train_end}) "
            f"valid=[{valid_start},{valid_end}) "
            f"test=[{test_start},{test_end})"
        )

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="raise")
    work = work.sort_values([date_col, stock_col], kind="mergesort").reset_index(drop=True)

    all_dates = pd.DatetimeIndex(work[date_col].drop_duplicates()).sort_values()

    # -- helpers --
    def _dates_in(begin: str, end: str) -> pd.DatetimeIndex:
        return all_dates[(all_dates >= pd.Timestamp(begin)) & (all_dates < pd.Timestamp(end))]

    def _select(begin: str, end: str) -> pd.DataFrame:
        d = _dates_in(begin, end)
        return work[work[date_col].isin(d)].copy().reset_index(drop=True)

    train_dates = _dates_in(train_start, train_end)
    valid_dates = _dates_in(valid_start, valid_end)
    test_dates = _dates_in(test_start, test_end)

    # -- purge --
    if purge > 0:
        if len(train_dates) > purge:
            train_dates = train_dates[:-purge]
        if len(valid_dates) > purge:
            valid_dates = valid_dates[:-purge]

    # Test: drop dates too close to data end for target label to settle.
    # Use trading-day index, not calendar days — weekends/holidays would bias.
    buffer = target_horizon + 1
    if len(all_dates) <= buffer:
        raise ValueError(
            f"Not enough trading dates ({len(all_dates)}) for "
            f"target_horizon={target_horizon} buffer ({buffer})"
        )
    last_usable = all_dates[-(buffer + 1)]
    test_dates = test_dates[test_dates <= last_usable]

    train_df = work[work[date_col].isin(train_dates)].copy().reset_index(drop=True)
    valid_df = work[work[date_col].isin(valid_dates)].copy().reset_index(drop=True)
    test_df = work[work[date_col].isin(test_dates)].copy().reset_index(drop=True)

    for name, df_, dates_ in [("train", train_df, train_dates),
                               ("valid", valid_df, valid_dates),
                               ("test", test_df, test_dates)]:
        if df_.empty:
            raise ValueError(
                f"{name} split is empty: {len(dates_)} dates matched after purge, "
                f"but no rows in panel. boundaries={boundaries}, purge={purge}"
            )

    return train_df, valid_df, test_df, str(train_dates[-1]), str(valid_dates[-1])
