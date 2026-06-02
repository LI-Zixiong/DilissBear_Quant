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