"""
Return-frame construction and prediction-return alignment.

This module centralizes return handling for experiments, ensemble evaluation,
and strategy research.

Core idea:
    - prediction rows are signal rows
    - returns_df rows are realized return rows
    - before backtesting, predictions must be aligned to stocks with valid
      next-period returns
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_returns_from_column(
    df: pd.DataFrame,
    date_col: str,
    stock_col: str,
    source_col: str,
    return_col: str,
    normalize_date: bool = True,
) -> pd.DataFrame:
    """
    Build a return DataFrame directly from an existing return column.

    This is used when the source column already represents realized return
    at the same date, for example ret_daily.

    Parameters
    ----------
    df:
        Input panel DataFrame.

    date_col:
        Date column name.

    stock_col:
        Stock identifier column name.

    source_col:
        Column to use as return source, e.g. "ret_daily".

    return_col:
        Output return column name, e.g. "return_1d".

    normalize_date:
        Whether to normalize timestamps to midnight.

    Returns
    -------
    pd.DataFrame
        Columns: date_col, stock_col, return_col.
    """

    _require_columns(df, [date_col, stock_col, source_col])

    returns_df = df[[date_col, stock_col, source_col]].copy()
    returns_df = returns_df.rename(columns={source_col: return_col})

    returns_df[date_col] = pd.to_datetime(returns_df[date_col], errors="raise")
    if normalize_date:
        returns_df[date_col] = returns_df[date_col].dt.normalize()

    returns_df[stock_col] = returns_df[stock_col].astype(str).str.strip()
    returns_df[return_col] = pd.to_numeric(returns_df[return_col], errors="coerce")

    finite_mask = np.isfinite(returns_df[return_col].to_numpy())
    returns_df = returns_df.loc[finite_mask].reset_index(drop=True)

    _validate_no_duplicate_panel_rows(
        returns_df,
        date_col=date_col,
        stock_col=stock_col,
        name="returns_df",
    )

    return returns_df[[date_col, stock_col, return_col]]


def build_returns_frame_from_next_target(
    df: pd.DataFrame,
    date_col: str,
    stock_col: str,
    target_col: str,
    return_col: str,
    normalize_date: bool = True,
) -> pd.DataFrame:
    """
    Convert a next-period target column into a backtest-compatible return frame.

    If target_col at signal date t means the return realized from t to the next
    market trading date, this function assigns that value to the next global
    trading date.

    Example
    -------
    Signal date:
        2020-01-02 has 1d_next_raw.

    Return date:
        The value is mapped to 2020-01-03 as return_1d.

    This matches the existing project convention used by previous experiment
    scan scripts.
    """

    _require_columns(df, [date_col, stock_col, target_col])

    work = df[[date_col, stock_col, target_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="raise")
    if normalize_date:
        work[date_col] = work[date_col].dt.normalize()

    work[stock_col] = work[stock_col].astype(str).str.strip()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")

    work = work.sort_values([date_col, stock_col], kind="mergesort").reset_index(
        drop=True
    )

    unique_dates = pd.DatetimeIndex(work[date_col].drop_duplicates()).sort_values()
    if len(unique_dates) < 2:
        raise ValueError(
            f"Need at least 2 unique dates to map next target to returns, "
            f"got {len(unique_dates)}"
        )

    next_date_map = {
        unique_dates[i]: unique_dates[i + 1]
        for i in range(len(unique_dates) - 1)
    }

    returns_df = work.copy()
    returns_df[date_col] = returns_df[date_col].map(next_date_map)
    returns_df = returns_df.loc[returns_df[date_col].notna()].copy()

    returns_df = returns_df.rename(columns={target_col: return_col})
    returns_df = returns_df[[date_col, stock_col, return_col]]

    finite_mask = np.isfinite(returns_df[return_col].to_numpy())
    returns_df = returns_df.loc[finite_mask].reset_index(drop=True)

    _validate_no_duplicate_panel_rows(
        returns_df,
        date_col=date_col,
        stock_col=stock_col,
        name="returns_df",
    )

    return returns_df


def align_predictions_to_returns(
    pred_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    date_col: str,
    stock_col: str,
    pred_col: str = "y_pred",
    return_col: str = "return_1d",
    normalize_date: bool = True,
) -> pd.DataFrame:
    """
    Keep only prediction rows whose stock has a valid future return.

    The backtest engine selects stocks on signal date t, while realized returns
    are observed on a later return date. This function checks the next available
    return date after each signal date and keeps only stocks available on that
    return date.

    It does not merge return values into pred_df. It only filters pred_df so
    that run_backtest(...) can safely align predictions and returns later.
    """

    _require_columns(pred_df, [date_col, stock_col, pred_col])
    _require_columns(returns_df, [date_col, stock_col, return_col])

    pred = pred_df.copy()
    ret = returns_df.copy()

    pred[date_col] = pd.to_datetime(pred[date_col], errors="raise")
    ret[date_col] = pd.to_datetime(ret[date_col], errors="raise")

    if normalize_date:
        pred[date_col] = pred[date_col].dt.normalize()
        ret[date_col] = ret[date_col].dt.normalize()

    pred[stock_col] = pred[stock_col].astype(str).str.strip()
    ret[stock_col] = ret[stock_col].astype(str).str.strip()

    pred_dates = pd.DatetimeIndex(pred[date_col].dropna().unique()).sort_values()
    ret_dates = pd.DatetimeIndex(ret[date_col].dropna().unique()).sort_values()

    if len(pred_dates) == 0:
        return pred.iloc[0:0].copy()

    if len(ret_dates) == 0:
        return pred.iloc[0:0].copy()

    next_return_date = {}
    for signal_date in pred_dates:
        future_dates = ret_dates[ret_dates > signal_date]
        if len(future_dates) > 0:
            next_return_date[signal_date] = future_dates[0]

    if not next_return_date:
        return pred.iloc[0:0].copy()

    stocks_by_return_date = {
        d: set(ret.loc[ret[date_col] == d, stock_col])
        for d in set(next_return_date.values())
    }

    keep = pd.Series(False, index=pred.index)

    for signal_date, return_date in next_return_date.items():
        valid_stocks = stocks_by_return_date[return_date]
        mask = (pred[date_col] == signal_date) & pred[stock_col].isin(valid_stocks)
        keep.loc[mask] = True

    return pred.loc[keep].copy().reset_index(drop=True)


def build_experiment_returns(
    df: pd.DataFrame,
    date_col: str,
    stock_col: str,
    return_col: str,
    source_col: str,
    source_mode: str = "column",
    normalize_date: bool = True,
) -> pd.DataFrame:
    """
    Build experiment returns using a selected source mode.

    Parameters
    ----------
    source_mode:
        "column":
            source_col is already a realized return column at the same date.
            Current default:
                source_col="ret_daily"
                return_col="return_1d"

        "next_target":
            source_col is a next-period target at signal date t and should be
            mapped to the next global trading date.
            Example:
                source_col="1d_next_raw"
                return_col="return_1d"

    Returns
    -------
    pd.DataFrame
        Columns: date_col, stock_col, return_col.
    """

    if source_mode == "column":
        return build_returns_from_column(
            df=df,
            date_col=date_col,
            stock_col=stock_col,
            source_col=source_col,
            return_col=return_col,
            normalize_date=normalize_date,
        )

    if source_mode == "next_target":
        return build_returns_frame_from_next_target(
            df=df,
            date_col=date_col,
            stock_col=stock_col,
            target_col=source_col,
            return_col=return_col,
            normalize_date=normalize_date,
        )

    raise ValueError(
        f"Unknown source_mode: {source_mode}. "
        "Expected one of {'column', 'next_target'}."
    )


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ValueError if df misses required columns."""

    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"df missing required columns: {missing_cols}")


def _validate_no_duplicate_panel_rows(
    df: pd.DataFrame,
    date_col: str,
    stock_col: str,
    name: str,
) -> None:
    """Validate that date-stock rows are unique."""

    duplicated = df.duplicated(subset=[date_col, stock_col])
    if duplicated.any():
        n_dup = int(duplicated.sum())
        sample = df.loc[duplicated, [date_col, stock_col]].head(5)
        raise ValueError(
            f"{name} contains duplicated {date_col}-{stock_col} rows: "
            f"{n_dup} duplicates. Sample:\n{sample}"
        )