"""
Backtest engine for converting predictions into portfolio returns.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd
import numpy as np

from src.backtest.metrics import (
    calculate_turnover,
    cumulative_nav,
    summarize_backtest,
)
from src.backtest.portfolio import PortfolioConfig, build_portfolio_weights


@dataclass
class TransactionCostConfig:
    buy_cost: float = 0.0003
    sell_cost: float = 0.0008

    def __post_init__(self) -> None:
        if not (0.0 <= self.buy_cost < 1.0):
            raise ValueError(f"buy_cost must be in [0, 1), got {self.buy_cost}")
        if not (0.0 <= self.sell_cost < 1.0):
            raise ValueError(f"sell_cost must be in [0, 1), got {self.sell_cost}")

def _validate_backtest_inputs(
    pred_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    portfolio_config: PortfolioConfig,
    return_col: str,
    date_col: str,
    stock_col: str,
) -> None:
    """
    Validate backtest input DataFrames.
    """
    if not isinstance(pred_df, pd.DataFrame):
        raise TypeError(f"pred_df must be a DataFrame, got {type(pred_df).__name__}")

    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError(
            f"returns_df must be a DataFrame, got {type(returns_df).__name__}"
        )

    if pred_df.empty:
        raise ValueError("pred_df must not be empty")

    if returns_df.empty:
        raise ValueError("returns_df must not be empty")

    if portfolio_config.stock_col != stock_col:
        raise ValueError(
            "portfolio_config.stock_col must match stock_col. "
            f"Got {portfolio_config.stock_col} vs {stock_col}."
        )

    required_pred_cols = [date_col, stock_col]

    if portfolio_config.strategy == "top_n":
        required_pred_cols.append(portfolio_config.pred_col)

    missing_pred_cols = [col for col in required_pred_cols if col not in pred_df.columns]
    if missing_pred_cols:
        raise ValueError(f"pred_df missing required columns: {missing_pred_cols}")

    required_return_cols = [date_col, stock_col, return_col]
    missing_return_cols = [
        col for col in required_return_cols if col not in returns_df.columns
    ]
    if missing_return_cols:
        raise ValueError(f"returns_df missing required columns: {missing_return_cols}")

    if pred_df[[date_col, stock_col]].isna().any().any():
        raise ValueError(f"pred_df contains missing values in {date_col}/{stock_col}")

    if returns_df[[date_col, stock_col, return_col]].isna().any().any():
        raise ValueError(
            f"returns_df contains missing values in {date_col}/{stock_col}/{return_col}"
        )

    return_values = returns_df[return_col].to_numpy(dtype=float)
    if not np.isfinite(return_values).all():
        raise ValueError(f"{return_col} contains NaN or Inf values")

    if pred_df.duplicated(subset=[date_col, stock_col]).any():
        raise ValueError(f"pred_df contains duplicated {date_col}-{stock_col} rows")

    if returns_df.duplicated(subset=[date_col, stock_col]).any():
        raise ValueError(
            f"returns_df contains duplicated {date_col}-{stock_col} rows"
        )

    pred_dates = pd.DatetimeIndex(pred_df[date_col].unique()).sort_values()
    return_dates = pd.DatetimeIndex(returns_df[date_col].unique()).sort_values()

    if pred_dates.empty or return_dates.empty:
        raise ValueError("pred_df and returns_df must contain valid dates")

    if pred_dates.min() >= return_dates.max():
        raise ValueError(
            "No valid next-period return date exists. "
            "At least one prediction date must be earlier than the last return date."
        )
    
def _get_next_return_date(
    signal_date: pd.Timestamp,
    return_dates: pd.DatetimeIndex,
) -> pd.Timestamp | None:
    """
    Find the next available return date strictly after the signal date.
    """
    position = return_dates.searchsorted(signal_date, side="right")

    if position >= len(return_dates):
        return None

    return pd.Timestamp(return_dates[position])

def _compute_period_return(
    weights: pd.Series,
    next_returns: pd.Series,
    signal_date: pd.Timestamp,
    return_date: pd.Timestamp,
) -> float:
    """
    Compute one-period portfolio return using selected weights and next returns.
    """
    if weights.empty:
        raise ValueError("weights must not be empty")

    missing_stocks = weights.index.difference(next_returns.index)

    if len(missing_stocks) > 0:
        raise ValueError(
            "Missing next-period returns for selected stocks. "
            f"signal_date={signal_date}, return_date={return_date}, "
            f"missing_stocks={list(missing_stocks[:10])}"
        )

    aligned_returns = next_returns.reindex(weights.index).astype(float)

    if aligned_returns.isna().any():
        raise ValueError(
            f"Aligned returns contain NaN values for return_date={return_date}"
        )

    if not np.isfinite(aligned_returns.to_numpy()).all():
        raise ValueError(
            f"Aligned returns contain NaN or Inf values for return_date={return_date}"
        )

    return float((weights * aligned_returns).sum())

def run_backtest(
    pred_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    portfolio_config: PortfolioConfig,
    return_col: str = "return_1d",
    date_col: str = "date",
    stock_col: str = "stock_id",
    periods_per_year: int = 252,
    cost_config: TransactionCostConfig | None = None,
) -> dict[str, Any]:
    """
    Run a simple next-period portfolio backtest.

    Assumption:
    - Predictions on signal_date are used to build a portfolio at that date.
    - The portfolio earns returns on the next available return_date.
    - If cost_config is provided, transaction costs are subtracted from daily returns
      and reflected in NAV, summary, and Sharpe ratio.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    pred_work = pred_df.copy()
    returns_work = returns_df.copy()

    pred_work[date_col] = pd.to_datetime(pred_work[date_col])
    returns_work[date_col] = pd.to_datetime(returns_work[date_col])

    pred_work[stock_col] = pred_work[stock_col].astype(str)
    returns_work[stock_col] = returns_work[stock_col].astype(str)

    _validate_backtest_inputs(
        pred_df=pred_work,
        returns_df=returns_work,
        portfolio_config=portfolio_config,
        return_col=return_col,
        date_col=date_col,
        stock_col=stock_col,
    )

    return_dates = pd.DatetimeIndex(returns_work[date_col].unique()).sort_values()

    return_lookup = {
        pd.Timestamp(date): group.set_index(stock_col)[return_col].astype(float)
        for date, group in returns_work.groupby(date_col, sort=False)
    }

    period_returns: dict[pd.Timestamp, float] = {}
    weights_records: list[pd.Series] = []
    weight_dates: list[pd.Timestamp] = []
    turnover_values: dict[pd.Timestamp, float] = {}
    signal_to_return: dict[pd.Timestamp, pd.Timestamp] = {}

    previous_weights: pd.Series | None = None

    for signal_date, daily_pred in pred_work.groupby(date_col, sort=True):
        signal_date = pd.Timestamp(signal_date)

        return_date = _get_next_return_date(
            signal_date=signal_date,
            return_dates=return_dates,
        )

        if return_date is None:
            continue

        if return_date in period_returns:
            raise ValueError(
                f"Multiple signal dates map to the same return_date={return_date}. "
                "Please check prediction dates."
            )

        weights = build_portfolio_weights(
            pred_df=daily_pred,
            config=portfolio_config,
            previous_weights=previous_weights,
        )

        next_returns = return_lookup[return_date]

        period_return = _compute_period_return(
            weights=weights,
            next_returns=next_returns,
            signal_date=pd.Timestamp(signal_date),
            return_date=return_date,
        )

        if previous_weights is None:
            turnover = 0.0
        else:
            turnover = calculate_turnover(
                current_weights=weights,
                previous_weights=previous_weights,
            )

        period_returns[return_date] = period_return
        turnover_values[signal_date] = turnover
        signal_to_return[signal_date] = return_date

        weights.name = signal_date
        weights_records.append(weights)
        weight_dates.append(signal_date)

        previous_weights = weights.copy()

    if not period_returns:
        raise ValueError("No valid backtest periods were generated")

    daily_returns_raw = pd.Series(period_returns).sort_index()
    daily_returns_raw.name = "strategy_return"

    daily_turnover = pd.Series(turnover_values).sort_index()
    daily_turnover.name = "turnover"

    if cost_config is not None:
        avg_cost = 0.5 * (cost_config.buy_cost + cost_config.sell_cost)
        turnover_by_return = pd.Series({
            signal_to_return[sd]: tv
            for sd, tv in turnover_values.items()
        }).sort_index()
        daily_returns = daily_returns_raw - turnover_by_return * avg_cost
        daily_returns.name = "strategy_return_net"
        result = {
            "daily_returns": daily_returns,
            "daily_returns_raw": daily_returns_raw,
        }
    else:
        daily_returns = daily_returns_raw
        result = {
            "daily_returns": daily_returns,
        }

    daily_nav = cumulative_nav(daily_returns)
    daily_nav.name = "nav"

    daily_weights = pd.DataFrame(weights_records, index=weight_dates).sort_index()
    daily_weights.index.name = date_col

    summary = summarize_backtest(
        returns=daily_returns,
        periods_per_year=periods_per_year,
    )

    summary["mean_turnover"] = float(daily_turnover.mean())
    summary["total_turnover"] = float(daily_turnover.sum())

    result["summary"] = summary
    result["daily_nav"] = daily_nav
    result["daily_weights"] = daily_weights
    result["daily_turnover"] = daily_turnover

    return result

if __name__ == "__main__":
    sample_pred = pd.DataFrame(
        {
            "date": [
                "2020-01-01",
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
                "2020-01-02",
                "2020-01-02",
            ],
            "stock_id": ["A", "B", "C", "A", "B", "C"],
            "y_pred": [0.03, 0.01, 0.02, 0.00, 0.04, 0.02],
        }
    )

    sample_returns = pd.DataFrame(
        {
            "date": [
                "2020-01-02",
                "2020-01-02",
                "2020-01-02",
                "2020-01-03",
                "2020-01-03",
                "2020-01-03",
            ],
            "stock_id": ["A", "B", "C", "A", "B", "C"],
            "return_1d": [0.01, -0.02, 0.03, -0.01, 0.04, 0.02],
        }
    )

    config = PortfolioConfig(
        strategy="top_n",
        top_n=2,
        pred_col="y_pred",
        stock_col="stock_id",
    )

    result = run_backtest(
        pred_df=sample_pred,
        returns_df=sample_returns,
        portfolio_config=config,
        return_col="return_1d",
        date_col="date",
        stock_col="stock_id",
    )

    print("Summary:")
    print(result["summary"])

    print("\nDaily returns:")
    print(result["daily_returns"])

    print("\nDaily NAV:")
    print(result["daily_nav"])

    print("\nDaily weights:")
    print(result["daily_weights"])

    print("\nDaily turnover:")
    print(result["daily_turnover"])