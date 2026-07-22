"""
Backtest performance metrics.
"""

from typing import Any

import pandas as pd
import numpy as np

ReturnInput = pd.Series | np.ndarray | list[float]
NavInput = pd.Series | np.ndarray | list[float]

def _to_numeric_series(values: ReturnInput, name: str) -> pd.Series:
    """
    Convert numeric input to a validated pandas Series.
    """
    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(values)

    series = series.astype(float)

    if series.empty:
        raise ValueError(f"{name} must not be empty")

    if not np.isfinite(series.to_numpy()).all():
        raise ValueError(f"{name} contains NaN or Inf values")

    return series

def _to_return_series(returns: ReturnInput) -> pd.Series:
    """
    Convert periodic returns to a pandas Series.
    """
    series = _to_numeric_series(returns, "returns")

    if (series <= -1.0).any():
        raise ValueError("returns contain values <= -100%, which are invalid")
    
    return series

def cumulative_nav(returns: ReturnInput, initial_nav: float = 1.0) -> pd.Series:
    """
    Convert periodic returns to cumulative net asset value (NAV) series.
    """
    if initial_nav <= 0:
        raise ValueError("Initial NAV must be positive.")
    
    return_series = _to_return_series(returns)

    nav = initial_nav * (1 + return_series).cumprod()
    nav.name = "nav"

    return nav

def total_return(returns: ReturnInput) -> float:
    """
    Calculate total return from periodic returns.
    """
    return_series = _to_return_series(returns)

    final_growth = float((1.0 + return_series).prod() )

    return final_growth - 1.0

def annualized_return(returns: ReturnInput, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return from periodic returns.
    """
    if periods_per_year <= 0:
        raise ValueError("Periods per year must be positive.")
    
    return_series = _to_return_series(returns)
    n_periods = len(return_series)

    if n_periods == 0:
        raise ValueError("Returns series is empty.")
    
    final_growth = float((1.0 + return_series).prod() )

    return float(final_growth ** (periods_per_year / n_periods) - 1.0)

def annualized_volatility(returns: ReturnInput, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility from periodic returns.
    """
    if periods_per_year <= 0:
        raise ValueError("Periods per year must be positive.")
    
    return_series = _to_return_series(returns)

    if len(return_series) < 2:
        return 0.0

    return float(return_series.std(ddof=1) * np.sqrt(periods_per_year))

def sharpe_ratio(returns: ReturnInput, periods_per_year: int = 252, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe ratio from periodic returns.
    """
    if periods_per_year <= 0:
        raise ValueError("Periods per year must be positive.")

    return_series = _to_return_series(returns)

    periodic_rf = risk_free_rate / periods_per_year
    excess_returns = return_series - periodic_rf

    volatility = excess_returns.std(ddof=1)

    if len(excess_returns) < 2 or volatility == 0.0 or np.isnan(volatility):
        return 0.0
    
    return float(excess_returns.mean() / volatility * np.sqrt(periods_per_year))

def _to_nav_series(nav: NavInput) -> pd.Series:
    """
    Convert NAV values to a pandas Series.
    """
    series = _to_numeric_series(nav, "nav")

    if (series <= 0.0).any():
        raise ValueError("nav series contains non-positive values")
    
    return series

def max_drawdown(nav: NavInput, initial_nav: float = 1.0) -> float:
    """
    Calculate the maximum drawdown from a NAV series.
    """
    nav_series = _to_nav_series(nav)

    if not np.isfinite(initial_nav) or initial_nav <= 0.0:
        raise ValueError("initial_nav must be finite and positive")
    # Include capital immediately before the first reported return.
    running_max = nav_series.cummax().clip(lower=float(initial_nav))
    drawdowns = nav_series / running_max - 1.0

    return float(-drawdowns.min())

def hit_rate(returns: ReturnInput, threshold: float = 0.0) -> float:
    """
    Calculate the fraction of periods whose return is above a threshold.
    """
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    return_series = _to_return_series(returns)

    return float((return_series > threshold).mean())

def ic_summary(ic_values: ReturnInput) -> dict[str, Any]:
    """
    Summarize an IC series with common descriptive statistics.
    """
    ic_series = _to_numeric_series(ic_values, "ic_values")

    if ((ic_series < -1.0) | (ic_series > 1.0)).any():
        raise ValueError("ic_values must be within [-1, 1]")

    ic_std = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else 0.0
    ic_mean = float(ic_series.mean())

    return {
        "n_periods": int(len(ic_series)),
        "mean_ic": ic_mean,
        "std_ic": ic_std,
        "ic_ir": float(ic_mean / ic_std) if ic_std > 0.0 else 0.0,
        "positive_ic_rate": float((ic_series > 0.0).mean()),
        "min_ic": float(ic_series.min()),
        "max_ic": float(ic_series.max()),
    }

def calculate_turnover(current_weights: pd.Series, previous_weights: pd.Series | None = None, half_turnover: bool = True,) -> float:
    """
    Calculate portfolio turnover from current and previous weights.
    """
    if previous_weights is None:
        previous_weights = pd.Series(dtype=float)

    current_weights = current_weights.astype(float)
    previous_weights = previous_weights.astype(float)

    all_assets = current_weights.index.union(previous_weights.index)

    current = current_weights.reindex(all_assets).fillna(0.0)
    previous = previous_weights.reindex(all_assets).fillna(0.0)

    turnover = float((current - previous).abs().sum())

    if half_turnover:
        turnover *= 0.5
    
    return turnover

def summarize_backtest(returns: ReturnInput, periods_per_year: int = 252, risk_free_rate: float = 0.0) -> dict[str, Any]:
    """
    Summarize backtest performance metrics in a dictionary.
    """
    return_series = _to_return_series(returns)
    nav = cumulative_nav(return_series)

    summary = {
        "n_periods": int(len(return_series)),
        "total_return": float(total_return(return_series)),
        "annualized_return": float(annualized_return(return_series, periods_per_year)),
        "annualized_volatility": float(annualized_volatility(return_series, periods_per_year)),
        "sharpe_ratio": float(sharpe_ratio(return_series, periods_per_year, risk_free_rate)),
        "max_drawdown": float(max_drawdown(nav)),
        "hit_rate": float(hit_rate(return_series)),
        "final_nav": float(nav.iloc[-1]),
    }

    return summary


def calculate_ic_series(
    pred_df: pd.DataFrame,
    date_col: str = "date",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    method: str = "pearson",
    min_obs: int = 10,
) -> pd.Series:
    """
    Calculate daily IC or Rank IC from prediction DataFrame.

    method="pearson" gives normal IC.
    method="spearman" gives Rank IC.
    """
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be one of: pearson, spearman")

    required_cols = [date_col, y_true_col, y_pred_col]
    missing_cols = [col for col in required_cols if col not in pred_df.columns]
    if missing_cols:
        raise ValueError(f"pred_df missing required columns: {missing_cols}")

    work = pred_df[required_cols].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="raise")

    y_true = work[y_true_col].to_numpy(dtype=float)
    y_pred = work[y_pred_col].to_numpy(dtype=float)

    if not np.isfinite(y_true).all():
        raise ValueError(f"{y_true_col} contains NaN or Inf values")

    if not np.isfinite(y_pred).all():
        raise ValueError(f"{y_pred_col} contains NaN or Inf values")

    ic_values = {}

    for date, group in work.groupby(date_col, sort=True):
        if len(group) < min_obs:
            continue

        if group[y_true_col].nunique() <= 1 or group[y_pred_col].nunique() <= 1:
            continue

        ic = group[y_true_col].corr(group[y_pred_col], method=method)

        if pd.notna(ic) and np.isfinite(ic):
            ic_values[pd.Timestamp(date)] = float(ic)

    if not ic_values:
        raise ValueError("No valid IC values were generated")

    name = "rank_ic" if method == "spearman" else "ic"
    return pd.Series(ic_values, name=name).sort_index()


def prediction_ic_summary(
    pred_df: pd.DataFrame,
    date_col: str = "date",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    min_obs: int = 10,
) -> dict[str, float | int]:
    """
    Summarize daily IC and Rank IC.
    """
    ic = calculate_ic_series(
        pred_df=pred_df,
        date_col=date_col,
        y_true_col=y_true_col,
        y_pred_col=y_pred_col,
        method="pearson",
        min_obs=min_obs,
    )

    rank_ic = calculate_ic_series(
        pred_df=pred_df,
        date_col=date_col,
        y_true_col=y_true_col,
        y_pred_col=y_pred_col,
        method="spearman",
        min_obs=min_obs,
    )

    ic_stats = ic_summary(ic)
    rank_ic_stats = ic_summary(rank_ic)

    return {
        "ic_n_periods": ic_stats["n_periods"],
        "ic_mean": ic_stats["mean_ic"],
        "ic_std": ic_stats["std_ic"],
        "ic_ir": ic_stats["ic_ir"],
        "ic_positive_rate": ic_stats["positive_ic_rate"],
        "rank_ic_n_periods": rank_ic_stats["n_periods"],
        "rank_ic_mean": rank_ic_stats["mean_ic"],
        "rank_ic_std": rank_ic_stats["std_ic"],
        "rank_ic_ir": rank_ic_stats["ic_ir"],
        "rank_ic_positive_rate": rank_ic_stats["positive_ic_rate"],
    }

def top_bottom_spread(
    pred_df: pd.DataFrame,
    date_col: str = "date",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    top_frac: float = 0.1,
    min_obs: int = 10,
) -> dict[str, float]:
    """
    Per-date top vs bottom decile realized return spread.

    For each date, selects the top and bottom `top_frac` stocks by
    prediction, computes the mean realized return difference, and
    reports the time-series mean and annualized Sharpe of that spread.
    """
    if not 0.0 < top_frac < 0.5:
        raise ValueError(f"top_frac must be in (0, 0.5), got {top_frac}")

    spreads: dict[pd.Timestamp, float] = {}

    for date, group in pred_df.groupby(date_col, sort=True):
        if len(group) < min_obs:
            continue

        n_select = max(int(len(group) * top_frac), 1)
        top = group.nlargest(n_select, y_pred_col)
        bot = group.nsmallest(n_select, y_pred_col)

        spread = float(top[y_true_col].mean() - bot[y_true_col].mean())

        if np.isfinite(spread):
            spreads[pd.Timestamp(date)] = spread

    if not spreads:
        return {"spread_mean": 0.0, "spread_sharpe": 0.0, "spread_n_periods": 0}

    series = pd.Series(spreads).sort_index()
    mean_val = float(series.mean())
    std_val = float(series.std(ddof=1))

    return {
        "spread_mean": mean_val,
        "spread_sharpe": float(mean_val / std_val * np.sqrt(252)) if std_val > 0 else 0.0,
        "spread_n_periods": len(series),
    }


if __name__ == "__main__":
    sample_returns = pd.Series(
        [0.01, -0.005, 0.02, -0.01, 0.015],
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
        name="strategy_return",
    )

    sample_nav = cumulative_nav(sample_returns)
    summary = summarize_backtest(sample_returns)

    print("NAV:")
    print(sample_nav)

    print("\nSummary:")
    print(summary)

    sample_ic = pd.Series([0.12, -0.05, 0.08, 0.0, -0.02], name="ic")

    print("\nIC Summary:")
    print(ic_summary(sample_ic))

    previous_weights = pd.Series({"A": 0.5, "B": 0.5})
    current_weights = pd.Series({"A": 0.2, "B": 0.3, "C": 0.5})

    print("\nTurnover:")
    print(calculate_turnover(current_weights, previous_weights))
