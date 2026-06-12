"""
Portfolio construction and management utilities.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass
class PortfolioConfig:
    """
    Configuration for portfolio construction.

    strategy : str
        Portfolio construction strategy. Supported values:
        - "equal_all": equal weight over all available stocks.
        - "top_n": select top N stocks by prediction and equal weight them.
    top_n : int
        Number of stocks selected for the top_n strategy.
    pred_col : str
        Prediction column used for ranking.
    stock_col : str
        Stock identifier column.
    """
    strategy: str = "top_n"
    top_n: int = 50
    buffer_n: int = 80
    pred_col: str = "y_pred"
    stock_col: str = "stock_id"
    # Bin thresholds for bin_weighted strategies (matches Bayes V2 rank_bins)
    bin_edges: tuple = (0.94, 0.96, 0.985)
    bin_weights: tuple = (0.5, 1.0, 2.0)  # bin1, bin2, bin3

    def __post_init__(self) -> None:
        if self.strategy not in {"equal_all", "top_n", "score_weighted",
                                  "top_n_buffer", "bin_weighted", "bin_weighted_buffer"}:
            raise ValueError(f"unsupported strategy: {self.strategy}")
        if self.top_n <= 0:
            raise ValueError("top_n must be a positive integer")
        if self.strategy in ("top_n_buffer", "bin_weighted_buffer") and self.buffer_n <= self.top_n:
            raise ValueError(f"buffer_n ({self.buffer_n}) must be > top_n ({self.top_n})")
        if not self.pred_col:
            raise ValueError("pred_col must be a non-empty string")
        if not self.stock_col:
            raise ValueError("stock_col must be a non-empty string")
        
def _validate_prediction_frame(pred_df: pd.DataFrame, config: PortfolioConfig) -> None:
    """
    Validate the prediction DataFrame against the portfolio configuration.
    """
    if not isinstance(pred_df, pd.DataFrame):
        raise TypeError(f"pred_df must be a DataFrame, got {type(pred_df).__name__}")

    if pred_df.empty:
        raise ValueError("pred_df must not be empty")

    if config.stock_col not in pred_df.columns:
        raise ValueError(f"pred_df missing stock column: {config.stock_col}")

    if pred_df[config.stock_col].isna().any():
        raise ValueError(f"{config.stock_col} contains missing values")

    if pred_df[config.stock_col].duplicated().any():
        raise ValueError(f"{config.stock_col} contains duplicated stocks")

    if config.strategy == "top_n":
        if config.pred_col not in pred_df.columns:
            raise ValueError(f"pred_df missing prediction column: {config.pred_col}")

        pred_values = pred_df[config.pred_col].to_numpy(dtype=float)

        if not np.isfinite(pred_values).all():
            raise ValueError(f"{config.pred_col} contains NaN or Inf values")
        
def _equal_weight(stock_ids: pd.Series) -> pd.Series:
    """
    Assign equal weights to the given stock IDs.
    """
    n_stocks = len(stock_ids)

    if n_stocks == 0:
        raise ValueError("no stocks to assign weights to")
    
    weight = 1.0 / n_stocks

    return pd.Series(
        data=weight,
        index=stock_ids,
        name="weight",
        dtype=float
    )

def _build_equal_all_weights(pred_df: pd.DataFrame, config: PortfolioConfig,) -> pd.Series:
    """
    Build equal weights over the full available stock universe.
    """
    stock_ids = pred_df[config.stock_col].reset_index(drop=True)

    return _equal_weight(stock_ids)

def _build_top_n_weights(pred_df: pd.DataFrame, config: PortfolioConfig) -> pd.Series:
    """
    Select top N stocks by prediction and assign equal weights to them.
    """
    sorted_df = pred_df.sort_values(
        by=[config.pred_col, config.stock_col],
        ascending=[False, True],
        kind="mergesort"
    )

    n_select = min(config.top_n, len(sorted_df))
    selected = sorted_df.head(n_select)

    stock_ids = selected[config.stock_col].reset_index(drop=True)

    return _equal_weight(stock_ids)

def _build_score_weighted_weights(pred_df: pd.DataFrame, config: PortfolioConfig) -> pd.Series:
    """Select top N by score, weight proportional to score."""
    sorted_df = pred_df.sort_values(
        by=[config.pred_col, config.stock_col],
        ascending=[False, True],
        kind="mergesort",
    )
    n_select = min(config.top_n, len(sorted_df))
    selected = sorted_df.head(n_select).copy()

    scores = selected[config.pred_col].to_numpy(dtype=float)
    # Softmax over the selected stocks — higher score → higher weight
    scores = np.exp(np.clip(scores - scores.max(), -50, 50))
    weights = scores / scores.sum()

    return pd.Series(
        data=weights,
        index=selected[config.stock_col].reset_index(drop=True),
        name="weight",
        dtype=float,
    )


def _build_buffer_weights(
    pred_df: pd.DataFrame, config: PortfolioConfig,
    previous_weights: pd.Series | None = None,
) -> pd.Series:
    """Select top_n from buffer_n, preserving existing holdings when possible.

    Stocks in previous top_n that remain in top buffer_n stay.
    New slots filled from buffer by score.
    """
    sorted_df = pred_df.sort_values(
        by=[config.pred_col, config.stock_col],
        ascending=[False, True],
        kind="mergesort",
    )
    buffer_df = sorted_df.head(min(config.buffer_n, len(sorted_df)))
    buffer_stocks = set(buffer_df[config.stock_col])

    holdings: set[str] = set()
    if previous_weights is not None:
        prev_stocks = set(str(s) for s in previous_weights.index)
        holdings = prev_stocks & buffer_stocks

    # Fill remaining slots from buffer (excluding existing holdings), by score
    remaining = int(config.top_n - len(holdings))
    if remaining > 0:
        new_stocks = buffer_df[~buffer_df[config.stock_col].isin(holdings)]
        new_stocks = new_stocks.head(remaining)
        holdings.update(new_stocks[config.stock_col])

    selected = sorted_df[sorted_df[config.stock_col].isin(holdings)]
    return _equal_weight(selected[config.stock_col].reset_index(drop=True))


def _bin_index(rank_pct: float, edges: tuple) -> int:
    """Map rank percentile to bin index (0, 1, 2, 3)."""
    for i, edge in enumerate(edges):
        if rank_pct < edge:
            return i
    return len(edges)


def _build_bin_weighted_weights(pred_df: pd.DataFrame, config: PortfolioConfig) -> pd.Series:
    """Top N by score, weight by Bayes bin (coarse, realisable)."""
    sorted_df = pred_df.sort_values(
        by=[config.pred_col, config.stock_col],
        ascending=[False, True], kind="mergesort",
    )
    n_select = min(config.top_n, len(sorted_df))
    selected = sorted_df.head(n_select).reset_index(drop=True).copy()

    full_ranks = sorted_df[config.pred_col].rank(pct=True)
    ranks = full_ranks.head(n_select)
    weights = np.ones(len(selected), dtype=float)
    for i in range(len(selected)):
        bi = _bin_index(float(ranks.iloc[i]), config.bin_edges)
        if bi == 0:
            weights[i] = 0.0
        elif bi == 1:
            weights[i] = config.bin_weights[0]
        elif bi == 2:
            weights[i] = config.bin_weights[1]
        else:
            weights[i] = config.bin_weights[2]

    weights = weights / weights.sum()
    return pd.Series(weights, index=selected[config.stock_col],
                     name="weight", dtype=float)


def _build_bin_weighted_buffer_weights(
    pred_df: pd.DataFrame, config: PortfolioConfig,
    previous_weights: pd.Series | None = None,
) -> pd.Series:
    """Buffer selection + bin-weighted allocation."""
    sorted_df = pred_df.sort_values(
        by=[config.pred_col, config.stock_col],
        ascending=[False, True], kind="mergesort",
    )
    buffer_df = sorted_df.head(min(config.buffer_n, len(sorted_df)))
    buffer_stocks = set(buffer_df[config.stock_col])

    holdings: set[str] = set()
    if previous_weights is not None:
        prev_stocks = set(str(s) for s in previous_weights.index)
        holdings = prev_stocks & buffer_stocks

    remaining = int(config.top_n - len(holdings))
    if remaining > 0:
        new_stocks = buffer_df[~buffer_df[config.stock_col].isin(holdings)]
        new_stocks = new_stocks.head(remaining)
        holdings.update(new_stocks[config.stock_col])

    selected = sorted_df[sorted_df[config.stock_col].isin(holdings)].copy()
    if selected.empty:
        return pd.Series([], name="weight", dtype=float)
    full_ranks = sorted_df[config.pred_col].rank(pct=True)
    weights = np.ones(len(selected), dtype=float)
    for i, idx in enumerate(selected.index):
        bi = _bin_index(float(full_ranks.loc[idx]), config.bin_edges)
        if bi == 0:
            weights[i] = 0.0
        elif bi == 1:
            weights[i] = config.bin_weights[0]
        elif bi == 2:
            weights[i] = config.bin_weights[1]
        else:
            weights[i] = config.bin_weights[2]
    weights = weights / weights.sum()
    return pd.Series(weights, index=selected[config.stock_col].reset_index(drop=True),
                     name="weight", dtype=float)


def build_portfolio_weights(
    pred_df: pd.DataFrame, config: PortfolioConfig | None = None,
    previous_weights: pd.Series | None = None,
) -> pd.Series:
    """
    Build portfolio weights from one-period prediction DataFrame.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Prediction DataFrame for a single date. It should contain at least
        stock_col, and pred_col when using the top_n strategy.
    config : PortfolioConfig, optional
        Portfolio construction configuration.

    Returns
    -------
    pd.Series
        Portfolio weights indexed by stock identifier.
    """
    if config is None:
        config = PortfolioConfig()

    _validate_prediction_frame(pred_df, config)

    if config.strategy == "equal_all":
        weights = _build_equal_all_weights(pred_df, config)
    elif config.strategy == "top_n":
        weights = _build_top_n_weights(pred_df, config)
    elif config.strategy == "score_weighted":
        weights = _build_score_weighted_weights(pred_df, config)
    elif config.strategy == "top_n_buffer":
        weights = _build_buffer_weights(pred_df, config, previous_weights)
    elif config.strategy == "bin_weighted":
        weights = _build_bin_weighted_weights(pred_df, config)
    elif config.strategy == "bin_weighted_buffer":
        weights = _build_bin_weighted_buffer_weights(pred_df, config, previous_weights)
    else:
        raise ValueError(f"unsupported strategy: {config.strategy}")

    weight_sum = float(weights.sum())

    if not np.isclose(weight_sum, 1.0):
        raise ValueError(f"weights must sum to 1.0, got {weight_sum}")

    if (weights < 0).any():
        raise ValueError("weights must be non-negative")

    return weights

if __name__ == "__main__":
    sample_pred = pd.DataFrame(
        {
            "date": ["2020-01-01"] * 5,
            "stock_id": ["A", "B", "C", "D", "E"],
            "y_pred": [0.03, -0.01, 0.02, 0.05, 0.00],
        }
    )

    equal_config = PortfolioConfig(strategy="equal_all")
    equal_weights = build_portfolio_weights(sample_pred, equal_config)

    print("Equal-all weights:")
    print(equal_weights)
    print("Sum:", equal_weights.sum())

    top_n_config = PortfolioConfig(strategy="top_n", top_n=3)
    top_n_weights = build_portfolio_weights(sample_pred, top_n_config)

    print("\nTop-N weights:")
    print(top_n_weights)
    print("Sum:", top_n_weights.sum())