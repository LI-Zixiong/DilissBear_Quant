"""
Experiment evaluation utilities.

This module does not implement new metrics.
It adapts existing metrics from src.backtest.metrics and run_backtest into
experiment-level valid/test outputs.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.backtest.engine import TransactionCostConfig, run_backtest
from src.backtest.metrics import prediction_ic_summary, top_bottom_spread
from src.backtest.portfolio import PortfolioConfig
from src.experiment.config import ExperimentConfig
from src.experiment.returns import align_predictions_to_returns


def evaluate_prediction_split(
    pred_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    config: ExperimentConfig,
    portfolio_config: PortfolioConfig,
    cost_config: TransactionCostConfig | None,
    split_name: str,
    pred_col: str = "y_pred",
    y_true_col: str = "y_true",
    min_obs: int = 50,
    top_frac: float = 0.1,
) -> dict[str, Any]:
    """
    Evaluate one prediction split, such as valid or test.

    This function reuses:
        - prediction_ic_summary()
        - top_bottom_spread()
        - run_backtest()

    Returns
    -------
    dict
        {
            "metrics": prefixed metric dictionary,
            "backtest_result": raw backtest result,
            "pred_df_bt": prediction rows filtered to valid future returns
        }
    """

    pred_df_bt = align_predictions_to_returns(
        pred_df=pred_df,
        returns_df=returns_df,
        date_col=config.date_col,
        stock_col=config.stock_col,
        pred_col=pred_col,
        return_col=config.return_col,
    )

    metrics: dict[str, Any] = {}

    # 1. IC / Rank IC / ICIR
    try:
        ic_metrics = prediction_ic_summary(
            pred_df=pred_df,
            date_col=config.date_col,
            y_true_col=y_true_col,
            y_pred_col=pred_col,
            min_obs=min_obs,
        )
    except ValueError as exc:
        ic_metrics = {
            "ic_error": repr(exc),
            "ic_n_periods": 0,
            "ic_mean": 0.0,
            "ic_std": 0.0,
            "ic_ir": 0.0,
            "ic_positive_rate": 0.0,
            "rank_ic_n_periods": 0,
            "rank_ic_mean": 0.0,
            "rank_ic_std": 0.0,
            "rank_ic_ir": 0.0,
            "rank_ic_positive_rate": 0.0,
        }

    metrics.update(_prefix_keys(ic_metrics, split_name))

    # 2. Top-bottom spread
    spread_metrics = top_bottom_spread(
        pred_df=pred_df,
        date_col=config.date_col,
        y_true_col=y_true_col,
        y_pred_col=pred_col,
        top_frac=top_frac,
        min_obs=min_obs,
    )
    metrics.update(_prefix_keys(spread_metrics, split_name))

    # 3. Portfolio backtest
    backtest_result = run_backtest(
        pred_df=pred_df_bt,
        returns_df=returns_df,
        portfolio_config=portfolio_config,
        return_col=config.return_col,
        date_col=config.date_col,
        stock_col=config.stock_col,
        periods_per_year=config.periods_per_year,
        cost_config=cost_config,
    )

    metrics.update(_prefix_keys(backtest_result["summary"], split_name))

    return {
        "metrics": metrics,
        "backtest_result": backtest_result,
        "pred_df_bt": pred_df_bt,
    }


def _prefix_keys(values: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Prefix metric keys with split name."""

    return {f"{prefix}_{key}": value for key, value in values.items()}