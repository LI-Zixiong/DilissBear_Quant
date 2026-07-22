import numpy as np
import pandas as pd
import unittest

from src.backtest.engine import (
    TransactionCostConfig,
    run_backtest,
)
from src.backtest.ensemble_methods import _smooth_with_history
from src.backtest.metrics import cumulative_nav, max_drawdown
from src.backtest.portfolio import PortfolioConfig
from src.experiment.returns import align_predictions_to_returns


def _top_n(n: int = 1) -> PortfolioConfig:
    return PortfolioConfig(
        strategy="top_n", top_n=n, pred_col="y_pred", stock_col="stock_id",
    )


class BacktestCorrectionTests(unittest.TestCase):

 def test_future_return_availability_does_not_change_candidate_set(self):
    signal = pd.Timestamp("2026-01-05")
    realized = pd.Timestamp("2026-01-06")
    pred = pd.DataFrame({
        "time": [signal, signal],
        "stock_id": ["A", "B"],
        "y_pred": [2.0, 1.0],
    })
    returns = pd.DataFrame({
        "time": [realized],
        "stock_id": ["B"],
        "return_1d": [0.10],
    })

    aligned = align_predictions_to_returns(
        pred, returns, "time", "stock_id",
    )
    self.assertEqual(set(aligned["stock_id"]), {"A", "B"})

    # A remains the true T-day Top-1. Paper mode must fail loudly instead of
    # secretly replacing it with B using knowledge of T+1 availability.
    with self.assertRaisesRegex(ValueError, "Missing next-period returns"):
        run_backtest(
            aligned, returns, _top_n(1), return_col="return_1d",
            date_col="time", stock_col="stock_id",
        )


 def test_turnover_and_cost_use_post_return_drift_weights(self):
    dates = pd.date_range("2026-01-05", periods=3, freq="B")
    pred = pd.DataFrame({
        "time": [dates[0], dates[0], dates[1], dates[1]],
        "stock_id": ["A", "B", "A", "B"],
        "y_pred": [1.0, 1.0, 1.0, 1.0],
    })
    returns = pd.DataFrame({
        "time": [dates[1], dates[1], dates[2], dates[2]],
        "stock_id": ["A", "B", "A", "B"],
        "return_1d": [0.10, -0.10, 0.0, 0.0],
    })
    result = run_backtest(
        pred, returns, _top_n(2), return_col="return_1d",
        date_col="time", stock_col="stock_id",
        cost_config=TransactionCostConfig(buy_cost=0.0003, sell_cost=0.0008),
    )

    # Day 1 ends at 55% A / 45% B. Restoring 50/50 buys 5% B and sells 5% A.
    np.testing.assert_allclose(result["daily_turnover"].to_numpy(), [1.0, 0.05])
    np.testing.assert_allclose(
        result["daily_costs"].to_numpy(),
        [0.0003, 0.05 * (0.0003 + 0.0008)],
    )
    np.testing.assert_allclose(
        result["daily_end_weights"].iloc[0][["A", "B"]], [0.55, 0.45],
    )


 def test_test_split_smoothing_uses_valid_tail_as_warmup(self):
    valid_dates = pd.date_range("2026-01-01", periods=4, freq="B")
    test_date = pd.Timestamp("2026-01-07")
    valid = pd.DataFrame({
        "time": valid_dates,
        "stock_id": ["A"] * 4,
        "m": [1.0, 2.0, 3.0, 4.0],
    })
    test = pd.DataFrame({
        "time": [test_date], "stock_id": ["A"], "m": [5.0],
    })
    smoothed = _smooth_with_history(test, ["m"], 5, valid)
    self.assertEqual(len(smoothed), 1)
    self.assertAlmostEqual(smoothed.iloc[0]["m"], 3.0)


 def test_max_drawdown_includes_initial_capital_peak(self):
    nav = cumulative_nav(pd.Series([-0.10, 0.05]))
    self.assertAlmostEqual(max_drawdown(nav), 0.10)


if __name__ == "__main__":
    unittest.main()
