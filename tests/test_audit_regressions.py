import numpy as np
import pandas as pd

from src.backtest.bayes_blender import apply_online_gate
from src.backtest.engine import TransactionCostConfig, run_backtest
from src.backtest.portfolio import PortfolioConfig
from src.data.dataset_builder import PanelDatasetBuilder
from src.experiment.split import split_panel_by_date_ratio


def test_ratio_split_purges_forward_label_boundaries():
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    panel = pd.DataFrame({
        "time": np.repeat(dates, 2),
        "stock_id": ["000001", "000002"] * len(dates),
    })
    train, valid, test, _, _ = split_panel_by_date_ratio(
        panel, "time", "stock_id", (0.6, 0.2, 0.2), purge=6,
    )
    assert train["time"].nunique() == 18
    assert valid["time"].nunique() == 2
    assert test["time"].nunique() == 8


def test_dataset_builder_can_materialize_unlabelled_live_endpoint():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    panel = pd.DataFrame({
        "time": dates,
        "stock_id": ["000001"] * 3,
        "factor": [1.0, 2.0, 3.0],
        "target": [0.1, 0.2, np.nan],
    })
    builder = PanelDatasetBuilder(
        ["factor"], "target", "time", "stock_id", seq_len=2,
    )
    tabular = builder.build_tabular_dataset(panel.tail(1), require_target=False)
    sequence = builder.build_sequence_dataset(panel, require_target=False)
    assert np.isnan(tabular.y[0])
    assert np.isnan(sequence.y[-1])
    assert sequence.meta.iloc[-1]["time"] == dates[-1]


def test_one_way_turnover_costs_charge_initial_buy_and_round_trip_rebalance():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    pred = pd.DataFrame({
        "time": [dates[0], dates[0], dates[1], dates[1]],
        "stock_id": ["000001", "000002", "000001", "000002"],
        "y_pred": [2.0, 1.0, 1.0, 2.0],
    })
    returns = pd.DataFrame({
        "time": [dates[1], dates[1], dates[2], dates[2]],
        "stock_id": ["000001", "000002", "000001", "000002"],
        "return_1d": [0.0, 0.0, 0.0, 0.0],
    })
    result = run_backtest(
        pred, returns, PortfolioConfig(top_n=1),
        return_col="return_1d", date_col="time",
        cost_config=TransactionCostConfig(buy_cost=0.0003, sell_cost=0.0008),
    )
    np.testing.assert_allclose(result["daily_costs"], [0.0003, 0.0011])


def _gate_fixture(last_return: float, gate_burnin: int = 1):
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    rows = []
    for date in dates:
        rows.extend([
            {"time": date, "stock_id": "000001", "bayes_score": 0.0,
             "m1_rank": 1.0, "m2_rank": 0.0,
             "m1_llr_blended": 1.0, "m2_llr_blended": 0.0},
            {"time": date, "stock_id": "000002", "bayes_score": 0.0,
             "m1_rank": 0.0, "m2_rank": 1.0,
             "m1_llr_blended": 0.0, "m2_llr_blended": 1.0},
        ])
    returns = pd.DataFrame([
        {"time": dates[1], "stock_id": "000001", "return_1d": 0.1},
        {"time": dates[1], "stock_id": "000002", "return_1d": -0.1},
        {"time": dates[2], "stock_id": "000001", "return_1d": last_return},
        {"time": dates[2], "stock_id": "000002", "return_1d": -last_return},
    ])
    return apply_online_gate(
        pd.DataFrame(rows), returns, models=["m1", "m2"], top_n=1,
        gate_burnin=gate_burnin, reward_window=1,
    )[1]


def test_online_gate_uses_only_already_realized_return_and_honours_burnin():
    history_a = _gate_fixture(-0.9)
    history_b = _gate_fixture(0.9)
    assert history_a.loc[1, "m1_weight"] > 0.5
    assert np.isclose(
        history_a.loc[1, "m1_weight"], history_b.loc[1, "m1_weight"]
    )
    burnin = _gate_fixture(0.1, gate_burnin=2)
    assert np.isclose(burnin.loc[1, "m1_weight"], 0.5)
    assert burnin.loc[2, "m1_weight"] > 0.5
