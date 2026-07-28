from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _project_file(flat_name: str, project_relative: str) -> Path:
    flat = ROOT / flat_name
    return flat if flat.exists() else ROOT / project_relative


def _load_real_backtest():
    src = types.ModuleType("src")
    backtest = types.ModuleType("src.backtest")
    metrics = types.ModuleType("src.backtest.metrics")

    def summarize_backtest(returns, periods_per_year=252):
        values = pd.Series(returns).fillna(0.0)
        nav = pd.concat([pd.Series([1.0]), (1 + values).cumprod()], ignore_index=True)
        dd = nav / nav.cummax() - 1
        return {"sharpe_ratio": 0.0, "max_drawdown": float(-dd.min()),
                "annualized_return": float(values.mean() * periods_per_year),
                "hit_rate": float((values > 0).mean())}

    metrics.summarize_backtest = summarize_backtest
    sys.modules.update({"src": src, "src.backtest": backtest, "src.backtest.metrics": metrics})
    path = _project_file("real_backtest.py", "src/backtest/real_backtest.py")
    spec = importlib.util.spec_from_file_location("real_backtest_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RB = _load_real_backtest()


def _panels(limit_down_a_on_day2=False):
    dates = pd.to_datetime(["2026-07-20", "2026-07-21", "2026-07-22"])
    stocks = [f"{i:06d}" for i in range(1, 101)]
    rows_pred, rows_price = [], []
    for day_i, date in enumerate(dates):
        for i, stock in enumerate(stocks, start=1):
            score = float(i)
            if day_i >= 1 and stock == "000100":
                score = 0.0
            if day_i >= 1 and stock == "000050":
                score = 200.0
            close = 10.0 + i / 100.0 + day_i * 0.1
            pre = 10.0 + i / 100.0 + max(0, day_i - 1) * 0.1
            if limit_down_a_on_day2 and day_i == 1 and stock == "000100":
                close = round(pre * 0.9, 2)
            rows_pred.append({"time": date, "stock_id": stock, "ew_score": score})
            rows_price.append({"time": date, "stock_id": stock, "close": close,
                               "pre_close": pre, "up_limit": round(pre * 1.1, 2),
                               "down_limit": round(pre * 0.9, 2)})
    return pd.DataFrame(rows_pred), pd.DataFrame(rows_price)


def _config(**overrides):
    values = dict(capital=200_000, max_stocks=50, pred_col="ew_score",
                  position_sizing="budget", cash_ratio=0.80,
                  min_commission=0.0, buy_slippage_bps={1: 3, 2: 5, 3: 8},
                  sell_slippage_bps=3.0, start_date="2026-07-20")
    values.update(overrides)
    return RB.RealBacktestConfig(**values)


def test_top50_uses_80pct_budget_with_two_to_one_units_and_round_lots():
    rows = []
    for i in range(50):
        elite = i < 20
        rows.append({"stock_id": f"{i:06d}", "open": 10.0,
                     "score": float(50 - i), "rank_pct": 0.99 if elite else 0.97,
                     "bin": 3 if elite else 2})
    eligible = pd.DataFrame(rows)
    cfg = _config(buy_slippage_bps={1: 0, 2: 0, 3: 0})
    plans, spent = RB._allocate_by_budget(eligible, 200_000.0, cfg)
    elite = [p for p in plans if p["bin"] == 3]
    strong = [p for p in plans if p["bin"] == 2]
    assert len(elite) == 20 and len(strong) == 30
    assert {p["lots"] if "lots" in p else p["n_lots"] for p in elite} == {5}
    assert {p["lots"] if "lots" in p else p["n_lots"] for p in strong} == {2}
    assert np.isclose(spent, 160_000.0)
    assert spent <= 200_000.0 * 0.80


def test_rounding_overrun_removes_lowest_ranked_names_first():
    eligible = pd.DataFrame([
        {"stock_id": f"{i:06d}", "open": 53.0, "score": float(50 - i),
         "rank_pct": 0.99 if i < 20 else 0.97, "bin": 3 if i < 20 else 2}
        for i in range(50)
    ])
    cfg = _config(buy_slippage_bps={1: 0, 2: 0, 3: 0})
    plans, spent = RB._allocate_by_budget(eligible, 200_000.0, cfg)
    kept_scores = [p["score"] for p in plans]
    assert spent <= 160_000.0
    assert kept_scores == sorted(kept_scores, reverse=True)
    assert kept_scores == list(range(50, 20, -1))


def test_top50_is_frozen_before_price_availability_without_backfill():
    date = pd.Timestamp("2026-07-22")
    pred = pd.DataFrame([
        {"time": date, "stock_id": f"{i:06d}", "ew_score": float(i)}
        for i in range(1, 1501)
    ])
    prices = pd.DataFrame([
        {"time": date, "stock_id": f"{i:06d}", "close": 10.0, "pre_close": 10.0,
         "up_limit": 11.0, "down_limit": 9.0}
        for i in range(1, 1500)  # highest-ranked 001500 deliberately has no price
    ])
    result = RB.run_real_backtest(pred, prices, _config())
    bought = set(result["trade_log"].loc[result["trade_log"]["action"] == "BUY", "stock_id"])
    assert "001500" not in bought
    assert "001450" in bought  # slot fills from pool, rank 51 enters


def test_latest_signal_is_bought_without_known_t_plus_one():
    pred, prices = _panels()
    latest = pd.Timestamp("2026-07-22")
    result = RB.run_real_backtest(pred[pred["time"] == latest], prices, _config())
    snapshots = result["position_snapshots"]
    assert not snapshots.empty
    assert snapshots["last_buy_date"].max() == latest
    assert snapshots["target_exit_date"].isna().all()


def test_limit_down_exit_is_carried_then_sold():
    pred, prices = _panels(limit_down_a_on_day2=True)
    result = RB.run_real_backtest(pred, prices, _config())
    day2 = result["position_snapshots"]
    day2 = day2[day2["date"] == pd.Timestamp("2026-07-21")]
    assert "000100" in set(day2["stock_id"])
    assert int(day2.loc[day2["stock_id"] == "000100", "delayed_exit_days"].iloc[0]) == 1
    sold = result["trade_log"]
    sold = sold[(sold["date"] == pd.Timestamp("2026-07-22")) &
                (sold["stock_id"] == "000100") & (sold["action"] == "SELL")]
    assert not sold.empty


def test_continuous_selection_trades_only_lot_delta():
    pred, prices = _panels()
    result = RB.run_real_backtest(pred, prices, _config())
    # 000099 stays selected but moves from elite to strong after another name
    # jumps above it.  Only the one-lot target reduction may be sold.
    trades = result["trade_log"]
    stock = trades[trades["stock_id"] == "000099"]
    assert len(stock[stock["action"] == "BUY"]) == 1
    initial_shares = int(stock.loc[stock["action"] == "BUY", "shares"].iloc[0])
    reductions = stock.loc[stock["action"] == "SELL", "shares"]
    assert not reductions.empty
    assert int(reductions.max()) < initial_shares
    assert stock["position_id"].nunique() == 1


def test_all_fills_are_board_lots_and_cash_never_negative():
    pred, prices = _panels()
    result = RB.run_real_backtest(pred, prices, _config())
    trades = result["trade_log"]
    assert (trades["shares"] % 100 == 0).all()
    assert (result["daily_cash"] >= -1e-8).all()


def test_public_nav_has_exact_inception_anchor():
    path = _project_file("convert_data.py", "website/convert_data.py")
    spec = importlib.util.spec_from_file_location("convert_data_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    account = pd.DataFrame({
        "date": pd.to_datetime(["2026-07-20", "2026-07-21"]),
        "equity": [199_900.0, 201_000.0], "cash": [10_000.0, 11_000.0],
        "market_value": [189_900.0, 190_000.0], "n_positions": [2, 2],
    })
    public = module._published_nav(account, pd.Timestamp("2026-07-20"), 200_000.0)
    assert public.iloc[0]["published_nav"] == 1.0
    assert np.isclose(public.iloc[1]["published_nav"], 1.005)
