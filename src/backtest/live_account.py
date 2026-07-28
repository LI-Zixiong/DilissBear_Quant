"""Build the production EW-rank account ledger from frozen model predictions.

Run after ``daily_update`` and ``live_predict``.  Deterministic replay of the
real account from a start date using only date-available information.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.ensemble_utils import normalize_keys, smooth_predictions
from src.backtest.real_backtest import RealBacktestConfig, run_real_backtest
from src.pipeline.live_store import LiveStore

MODELS = ("lightgbm", "dlinear", "gated_dwtcn")
OPTIONAL_PRICE_COLUMNS = (
    "up_limit", "high_limit", "limit_up_price", "down_limit", "low_limit",
    "limit_down_price", "is_paused", "paused", "suspend", "suspended",
    "is_suspended", "trade_status", "is_st", "st", "risk_warning",
)


def _atomic_parquet(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=index)
    tmp.replace(path)


def _atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def load_live_predictions(
    exp_dir: Path, smooth_window: int, start: pd.Timestamp,
) -> pd.DataFrame:
    """Load frozen model predictions, smooth, rank, and compute EW score."""
    merged = None
    for model in MODELS:
        path = exp_dir / f"predictions_live_{model}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing live prediction: {path}")
        part = normalize_keys(pd.read_parquet(path))
        part = part[["time", "stock_id", "y_pred"]].rename(columns={"y_pred": model})
        merged = part if merged is None else merged.merge(part, on=["time", "stock_id"], how="inner")
    merged = merged.drop_duplicates(["time", "stock_id"], keep="last")
    merged = merged.sort_values(["time", "stock_id"]).reset_index(drop=True)
    dates = sorted(pd.to_datetime(merged["time"].unique()))
    before = [d for d in dates if d < start]
    if smooth_window > 1 and len(before) < smooth_window - 1:
        raise ValueError(
            f"EW {smooth_window}d needs {smooth_window - 1} warm-up dates before {start.date()}, "
            f"but predictions_live_* contain only {len(before)}. Run live_predict.py again."
        )
    smoothed = smooth_predictions(merged, MODELS, smooth_window)
    for model in MODELS:
        smoothed[f"{model}_rank"] = smoothed.groupby("time")[model].rank(pct=True)
    rank_cols = [f"{m}_rank" for m in MODELS]
    smoothed["ew_score"] = smoothed[rank_cols].mean(axis=1)
    smoothed["rank"] = smoothed.groupby("time")["ew_score"].rank(method="first", ascending=False)
    return smoothed[smoothed["time"] >= start].sort_values(["time", "rank"]).reset_index(drop=True)


def load_prices(path: Path, start: pd.Timestamp, store: LiveStore | None = None) -> pd.DataFrame:
    if store is not None and store.exists():
        prices = store.read("base_panel", start=start)
        required = {"time", "stock_id", "close", "pre_close"}
        missing = required - set(prices.columns)
        if not missing:
            return prices[prices["time"] >= start].copy()
    try:
        import pyarrow.parquet as pq
        available = set(pq.ParquetFile(path).schema.names)
        columns = [c for c in ("time", "stock_id", "close", "pre_close", *OPTIONAL_PRICE_COLUMNS)
                   if c in available]
        prices = pd.read_parquet(path, columns=columns)
    except Exception:
        prices = pd.read_parquet(path)
    required = {"time", "stock_id", "close", "pre_close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Unified panel missing price columns: {sorted(missing)}")
    prices = normalize_keys(prices)
    return prices[prices["time"] >= start].copy()


def _website_ledger_tables(
    result: dict, config: RealBacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades = result["trade_log"].copy()
    snapshots = result["position_snapshots"].copy()
    if trades.empty:
        trades = pd.DataFrame(columns=[
            "date", "action", "stock_id", "shares", "lots", "price", "gross", "fee",
            "commission", "transfer_fee", "stamp_tax", "reference_price", "slippage_cost",
        ])
    else:
        is_buy = trades["action"].eq("BUY")
        buy_bps = trades.get("bin", pd.Series(2, index=trades.index)).map(
            config.buy_slippage_bps,
        ).fillna(0.0)
        slip_rate = np.where(
            is_buy, buy_bps / 10000.0, config.sell_slippage_bps / 10000.0,
        )
        trades["reference_price"] = np.where(
            is_buy, trades["price"] / (1.0 + slip_rate),
            trades["price"] / (1.0 - slip_rate),
        )
        trades["slippage_cost"] = trades["shares"] * (trades["price"] - trades["reference_price"]).abs()
        trades["commission"] = np.maximum(trades["gross"] * config.commission_rate, config.min_commission)
        trades["stamp_tax"] = np.where(is_buy, 0.0, trades["gross"] * config.stamp_tax_rate)
        trades["transfer_fee"] = 0.0

    daily = pd.concat([
        result["daily_equity"], result["daily_cash"], result["daily_market_value"],
        result["daily_n_positions"], result["daily_n_buys"], result["daily_n_sells"],
        result["daily_filtered"].rename("blocked_buys"),
        result["daily_blocked_sells"].rename("blocked_sells"),
        result["daily_returns"].rename("daily_return"), result["daily_nav"],
    ], axis=1).reset_index().rename(columns={"index": "date"})

    if trades.empty:
        cost_daily = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    else:
        temp = trades.copy()
        temp["gross_buy"] = np.where(temp["action"].eq("BUY"), temp["gross"], 0.0)
        temp["gross_sell"] = np.where(temp["action"].eq("SELL"), temp["gross"], 0.0)
        temp["realized_pnl_daily"] = temp.get("realized_pnl", pd.Series(0.0, index=temp.index)).fillna(0.0)
        cost_daily = temp.groupby("date")[[
            "gross_buy", "gross_sell", "commission", "transfer_fee", "stamp_tax",
            "slippage_cost", "realized_pnl_daily",
        ]].sum()
    daily = daily.merge(cost_daily, left_on="date", right_index=True, how="left")
    for col in ("gross_buy", "gross_sell", "commission", "transfer_fee", "stamp_tax",
                "slippage_cost", "realized_pnl_daily"):
        if col not in daily:
            daily[col] = 0.0
        daily[col] = daily[col].fillna(0.0)
    daily["fees"] = daily["commission"] + daily["transfer_fee"] + daily["stamp_tax"]
    daily["friction_cost"] = daily["fees"] + daily["slippage_cost"]
    daily["turnover"] = np.where(
        daily["equity"] > 0,
        (daily["gross_buy"] + daily["gross_sell"]) / (2.0 * daily["equity"]), 0.0,
    )
    daily["realized_pnl"] = daily["realized_pnl_daily"]
    if snapshots.empty:
        daily["unrealized_pnl"] = 0.0
    else:
        unrealized = snapshots.groupby("date")["unrealized_pnl"].sum()
        daily = daily.merge(unrealized.rename("unrealized_pnl"), left_on="date", right_index=True, how="left")
        daily["unrealized_pnl"] = daily["unrealized_pnl"].fillna(0.0)
    orders = pd.DataFrame(columns=["date", "stock_id", "side", "shares", "status", "reason"])
    return daily, trades, orders


def build_ledger(
    exp_dir: Path,
    price_path: Path,
    output_dir: Path,
    start: pd.Timestamp,
    capital: float,
    top_n: int,
    smooth_window: int,
    commission_rate: float,
    min_commission: float,
    stamp_tax_rate: float,
    buy_slippage_bps: tuple[float, float, float],
    sell_slippage_bps: float,
    cash_ratio: float,
    buffer_exit_n: int = 50,
    buffer_mode: str = "fixed",
    regime_csv_path: str = "",
    regime_defense_score: float | None = None,
    regime_danger_ratio: float = 0.80,
    regime_recovery_steps: int = 3,
) -> dict:
    scores = load_live_predictions(exp_dir, smooth_window, start)
    store = LiveStore()
    prices = load_prices(price_path, start, store=store)
    config = RealBacktestConfig(
        capital=capital, max_stocks=top_n, pred_col="ew_score",
        position_sizing="budget", cash_ratio=cash_ratio,
        start_date=start,
        commission_rate=commission_rate, min_commission=min_commission,
        stamp_tax_rate=stamp_tax_rate,
        buy_slippage_bps={1: buy_slippage_bps[0], 2: buy_slippage_bps[1],
                          3: buy_slippage_bps[2]},
        sell_slippage_bps=sell_slippage_bps,
        buffer_exit_n=buffer_exit_n,
        buffer_mode=buffer_mode,
        regime_csv_path=regime_csv_path,
        regime_defense_score=regime_defense_score,
        regime_danger_ratio=regime_danger_ratio,
        regime_recovery_steps=regime_recovery_steps,
    )
    result = run_real_backtest(scores, prices, config=config)
    account, trades, orders = _website_ledger_tables(result, config)
    summary = dict(result["summary"])
    summary.update({
        "total_fees": float(account["fees"].sum()),
        "total_slippage_cost": float(account["slippage_cost"].sum()),
        "total_friction_cost": float(account["friction_cost"].sum()),
        "execution_assumptions": {
            "position_sizing": "top50_bin_weighted_budget",
            "bin_budget_weights": {"strong": 1.0, "elite": 2.0},
            "cash_ratio": float(cash_ratio),
            "commission_rate_each_side": float(commission_rate),
            "minimum_commission_per_order": float(min_commission),
            "stamp_tax_sell_rate": float(stamp_tax_rate),
            "transfer_fee_each_side_rate": 0.0,
            "buy_slippage_bps_by_tier": {
                "1": float(buy_slippage_bps[0]), "2": float(buy_slippage_bps[1]),
                "3": float(buy_slippage_bps[2]),
            },
            "sell_slippage_bps": float(sell_slippage_bps),
            "lot_size": int(config.lot_size),
        },
    })
    _atomic_parquet(account, output_dir / "real_account_ew5d.parquet")
    _atomic_parquet(result["daily_returns"].rename("strategy_return").reset_index(),
                    output_dir / "real_returns_ew5d.parquet")
    _atomic_parquet(result["position_snapshots"], output_dir / "real_positions_ew5d.parquet")
    _atomic_parquet(trades, output_dir / "real_trades_ew5d.parquet")
    _atomic_parquet(orders, output_dir / "real_orders_ew5d.parquet")
    _atomic_parquet(scores, output_dir / "real_scores_ew5d.parquet")
    _atomic_json(summary, output_dir / "real_summary_ew5d.json")
    result["daily_account"] = account.set_index("date")
    result["trade_log"] = trades
    result["order_log"] = orders
    result["summary"] = summary
    return result
