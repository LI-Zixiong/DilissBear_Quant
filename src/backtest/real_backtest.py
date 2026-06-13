"""
Account-based real backtest engine for A-share open-to-open execution.

Trading convention
------------------
Signal date T close      : model / Bayes score is known after close.
Entry date  T+1 open     : buy at open if the stock is tradable and not open limit-up.
Target exit T+2 open     : attempt to sell at open. If the stock is not tradable or
                           opens limit-down, carry the position and try again on the
                           next trading day's open.

Main differences from the quick daily-independent engine
--------------------------------------------------------
- True account simulation: cash, holdings, equity and delayed exits are tracked.
- No stock-level future filter: a stock is not removed just because T+2 price is missing.
- Buy cash check includes buy commission.
- Sell proceeds deduct sell commission + stamp tax.
- Integer lots only.
- Optional exact limit / suspension fields are used when available.

Required price_df columns
-------------------------
[time, stock_id, open, pre_close]

Optional price_df columns, used automatically if present
--------------------------------------------------------
up_limit / high_limit / limit_up_price
    Exact upper limit price.

down_limit / low_limit / limit_down_price
    Exact lower limit price.

is_paused / paused / suspend / suspended / is_suspended / trade_status
    Suspension / tradability marker. trade_status values containing 停牌/suspend/paused
    are treated as not tradable.

is_st / st / risk_warning
    ST marker. Used only when exact limit columns are absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.metrics import summarize_backtest


@dataclass
class RealBacktestConfig:
    # Account / execution
    capital: float = 200_000.0
    lot_size: int = 100
    commission_rate: float = 0.0003       # buy and sell commission: 万3
    stamp_tax_rate: float = 0.0005        # sell-side stamp tax: 万5, so sell total ~= 万8
    min_commission: float = 0.0           # set to 5.0 if you want brokerage minimum fee

    # Bin execution: rank_pct bins are [0.94,0.96), [0.96,0.985), [0.985,1.0]
    # bin_lots maps bin1/bin2/bin3. Default skips buffer, buys strong 1 lot, elite 2 lots.
    bin_lots: tuple[int, int, int] = (0, 1, 2)
    bin_edges: tuple[float, float, float] = (0.94, 0.96, 0.985)
    max_stocks: int = 50                 # max new buys per entry day
    pred_col: str = "bayes_score"

    # Trading realism switches
    block_limit_up_buy: bool = True
    block_limit_down_sell: bool = True
    skip_suspended: bool = True
    allow_duplicate_position: bool = False

    # Position sizing mode
    # "fixed": bin_lots slots per stock (price-distorted, simple)
    # "budget": bin-weighted cash budget → per-stock target → round lots (price-fair)
    position_sizing: str = "fixed"
    cash_ratio: float = 0.80  # fraction of post-sell cash to deploy (budget mode)

    # Price column names. Keep open-to-open by default.
    open_col: str = "open"
    pre_close_col: str = "pre_close"

    def __post_init__(self) -> None:
        if self.capital <= 0:
            raise ValueError(f"capital must be positive, got {self.capital}")
        if self.lot_size <= 0:
            raise ValueError(f"lot_size must be positive, got {self.lot_size}")
        if self.commission_rate < 0 or self.stamp_tax_rate < 0 or self.min_commission < 0:
            raise ValueError("transaction cost rates and min_commission must be non-negative")
        if len(self.bin_lots) != 3:
            raise ValueError(f"bin_lots must have 3 elements, got {len(self.bin_lots)}")
        if len(self.bin_edges) != 3:
            raise ValueError(f"bin_edges must have 3 elements, got {len(self.bin_edges)}")
        if tuple(self.bin_edges) != tuple(sorted(self.bin_edges)):
            raise ValueError(f"bin_edges must be sorted ascending, got {self.bin_edges}")
        if self.max_stocks <= 0:
            raise ValueError(f"max_stocks must be positive, got {self.max_stocks}")
        if self.position_sizing not in ("fixed", "budget"):
            raise ValueError(f"position_sizing must be 'fixed' or 'budget', got {self.position_sizing}")
        if not (0.0 < self.cash_ratio <= 1.0):
            raise ValueError(f"cash_ratio must be in (0, 1], got {self.cash_ratio}")


def _is_finite_positive(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return np.isfinite(v) and v > 0


def _as_bool(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "停牌", "suspend", "suspended", "paused"}:
        return True
    if s in {"0", "false", "f", "no", "n", "正常", "trading", "active", "交易"}:
        return False
    return any(key in s for key in ["停牌", "suspend", "paused"])


def _row_is_suspended(row: dict[str, Any], available_cols: set[str]) -> bool:
    for col in ["is_paused", "paused", "suspend", "suspended", "is_suspended"]:
        if col in available_cols and _as_bool(row.get(col)):
            return True
    if "trade_status" in available_cols:
        return _as_bool(row.get("trade_status"))
    return False


def _row_is_st(row: dict[str, Any], available_cols: set[str]) -> bool:
    for col in ["is_st", "st", "risk_warning"]:
        if col in available_cols and _as_bool(row.get(col)):
            return True
    return False


def _limit_pct_from_code(stock_id: str, is_st: bool = False) -> float:
    """Fallback limit rule when exact up/down limit columns are not available."""
    code = str(stock_id).strip().zfill(6)
    if is_st:
        return 0.05
    # BSE / NEEQ-style prefixes; harmless if your universe excludes them.
    if code.startswith(("4", "8")):
        return 0.30
    # ChiNext / STAR Market.
    if code.startswith(("300", "301", "688")):
        return 0.20
    return 0.10


def _exact_limit_price(row: dict[str, Any], candidates: list[str]) -> float | None:
    for col in candidates:
        if col in row and _is_finite_positive(row.get(col)):
            return float(row[col])
    return None


def _is_open_limit_up(row: dict[str, Any], stock_id: str, available_cols: set[str]) -> bool:
    """Opened at limit-up, so buy is not assumed executable."""
    open_px = row.get("open")
    pre_close = row.get("pre_close")
    if not _is_finite_positive(open_px) or not _is_finite_positive(pre_close):
        return True

    exact = _exact_limit_price(row, ["up_limit", "high_limit", "limit_up", "limit_up_price"])
    if exact is not None:
        return float(open_px) >= exact - 1e-3

    is_st = _row_is_st(row, available_cols)
    limit_pct = _limit_pct_from_code(stock_id, is_st=is_st)
    limit_price = round(float(pre_close) * (1.0 + limit_pct), 2)
    return float(open_px) >= limit_price - 1e-3


def _is_open_limit_down(row: dict[str, Any], stock_id: str, available_cols: set[str]) -> bool:
    """Opened at limit-down, so sell is not assumed executable."""
    open_px = row.get("open")
    pre_close = row.get("pre_close")
    if not _is_finite_positive(open_px) or not _is_finite_positive(pre_close):
        return True

    exact = _exact_limit_price(row, ["down_limit", "low_limit", "limit_down", "limit_down_price"])
    if exact is not None:
        return float(open_px) <= exact + 1e-3

    is_st = _row_is_st(row, available_cols)
    limit_pct = _limit_pct_from_code(stock_id, is_st=is_st)
    limit_price = round(float(pre_close) * (1.0 - limit_pct), 2)
    return float(open_px) <= limit_price + 1e-3


def _assign_bin(rank_pct: float, edges: tuple[float, float, float]) -> int:
    """Map rank percentile to bin: 0=background, 1=buffer, 2=strong, 3=elite."""
    if not np.isfinite(rank_pct):
        return 0
    for i, edge in enumerate(edges):
        if rank_pct < edge:
            return i
    return len(edges)


def _commission(amount: float, config: RealBacktestConfig) -> float:
    if amount <= 0:
        return 0.0
    return max(amount * config.commission_rate, config.min_commission)


def _clean_price_frame(
    price_df: pd.DataFrame,
    date_col: str,
    stock_col: str,
    config: RealBacktestConfig,
) -> tuple[pd.DataFrame, set[str]]:
    prices = price_df.copy()
    prices[date_col] = pd.to_datetime(prices[date_col])
    prices[stock_col] = prices[stock_col].astype(str).str.strip().str.zfill(6)

    rename_map = {}
    if config.open_col != "open":
        rename_map[config.open_col] = "open"
    if config.pre_close_col != "pre_close":
        rename_map[config.pre_close_col] = "pre_close"
    if rename_map:
        prices = prices.rename(columns=rename_map)

    required = {date_col, stock_col, "open", "pre_close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"price_df missing columns: {missing}")

    # Keep the last duplicate record if duplicates exist; this prevents set_index ambiguity.
    prices = prices.sort_values([date_col, stock_col]).drop_duplicates([date_col, stock_col], keep="last")
    return prices, set(prices.columns)


def _clean_pred_frame(
    pred_df: pd.DataFrame,
    date_col: str,
    stock_col: str,
    pred_col: str,
) -> pd.DataFrame:
    pred = pred_df.copy()
    pred[date_col] = pd.to_datetime(pred[date_col])
    pred[stock_col] = pred[stock_col].astype(str).str.strip().str.zfill(6)

    required = {date_col, stock_col, pred_col}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"pred_df missing columns: {missing}")

    pred[pred_col] = pd.to_numeric(pred[pred_col], errors="coerce")
    pred = pred.replace([np.inf, -np.inf], np.nan).dropna(subset=[date_col, stock_col, pred_col])
    # If there are duplicate scores for one stock on one day, keep the last record.
    pred = pred.sort_values([date_col, stock_col]).drop_duplicates([date_col, stock_col], keep="last")
    return pred


# Budget allocation weights: bin2 = 1.0x, bin3 = 2.0x  (bin1 skipped)
_BUDGET_BIN_WEIGHTS = {2: 1.0, 3: 2.0}


def _allocate_by_budget(
    eligible: pd.DataFrame,
    available_cash: float,
    config: RealBacktestConfig,
) -> tuple[list[dict[str, Any]], float]:
    """
    Budget-based lot allocation across bin2 + bin3.

    Algorithm:
      1. budget = available_cash * cash_ratio
      2. bin_budget = budget * (n_stocks_in_bin * bin_weight) / sum(all bins)
      3. per_stock_budget = bin_budget / n_stocks_in_bin
      4. lots = round(per_stock_budget / (price * lot_size))
      5. If total cost > bin_budget, skip lowest-ranked stocks.

    Returns (position_dicts, total_spent).
    """
    budget = available_cash * config.cash_ratio
    if budget <= 0:
        return [], 0.0

    # Separate by bin, already sorted by score desc
    bin_groups: dict[int, pd.DataFrame] = {}
    for bi in [2, 3]:
        grp = eligible[eligible["bin"] == bi]
        if not grp.empty:
            bin_groups[bi] = grp

    if not bin_groups:
        return [], 0.0

    # Compute bin-level budgets
    bin_total_weight = sum(
        len(grp) * _BUDGET_BIN_WEIGHTS[bi] for bi, grp in bin_groups.items()
    )
    bin_budgets: dict[int, float] = {}
    for bi, grp in bin_groups.items():
        bin_budgets[bi] = budget * (len(grp) * _BUDGET_BIN_WEIGHTS[bi]) / bin_total_weight

    positions_out: list[dict[str, Any]] = []
    total_spent = 0.0
    lot_size = config.lot_size

    for bi in [3, 2]:  # bin3 first (higher priority)
        if bi not in bin_groups:
            continue
        grp = bin_groups[bi]
        n_stocks = len(grp)
        per_stock = bin_budgets[bi] / n_stocks

        # Compute target lots for each stock (round to nearest integer)
        stock_plans = []
        for _, row in grp.iterrows():
            price_per_lot = float(row["open"]) * lot_size
            if price_per_lot <= 0:
                continue
            raw_lots = per_stock / price_per_lot
            n_lots = max(1, int(round(raw_lots)))
            entry_gross = n_lots * price_per_lot
            stock_plans.append({
                "stock_id": str(row["stock_id"]),
                "open": float(row["open"]),
                "score": float(row["score"]),
                "rank_pct": float(row["rank_pct"]),
                "bin": bi,
                "n_lots": n_lots,
                "shares": n_lots * lot_size,
                "entry_gross": entry_gross,
                "buy_fee": _commission(entry_gross, config),
            })

        # Sort by score desc; skip lowest-ranked if overspent
        stock_plans.sort(key=lambda x: x["score"], reverse=True)
        bin_spent = 0.0
        bin_positions = []
        for plan in stock_plans:
            if bin_spent + plan["entry_gross"] > bin_budgets[bi] + 1e-9:
                continue  # skip lowest-ranked
            bin_spent += plan["entry_gross"]
            bin_positions.append(plan)

        positions_out.extend(bin_positions)
        total_spent += bin_spent

    return positions_out, total_spent


def _build_buy_targets(
    current_date: pd.Timestamp,
    entry_signals: dict,
    today_px: dict,
    positions: list[dict[str, Any]],
    cash: float,
    config: RealBacktestConfig,
    stock_col: str,
    price_cols: set[str],
) -> tuple[dict[str, dict[str, Any]], int]:
    """
    Build today's buy targets from signal predictions (before selling).

    Returns (buy_targets, n_filtered) where buy_targets maps stock_id → {lots, shares, open,
    score, rank_pct, bin, entry_gross (budget only), buy_fee (budget only), signal_date, target_exit_date}.
    """
    # Budget base = cash + current market value (smart-hold locks value in positions).
    pre_sell_mv = 0.0
    for pos in positions:
        row_ = today_px.get(pos["stock_id"])
        if row_ is not None and _is_finite_positive(row_.get("open")):
            pre_sell_mv += pos["shares"] * float(row_["open"])
        else:
            pre_sell_mv += pos["shares"] * float(pos.get("last_price", pos["entry_open"]))
    deployable = cash + pre_sell_mv

    buy_targets: dict[str, dict[str, Any]] = {}
    n_filtered = 0

    for signal_date, target_exit_date, day_pred in entry_signals.get(current_date, []):
        rows = []
        for _, pred_row in day_pred.iterrows():
            sid = pred_row[stock_col]
            row = today_px.get(sid)
            if row is None:
                continue
            open_px = row.get("open")
            pre_close = row.get("pre_close")
            if not _is_finite_positive(open_px) or not _is_finite_positive(pre_close):
                continue
            rows.append({
                "stock_id": sid,
                "score": float(pred_row[config.pred_col]),
                "open": float(open_px),
                "pre_close": float(pre_close),
                "raw_price_row": row,
            })

        if not rows:
            continue

        stocks_df = pd.DataFrame(rows)
        stocks_df = stocks_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["stock_id", "score", "open", "pre_close"])
        if stocks_df.empty:
            continue

        stocks_df["rank_pct"] = stocks_df["score"].rank(pct=True)

        def _buy_blocked(r: pd.Series) -> bool:
            row_ = dict(r["raw_price_row"])
            row_["open"] = r["open"]
            row_["pre_close"] = r["pre_close"]
            if config.skip_suspended and _row_is_suspended(row_, price_cols):
                return True
            if config.block_limit_up_buy and _is_open_limit_up(row_, r["stock_id"], price_cols):
                return True
            return False

        stocks_df["buy_blocked"] = stocks_df.apply(_buy_blocked, axis=1)
        n_filtered += int(stocks_df["buy_blocked"].sum())
        eligible = stocks_df[~stocks_df["buy_blocked"]].copy()
        if eligible.empty:
            continue

        eligible["bin"] = eligible["rank_pct"].apply(lambda x: _assign_bin(float(x), config.bin_edges))
        eligible = eligible.sort_values("score", ascending=False)

        if config.position_sizing == "budget":
            budget_plans, _ = _allocate_by_budget(eligible, deployable, config)
            for plan in budget_plans:
                buy_targets[plan["stock_id"]] = {
                    "lots": plan["n_lots"], "shares": plan["shares"],
                    "open": plan["open"], "score": plan["score"],
                    "rank_pct": plan["rank_pct"], "bin": plan["bin"],
                    "entry_gross": plan["entry_gross"], "buy_fee": plan["buy_fee"],
                    "signal_date": signal_date, "target_exit_date": target_exit_date,
                }
        else:
            for _, stock in eligible.iterrows():
                sid = str(stock["stock_id"])
                bi = int(stock["bin"])
                if bi <= 0:
                    continue
                n_lots = int(config.bin_lots[bi - 1])
                if n_lots <= 0:
                    continue
                if sid in buy_targets:
                    continue
                buy_targets[sid] = {
                    "lots": n_lots, "shares": n_lots * config.lot_size,
                    "open": float(stock["open"]),
                    "score": float(stock["score"]),
                    "rank_pct": float(stock["rank_pct"]), "bin": bi,
                    "signal_date": signal_date, "target_exit_date": target_exit_date,
                }

    return buy_targets, n_filtered


def run_real_backtest(
    pred_df: pd.DataFrame,
    price_df: pd.DataFrame,
    config: RealBacktestConfig | None = None,
    date_col: str = "time",
    stock_col: str = "stock_id",
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """
    Account-based real backtest: T close signal -> T+1 open buy -> T+2 open target sell.

    Returns
    -------
    dict with:
      daily_returns, daily_nav, daily_equity, daily_cash, daily_market_value,
      daily_n_positions, daily_n_buys, daily_n_sells, daily_capital_used,
      daily_filtered, daily_blocked_sells, trade_log, position_snapshots, summary.
    """
    if config is None:
        config = RealBacktestConfig()

    pred = _clean_pred_frame(pred_df, date_col, stock_col, config.pred_col)
    prices, price_cols = _clean_price_frame(price_df, date_col, stock_col, config)

    price_dates = [pd.Timestamp(d) for d in np.sort(prices[date_col].dropna().unique())]
    if len(price_dates) < 3:
        raise ValueError("price_df must contain at least 3 trading dates")
    date_to_idx = {d: i for i, d in enumerate(price_dates)}

    # Price lookup: {date: {stock_id: row_dict}}
    price_lookup: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}
    for d, grp in prices.groupby(date_col, sort=True):
        dd = pd.Timestamp(d)
        price_lookup[dd] = grp.set_index(stock_col).to_dict("index")

    # Map each signal date T to entry date T+1 and target exit date T+2.
    entry_signals: dict[pd.Timestamp, list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]] = {}
    skipped_signal_dates = 0
    for signal_date, day_pred in pred.groupby(date_col, sort=True):
        signal_date = pd.Timestamp(signal_date)
        idx = date_to_idx.get(signal_date)
        if idx is None or idx + 2 >= len(price_dates):
            skipped_signal_dates += 1
            continue
        entry_date = price_dates[idx + 1]
        target_exit_date = price_dates[idx + 2]
        entry_signals.setdefault(entry_date, []).append((signal_date, target_exit_date, day_pred.copy()))

    if not entry_signals:
        raise ValueError("No valid signal dates can be mapped to T+1/T+2 trading dates")

    first_entry_date = min(entry_signals)
    first_entry_idx = date_to_idx[first_entry_date]

    cash = float(config.capital)
    positions: list[dict[str, Any]] = []
    next_position_id = 1

    equity_by_date: dict[pd.Timestamp, float] = {}
    cash_by_date: dict[pd.Timestamp, float] = {}
    mv_by_date: dict[pd.Timestamp, float] = {}
    n_pos_by_date: dict[pd.Timestamp, int] = {}
    n_buy_by_date: dict[pd.Timestamp, int] = {}
    n_sell_by_date: dict[pd.Timestamp, int] = {}
    capital_used_by_date: dict[pd.Timestamp, float] = {}
    filtered_by_date: dict[pd.Timestamp, int] = {}
    blocked_sell_by_date: dict[pd.Timestamp, int] = {}

    trade_log: list[dict[str, Any]] = []
    position_snapshots: list[dict[str, Any]] = []

    last_entry_date = max(entry_signals) if entry_signals else None

    for current_date in price_dates[first_entry_idx:]:
        # Stop early: no more signals and no open positions
        if last_entry_date is not None and current_date > last_entry_date and len(positions) == 0:
            break

        today_px = price_lookup.get(current_date, {})
        n_buys = 0
        n_sells = 0
        n_filtered = 0
        n_blocked_sells = 0
        capital_used_today = 0.0

        # ------------------------------------------------------------------
        # 1) Build today's buy targets.
        # ------------------------------------------------------------------
        buy_targets, n_filtered_today = _build_buy_targets(
            current_date, entry_signals, today_px, positions, cash,
            config, stock_col, price_cols,
        )
        n_filtered += n_filtered_today

        # ------------------------------------------------------------------
        # 2) Smart-hold: sell only delta, not full position, when stock stays in buy list.
        # ------------------------------------------------------------------
        held_stock_ids = {p["stock_id"] for p in positions}
        still_holding: list[dict[str, Any]] = []

        for pos in positions:
            sid = pos["stock_id"]
            due_to_sell = current_date >= pos["target_exit_date"]
            row = today_px.get(sid)

            if row is not None and _is_finite_positive(row.get("open")):
                pos["last_price"] = float(row["open"])
                pos["last_price_date"] = current_date

            if not due_to_sell:
                still_holding.append(pos)
                continue

            target = buy_targets.get(sid)
            if target is not None:
                # ── Smart hold: stock still in buy list, only trade delta ──
                delta_lots = target["lots"] - pos["lots"]
                px = float(row["open"]) if row is not None and _is_finite_positive(row.get("open")) else pos.get("last_price", pos["entry_open"])

                if delta_lots > 0:
                    # Need more: buy delta lots at buy commission (万3)
                    delta_shares = delta_lots * config.lot_size
                    delta_gross = delta_shares * px
                    delta_fee = _commission(delta_gross, config)
                    if delta_gross + delta_fee <= cash + 1e-9:
                        cash -= (delta_gross + delta_fee)
                        pos["shares"] += delta_shares
                        pos["lots"] += delta_lots
                        pos["entry_gross"] += delta_gross
                        pos["buy_fee"] += delta_fee
                        pos["entry_open"] = (pos["entry_open"] * (pos["lots"] - delta_lots) + px * delta_lots) / pos["lots"] if pos["lots"] > 0 else px
                        n_buys += 1
                        capital_used_today += delta_gross
                        trade_log.append({
                            "date": current_date, "action": "BUY",
                            "position_id": pos["position_id"], "stock_id": sid,
                            "shares": delta_shares, "lots": delta_lots,
                            "price": px, "gross": delta_gross, "fee": delta_fee,
                            "cash_after": cash,
                            "signal_date": pos["signal_date"], "entry_date": current_date,
                            "target_exit_date": target["target_exit_date"],
                            "bin": pos["bin"], "rank_pct": pos["rank_pct"],
                            "score": pos["score"],
                        })
                elif delta_lots < 0:
                    # Need fewer: sell excess at sell commission (万8)
                    excess_lots = -delta_lots
                    excess_shares = excess_lots * config.lot_size
                    gross_sell = excess_shares * px
                    sell_fee = _commission(gross_sell, config) + gross_sell * config.stamp_tax_rate
                    cash += gross_sell - sell_fee
                    pos["shares"] -= excess_shares
                    pos["lots"] -= excess_lots
                    pos["entry_gross"] -= (excess_lots * config.lot_size * pos["entry_open"])
                    n_sells += 1
                    trade_log.append({
                        "date": current_date, "action": "SELL",
                        "position_id": pos["position_id"], "stock_id": sid,
                        "shares": excess_shares, "lots": excess_lots,
                        "price": px, "gross": gross_sell, "fee": sell_fee,
                        "cash_after": cash,
                        "signal_date": pos["signal_date"], "entry_date": pos["entry_date"],
                        "target_exit_date": pos["target_exit_date"],
                        "delayed_exit_days": pos.get("delayed_exit_days", 0),
                        "realized_pnl": gross_sell - excess_lots * config.lot_size * pos["entry_open"] - pos["buy_fee"] * excess_lots / max(pos["lots"] + excess_lots, 1) - sell_fee,
                        "return_on_gross": np.nan,
                    })

                # Extend exit date
                pos["target_exit_date"] = target["target_exit_date"]
                pos["signal_date"] = target["signal_date"]
                pos["score"] = target["score"]
                pos["rank_pct"] = target["rank_pct"]
                pos["bin"] = target["bin"]
                pos["delayed_exit_days"] = 0
                still_holding.append(pos)
                del buy_targets[sid]
                continue

            # ── Stock NOT in buy list: full sell ──
            can_sell = row is not None and _is_finite_positive(row.get("open"))
            if can_sell and config.skip_suspended and _row_is_suspended(row, price_cols):
                can_sell = False
            if can_sell and config.block_limit_down_sell and _is_open_limit_down(row, sid, price_cols):
                can_sell = False

            if not can_sell:
                pos["delayed_exit_days"] = int(pos.get("delayed_exit_days", 0)) + 1
                still_holding.append(pos)
                n_blocked_sells += 1
                continue

            sell_open = float(row["open"])
            gross_sell = pos["shares"] * sell_open
            sell_commission = _commission(gross_sell, config)
            stamp_tax = gross_sell * config.stamp_tax_rate
            sell_fee = sell_commission + stamp_tax
            cash += gross_sell - sell_fee

            realized_pnl = gross_sell - pos["entry_gross"] - pos["buy_fee"] - sell_fee
            n_sells += 1
            trade_log.append({
                "date": current_date, "action": "SELL",
                "position_id": pos["position_id"], "stock_id": sid,
                "shares": pos["shares"], "lots": pos["lots"],
                "price": sell_open, "gross": gross_sell, "fee": sell_fee,
                "cash_after": cash,
                "signal_date": pos["signal_date"], "entry_date": pos["entry_date"],
                "target_exit_date": pos["target_exit_date"],
                "delayed_exit_days": pos.get("delayed_exit_days", 0),
                "realized_pnl": realized_pnl,
                "return_on_gross": realized_pnl / pos["entry_gross"] if pos["entry_gross"] > 0 else np.nan,
            })

        positions = still_holding
        held_stock_ids = {p["stock_id"] for p in positions}

        # ------------------------------------------------------------------
        # 3) Buy new stocks (in buy_targets but not currently held).
        # ------------------------------------------------------------------
        new_plans = sorted(
            [(sid, t) for sid, t in buy_targets.items() if sid not in held_stock_ids],
            key=lambda x: x[1]["score"], reverse=True,
        )
        n_new = 0
        for sid, target in new_plans:
            if n_new >= config.max_stocks:
                break

            entry_gross = target.get("entry_gross", target["shares"] * target["open"])
            buy_fee = target.get("buy_fee", _commission(entry_gross, config))
            cash_needed = entry_gross + buy_fee
            if cash_needed > cash + 1e-9:
                continue

            cash -= cash_needed
            position = {
                "position_id": next_position_id,
                "stock_id": sid,
                "shares": target["shares"],
                "lots": target["lots"],
                "entry_open": target["open"],
                "entry_gross": entry_gross,
                "buy_fee": buy_fee,
                "signal_date": target["signal_date"],
                "entry_date": current_date,
                "target_exit_date": target["target_exit_date"],
                "score": target["score"],
                "rank_pct": target["rank_pct"],
                "bin": target["bin"],
                "last_price": target["open"],
                "last_price_date": current_date,
                "delayed_exit_days": 0,
            }
            positions.append(position)
            held_stock_ids.add(sid)
            n_buys += 1
            n_new += 1
            capital_used_today += entry_gross

            trade_log.append({
                "date": current_date, "action": "BUY",
                "position_id": next_position_id, "stock_id": sid,
                "shares": target["shares"], "lots": target["lots"],
                "price": target["open"], "gross": entry_gross, "fee": buy_fee,
                "cash_after": cash,
                "signal_date": target["signal_date"], "entry_date": current_date,
                "target_exit_date": target["target_exit_date"],
                "bin": target["bin"], "rank_pct": target["rank_pct"],
                "score": target["score"],
            })
            next_position_id += 1

        # ------------------------------------------------------------------
        # 3) Mark account equity at current open after sell/buy processing.
        # ------------------------------------------------------------------
        market_value = 0.0
        for pos in positions:
            sid = pos["stock_id"]
            row = today_px.get(sid)
            if row is not None and _is_finite_positive(row.get("open")):
                mark_price = float(row["open"])
                pos["last_price"] = mark_price
                pos["last_price_date"] = current_date
            else:
                mark_price = float(pos.get("last_price", pos["entry_open"]))
            market_value += pos["shares"] * mark_price

            position_snapshots.append({
                "date": current_date,
                "position_id": pos["position_id"],
                "stock_id": sid,
                "shares": pos["shares"],
                "lots": pos["lots"],
                "entry_date": pos["entry_date"],
                "target_exit_date": pos["target_exit_date"],
                "last_price": mark_price,
                "market_value": pos["shares"] * mark_price,
                "delayed_exit_days": pos.get("delayed_exit_days", 0),
            })

        equity = cash + market_value
        equity_by_date[current_date] = equity
        cash_by_date[current_date] = cash
        mv_by_date[current_date] = market_value
        n_pos_by_date[current_date] = len(positions)
        n_buy_by_date[current_date] = n_buys
        n_sell_by_date[current_date] = n_sells
        capital_used_by_date[current_date] = capital_used_today
        filtered_by_date[current_date] = n_filtered
        blocked_sell_by_date[current_date] = n_blocked_sells

    if not equity_by_date:
        raise ValueError("No valid backtest periods generated")

    daily_equity = pd.Series(equity_by_date).sort_index()
    daily_equity.name = "equity"

    daily_nav = daily_equity / float(config.capital)
    daily_nav.name = "nav"

    daily_returns = daily_nav.pct_change().fillna(daily_nav.iloc[0] - 1.0)
    daily_returns.name = "strategy_return"

    daily_cash = pd.Series(cash_by_date).sort_index()
    daily_cash.name = "cash"
    daily_market_value = pd.Series(mv_by_date).sort_index()
    daily_market_value.name = "market_value"
    daily_n_positions = pd.Series(n_pos_by_date).sort_index()
    daily_n_positions.name = "n_positions"
    daily_n_buys = pd.Series(n_buy_by_date).sort_index()
    daily_n_buys.name = "n_buys"
    daily_n_sells = pd.Series(n_sell_by_date).sort_index()
    daily_n_sells.name = "n_sells"
    daily_capital_used = pd.Series(capital_used_by_date).sort_index()
    daily_capital_used.name = "capital_used"
    daily_filtered = pd.Series(filtered_by_date).sort_index()
    daily_filtered.name = "filtered_buys"
    daily_blocked_sells = pd.Series(blocked_sell_by_date).sort_index()
    daily_blocked_sells.name = "blocked_sells"

    summary = summarize_backtest(returns=daily_returns, periods_per_year=periods_per_year)
    summary["final_nav"] = float(daily_nav.iloc[-1])
    summary["final_equity"] = float(daily_equity.iloc[-1])
    summary["total_pnl"] = float(daily_equity.iloc[-1] - config.capital)
    summary["capital"] = float(config.capital)
    summary["mean_n_positions"] = float(daily_n_positions.mean())
    summary["max_n_positions"] = int(daily_n_positions.max())
    summary["mean_n_buys"] = float(daily_n_buys.mean())
    summary["mean_n_sells"] = float(daily_n_sells.mean())
    summary["mean_capital_used"] = float(daily_capital_used.mean())
    summary["mean_capital_used_pct"] = float((daily_capital_used / daily_equity).mean())
    summary["mean_market_value"] = float(daily_market_value.mean())
    summary["mean_market_value_pct"] = float((daily_market_value / daily_equity).mean())
    summary["mean_cash"] = float(daily_cash.mean())
    summary["mean_cash_pct"] = float((daily_cash / daily_equity).mean())
    summary["mean_filtered"] = float(daily_filtered.mean())
    summary["total_filtered"] = int(daily_filtered.sum())
    summary["mean_blocked_sells"] = float(daily_blocked_sells.mean())
    summary["total_blocked_sells"] = int(daily_blocked_sells.sum())
    summary["skipped_signal_dates"] = int(skipped_signal_dates)
    summary["open_positions_at_end"] = int(len(positions))

    trade_log_df = pd.DataFrame(trade_log)
    if not trade_log_df.empty:
        trade_log_df = trade_log_df.sort_values(["date", "position_id", "action"]).reset_index(drop=True)

    position_snapshots_df = pd.DataFrame(position_snapshots)
    if not position_snapshots_df.empty:
        position_snapshots_df = position_snapshots_df.sort_values(["date", "position_id"]).reset_index(drop=True)

    return {
        "daily_returns": daily_returns,
        "daily_nav": daily_nav,
        "daily_equity": daily_equity,
        "daily_cash": daily_cash,
        "daily_market_value": daily_market_value,
        "daily_n_positions": daily_n_positions,
        "daily_n_buys": daily_n_buys,
        "daily_n_sells": daily_n_sells,
        "daily_capital_used": daily_capital_used,
        "daily_filtered": daily_filtered,
        "daily_blocked_sells": daily_blocked_sells,
        "daily_positions": trade_log_df[trade_log_df["action"] == "BUY"]
        [["signal_date", "stock_id", "lots", "gross", "entry_date",
          "target_exit_date", "bin", "rank_pct", "score"]]
        .rename(columns={"gross": "cost"}).reset_index(drop=True)
        if not trade_log_df.empty else pd.DataFrame(),
        "trade_log": trade_log_df,
        "position_snapshots": position_snapshots_df,
        "summary": summary,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.backtest.ensemble_utils import normalize_keys

    pred_path = Path("reports/strategy_v1/evidence/bayes_scores_test.parquet")
    if not pred_path.exists():
        print("No bayes scores found — run full_eval first")
        sys.exit(0)

    print("Loading data...")
    scores = pd.read_parquet(pred_path)
    scores = normalize_keys(scores)
    print(f"  scores: {len(scores):,} rows, {scores['time'].nunique()} dates")

    # Minimal columns. Add optional exact limit/suspension columns here if your panel has them.
    price_cols = ["time", "stock_id", "open", "pre_close"]
    panel_path = Path("dataset/processed/unified_daily_panel.parquet")
    prices = pd.read_parquet(panel_path, columns=price_cols)
    prices = normalize_keys(prices)
    prices["time"] = pd.to_datetime(prices["time"])
    print(f"  prices: {len(prices):,} rows, {prices['time'].nunique()} dates")

    config = RealBacktestConfig(
        capital=200_000,
        bin_lots=(0, 1, 2),
        max_stocks=50,
        commission_rate=0.0003,
        stamp_tax_rate=0.0005,
        min_commission=0.0,
    )
    result = run_real_backtest(scores, prices, config)
    s = result["summary"]

    print("\n=== Account Real Backtest (T+1 open -> T+2 open target) ===")
    print(f"  Sharpe:             {s['sharpe_ratio']:.4f}")
    print(f"  NAV:                {s['final_nav']:.4f}")
    print(f"  Final equity:       {s['final_equity']:,.0f} CNY")
    print(f"  Total PnL:          {s['total_pnl']:,.0f} CNY")
    print(f"  MaxDD:              {s['max_drawdown']:.1%}")
    print(f"  AnnualRet:          {s['annualized_return']:.1%}")
    print(f"  Mean positions:     {s['mean_n_positions']:.1f}")
    print(f"  Max positions:      {s['max_n_positions']}")
    print(f"  Mean buys/day:      {s['mean_n_buys']:.1f}")
    print(f"  Mean sells/day:     {s['mean_n_sells']:.1f}")
    print(f"  Mean capital used:  {s['mean_capital_used']:,.0f} CNY ({s['mean_capital_used_pct']:.1%})")
    print(f"  Mean market value:  {s['mean_market_value']:,.0f} CNY ({s['mean_market_value_pct']:.1%})")
    print(f"  Mean cash:          {s['mean_cash']:,.0f} CNY ({s['mean_cash_pct']:.1%})")
    print(f"  Mean filtered buys: {s['mean_filtered']:.1f} stocks/day")
    print(f"  Blocked sells:      {s['total_blocked_sells']} total")
    print(f"  Open positions end: {s['open_positions_at_end']}")
