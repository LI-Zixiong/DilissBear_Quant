"""
Account-based real backtest engine for A-share close-close execution.

Trading convention
------------------
Signal date T close      : model / Bayes score is known after close.
Entry date  T close      : buy at close through post-market close-price trading.
Target exit T+1 close    : attempt to sell at close. If the stock closes limit-down,
                           carry the position and try again on the next trading day.

Main differences from the quick daily-independent engine
--------------------------------------------------------
- True account simulation: cash, holdings, equity and delayed exits are tracked.
- No stock-level future filter: a stock is not removed just because its exit price is missing.
- Buy cash check includes buy commission.
- Sell proceeds deduct sell commission + stamp tax.
- Integer lots only.
- Optional exact limit / suspension fields are used when available.

Required price_df columns
-------------------------
[time, stock_id, close, pre_close]

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

    # Rank-aware slippage (bps). Bin3 signals are most crowded → highest buy slip.
    buy_slippage_bps: dict[int, float] | None = None  # {bin: bps}, default 0 (post-close trading at close price)
    sell_slippage_bps: float = 0.0  # uniform sell slip, default 0 (post-close trading at close price)

    # Trading realism switches
    block_limit_up_buy: bool = True
    block_limit_down_sell: bool = True
    skip_suspended: bool = True
    allow_duplicate_position: bool = False

    # Strategy: buffer — stocks outside Top buffer_exit_n are force-sold.
    # 50 = no buffer (same as entry); 80 = hold until rank > 80.
    buffer_exit_n: int = 50
    # Strategy: buffer mode.
    #   "expand"  — retain old stocks within exit_n, allow > max_stocks total (current).
    #   "fixed"   — retain old stocks first, fill remaining slots from new entries, cap at max_stocks.
    buffer_mode: str = "expand"
    # Strategy: gross daily turnover cap, (buys + sells) / pre-trade equity. None = unlimited.
    max_daily_turnover: float | None = None
    # Strategy: partial adjustment speed. 1.0 = full rebalance; 0.5 = half.
    lambda_weight: float = 1.0
    # Strategy: only buy top-tier (bin 3) stocks, skip strong/buffer
    tier1_only: bool = False
    # Strategy: elite budget weight relative to strong (default 2.0 = 2x).
    elite_budget_weight: float = 2.0
    # Strategy: use continuous softmax weights instead of bin weights.
    use_softmax_weights: bool = False
    # Strategy: single-industry budget cap as fraction of deployed capital (0 = disabled).
    industry_cap: float = 0.0
    # Strategy: regime-based position defence.
    regime_csv_path: str = ""
    regime_defense_score: float | None = None  # Bear + score < this triggers defence
    regime_danger_ratio: float = 0.80           # cash_ratio when defence is active
    regime_recovery_steps: int = 3              # days to linearly recover to normal

    # Position sizing mode
    # Production logic: deploy 80% of pre-trade total assets across Top50,
    # with elite stocks receiving 2x the per-stock budget of strong stocks.
    # "fixed" is retained only for reproducing the earlier fixed-lot experiment.
    position_sizing: str = "budget"
    cash_ratio: float = 0.98
    start_date: str | pd.Timestamp | None = None

    # Price column names.
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
        if self.buffer_exit_n < self.max_stocks:
            raise ValueError(f"buffer_exit_n ({self.buffer_exit_n}) must be >= max_stocks ({self.max_stocks})")
        if self.buffer_mode not in ("expand", "fixed"):
            raise ValueError(f"buffer_mode must be 'expand' or 'fixed', got {self.buffer_mode}")
        if self.max_daily_turnover is not None and not (0.0 < self.max_daily_turnover <= 1.0):
            raise ValueError(
                f"max_daily_turnover must be in (0, 1] or None, got {self.max_daily_turnover}"
            )
        if self.lambda_weight <= 0 or self.lambda_weight > 1:
            raise ValueError(f"lambda_weight must be in (0, 1], got {self.lambda_weight}")
        if self.elite_budget_weight <= 0:
            raise ValueError(f"elite_budget_weight must be positive, got {self.elite_budget_weight}")
        if not (0.0 < self.regime_danger_ratio <= 1.0):
            raise ValueError(
                f"regime_danger_ratio must be in (0, 1], got {self.regime_danger_ratio}"
            )
        if (
            (self.regime_csv_path or self.regime_defense_score is not None)
            and self.regime_danger_ratio > self.cash_ratio
        ):
            raise ValueError(
                "regime_danger_ratio cannot exceed normal cash_ratio when regime defence is enabled"
            )
        if self.regime_recovery_steps < 0:
            raise ValueError(
                f"regime_recovery_steps must be non-negative, got {self.regime_recovery_steps}"
            )
        if self.industry_cap < 0 or self.industry_cap > 1:
            raise ValueError(f"industry_cap must be in [0, 1], got {self.industry_cap}")
        if self.buy_slippage_bps is None:
            self.buy_slippage_bps = {1: 0, 2: 0, 3: 0}


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


def _is_close_limit_up(row: dict[str, Any], stock_id: str, available_cols: set[str]) -> bool:
    """Closed at limit-up, so buy at close is not executable."""
    close_px = row.get("close")
    pre_close = row.get("pre_close")
    if not _is_finite_positive(close_px) or not _is_finite_positive(pre_close):
        return True

    exact = _exact_limit_price(row, ["up_limit", "high_limit", "limit_up", "limit_up_price"])
    if exact is not None:
        return float(close_px) >= exact - 1e-3

    is_st = _row_is_st(row, available_cols)
    limit_pct = _limit_pct_from_code(stock_id, is_st=is_st)
    limit_price = round(float(pre_close) * (1.0 + limit_pct), 2)
    return float(close_px) >= limit_price - 1e-3


def _is_close_limit_down(row: dict[str, Any], stock_id: str, available_cols: set[str]) -> bool:
    """Closed at limit-down, so sell at close is not executable."""
    close_px = row.get("close")
    pre_close = row.get("pre_close")
    if not _is_finite_positive(close_px) or not _is_finite_positive(pre_close):
        return True

    exact = _exact_limit_price(row, ["down_limit", "low_limit", "limit_down", "limit_down_price"])
    if exact is not None:
        return float(close_px) <= exact + 1e-3

    is_st = _row_is_st(row, available_cols)
    limit_pct = _limit_pct_from_code(stock_id, is_st=is_st)
    limit_price = round(float(pre_close) * (1.0 - limit_pct), 2)
    return float(close_px) <= limit_price + 1e-3


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


def _lambda_adjusted_lots(delta_lots: int, lambda_weight: float) -> int:
    """Apply partial rebalancing while preserving the sign and whole-lot execution."""
    if delta_lots == 0 or lambda_weight >= 1.0:
        return delta_lots
    adjusted = int(np.floor(abs(delta_lots) * lambda_weight + 0.5))
    if adjusted == 0:
        return 0
    return adjusted if delta_lots > 0 else -adjusted


def _lots_within_turnover(
    desired_lots: int,
    gross_per_lot: float,
    remaining_gross: float,
) -> int:
    """Clip a whole-lot adjustment to the remaining gross turnover budget."""
    if desired_lots <= 0 or gross_per_lot <= 0 or remaining_gross <= 0:
        return 0
    if np.isinf(remaining_gross):
        return desired_lots
    return min(desired_lots, int(np.floor((remaining_gross + 1e-9) / gross_per_lot)))


def _clean_price_frame(
    price_df: pd.DataFrame,
    date_col: str,
    stock_col: str,
    config: RealBacktestConfig,
) -> tuple[pd.DataFrame, set[str]]:
    prices = price_df.copy()
    prices[date_col] = pd.to_datetime(prices[date_col])
    prices[stock_col] = prices[stock_col].astype(str).str.strip().str.zfill(6)

    if config.pre_close_col != "pre_close":
        prices = prices.rename(columns={config.pre_close_col: "pre_close"})

    required = {date_col, stock_col, "close", "pre_close"}
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


def _allocate_by_budget(
    eligible: pd.DataFrame,
    available_cash: float,
    config: RealBacktestConfig,
    cash_ratio_override: float | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """
    Budget-based lot allocation across bin2 + bin3.

    Algorithm:
      1. budget = pre-trade total assets * cash_ratio
      2. one budget unit = budget / (2 * elite_count + strong_count)
      3. elite gets 2 units per stock; strong gets 1 unit per stock
      4. lots = round(per_stock_budget / (price * lot_size))
      5. If rounded total cost exceeds budget, remove lowest ranks first.

    Returns (position_dicts, total_spent).
    """
    cr = cash_ratio_override if cash_ratio_override is not None else config.cash_ratio
    budget = available_cash * cr
    if budget <= 0:
        return [], 0.0

    # Separate by bin.  When buffer is active, bin 0-1 survivors also participate
    # (at standard 1x weight); they should not be dropped just because their rank
    # fell below the elite/strong threshold.
    active_buffer = config.buffer_exit_n > config.max_stocks
    bin_range = (0, 1, 2, 3) if active_buffer else (2, 3)
    bin_groups: dict[int, pd.DataFrame] = {}
    for bi in bin_range:
        grp = eligible[eligible["bin"] == bi]
        if not grp.empty:
            bin_groups[bi] = grp

    if not bin_groups:
        return [], 0.0

    # Standard mode keeps the original strong/elite ratio.  Softmax mode uses
    # each selected stock's score directly and therefore removes the bin step.
    selected = pd.concat([bin_groups[bi] for bi in sorted(bin_groups)]).copy()
    if config.use_softmax_weights:
        score_values = selected["score"].to_numpy(dtype=float)
        exp_scores = np.exp(score_values - np.max(score_values))
        selected["_budget_weight"] = exp_scores / exp_scores.sum()
    else:
        selected["_budget_weight"] = np.where(
            selected["bin"].to_numpy(dtype=int) == 3,
            config.elite_budget_weight,
            1.0,
        )
        selected["_budget_weight"] /= selected["_budget_weight"].sum()

    lot_size = config.lot_size

    stock_plans = []
    for _, row in selected.sort_values("score", ascending=False).iterrows():
        bi = int(row["bin"])
        per_stock = budget * float(row["_budget_weight"])
        slip_bps = config.buy_slippage_bps.get(bi, 0)
        slip_rate = slip_bps / 10000.0
        fill_price = float(row["close"]) * (1.0 + slip_rate)
        price_per_lot = fill_price * lot_size
        if price_per_lot <= 0:
            continue
        raw_lots = per_stock / price_per_lot
        n_lots = max(1, int(round(raw_lots)))
        entry_gross = n_lots * price_per_lot
        stock_plans.append({
            "stock_id": str(row["stock_id"]),
            "close": fill_price,
            "score": float(row["score"]),
            "rank_pct": float(row["rank_pct"]),
            "bin": bi,
            "n_lots": n_lots,
            "shares": n_lots * lot_size,
            "entry_gross": entry_gross,
            "buy_fee": _commission(entry_gross, config),
        })

    # Rounding can push the plan above 80%.  Preserve the highest-ranked prefix
    # and remove the lowest-ranked name until the gross plan fits the budget.
    stock_plans.sort(key=lambda x: x["score"], reverse=True)
    total_spent = sum(plan["entry_gross"] for plan in stock_plans)
    while stock_plans and total_spent > budget + 1e-9:
        removed = stock_plans.pop()
        total_spent -= removed["entry_gross"]

    return stock_plans, total_spent


def _build_buy_targets(
    current_date: pd.Timestamp,
    entry_signals: dict,
    today_px: dict,
    positions: list[dict[str, Any]],
    cash: float,
    config: RealBacktestConfig,
    stock_col: str,
    price_cols: set[str],
    cash_ratio_override: float | None = None,
    industry_map: dict[str, str] | None = None,
) -> tuple[dict[str, dict[str, Any]], int]:
    """
    Build today's buy targets from signal predictions (before selling).

    Buffer logic (when buffer_exit_n > max_stocks):
      1. Full stock ranking.
      2. Held stocks within buffer_exit_n survive (capped at max_stocks by rank).
      3. Remaining slots filled from today's Top entry (excluding survivors).
      4. All target stocks are assigned equal-target budget weights.

    Returns (buy_targets, n_filtered).  buy_targets maps stock_id → {lots,
    shares, open, score, rank_pct, bin, entry_gross, buy_fee, signal_date,
    target_exit_date}.
    """
    pre_sell_mv = 0.0
    for pos in positions:
        row_ = today_px.get(pos["stock_id"])
        if row_ is not None and _is_finite_positive(row_.get("close")):
            pre_sell_mv += pos["shares"] * float(row_["close"])
        else:
            pre_sell_mv += pos["shares"] * float(pos.get("last_price", pos["entry_open"]))
    deployable = cash + pre_sell_mv

    buy_targets: dict[str, dict[str, Any]] = {}
    n_filtered = 0
    held_ids = {str(pos["stock_id"]) for pos in positions}

    for signal_date, target_exit_date, day_pred in entry_signals.get(current_date, []):
        ranked = day_pred[[stock_col, config.pred_col]].copy()
        ranked["rank_pct"] = ranked[config.pred_col].rank(pct=True)
        ranked = ranked.sort_values(config.pred_col, ascending=False)
        ranked["ordinal_rank"] = np.arange(1, len(ranked) + 1)
        pred_lookup = ranked.set_index(stock_col)[
            [config.pred_col, "rank_pct"]
        ].to_dict("index")
        held_set = held_ids.copy()

        # ── Step 1: determine target membership (who stays, who enters) ──
        target_ids: list[str] = []
        is_held: dict[str, bool] = {}
        n_survivors = 0

        if config.buffer_exit_n > config.max_stocks:
            survivor_df = ranked[
                ranked[stock_col].isin(held_set)
                & (ranked["ordinal_rank"] <= config.buffer_exit_n)
            ].sort_values(config.pred_col, ascending=False)

            if config.buffer_mode == "fixed":
                survivor_df = survivor_df.head(config.max_stocks)
            elif config.buffer_mode == "expand":
                survivor_df = survivor_df.head(config.max_stocks * 3)

            for _, s in survivor_df.iterrows():
                sid = str(s[stock_col])
                target_ids.append(sid)
                is_held[sid] = True
                n_survivors += 1

        # Fill remaining slots from Top entries (not already held survivors)
        active_buffer = config.buffer_exit_n > config.max_stocks
        max_tg = (config.max_stocks
                  if not active_buffer or config.buffer_mode == "fixed"
                  else config.max_stocks * 3)
        slots = max_tg - len(target_ids)
        if slots > 0:
            for _, pred_row in ranked.iterrows():
                sid = str(pred_row[stock_col])
                if sid in set(target_ids):
                    continue
                if slots <= 0:
                    break
                target_ids.append(sid)
                is_held[sid] = sid in held_set
                slots -= 1

        # ── Step 2: build price rows; buy_blocked only applies to NEW entries ──
        rows = []
        for sid in target_ids:
            row = today_px.get(sid)
            if row is None:
                n_filtered += 1
                continue
            close_px = row.get("close")
            pre_close = row.get("pre_close")
            if not _is_finite_positive(close_px) or not _is_finite_positive(pre_close):
                n_filtered += 1
                continue
            pred_row = pred_lookup.get(sid)
            if pred_row is None:
                continue
            # Already-held stocks: block_buy only if suspended.  Limit-up on a
            # held stock does NOT change its membership; it just means "can't
            # increase position today", handled by the executor.
            blk = False
            if not is_held.get(sid, False):
                row_dict = dict(row)
                if config.skip_suspended and _row_is_suspended(row_dict, price_cols):
                    blk = True
                elif config.block_limit_up_buy and _is_close_limit_up(
                    row_dict, sid, price_cols
                ):
                    blk = True
            elif config.skip_suspended and _row_is_suspended(dict(row), price_cols):
                blk = True  # suspension affects everyone
            if blk and not is_held.get(sid, False):
                n_filtered += 1
                continue
            rows.append({
                "stock_id": sid,
                "score": float(pred_row[config.pred_col]),
                "rank_pct": float(pred_row["rank_pct"]),
                "close": float(close_px),
                "pre_close": float(pre_close),
                "raw_price_row": row,
                "_is_held": is_held.get(sid, False),
            })

        if not rows:
            continue

        stocks_df = pd.DataFrame(rows)
        stocks_df = stocks_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["stock_id", "score", "close", "pre_close"])
        if stocks_df.empty:
            continue

        # Bin assignment for ALL target stocks (buffer survivors included)
        stocks_df["bin"] = stocks_df["rank_pct"].apply(
            lambda x: _assign_bin(float(x), config.bin_edges)
        )
        stocks_df = stocks_df.sort_values("score", ascending=False)

        # buy_blocked flag: only true for NEW entries that can't be purchased.
        # Held stocks that are limit-up stay in the target set; executor handles
        # delta (can't buy more, won't sell).
        stocks_df["_can_buy"] = ~(stocks_df["_is_held"] & (
            stocks_df.apply(lambda r: (
                config.block_limit_up_buy and _is_close_limit_up(
                    dict(r["raw_price_row"]), r["stock_id"], price_cols
                )
            ), axis=1)
        ))
        eligible = stocks_df.copy()

        if config.position_sizing == "budget":
            if config.tier1_only:
                eligible = eligible[eligible["bin"] == 3]
            budget_plans, _ = _allocate_by_budget(eligible, deployable, config,
                cash_ratio_override=(cash_ratio_override if cash_ratio_override is not None else config.cash_ratio))
            # Industry cap: drop lowest-scored stocks from over-cap industries
            if config.industry_cap > 0 and industry_map is not None:
                cap_amount = deployable * (
                    cash_ratio_override if cash_ratio_override is not None else config.cash_ratio
                ) * config.industry_cap
                ind_spent: dict[str, float] = {}
                kept = []
                for plan in sorted(budget_plans, key=lambda x: x["score"], reverse=True):
                    ind = industry_map.get(plan["stock_id"], "UNKNOWN")
                    current = ind_spent.get(ind, 0.0)
                    if current + plan["entry_gross"] > cap_amount + 1e-9:
                        continue
                    ind_spent[ind] = current + plan["entry_gross"]
                    kept.append(plan)
                budget_plans = kept
            for plan in budget_plans:
                buy_targets[plan["stock_id"]] = {
                    "lots": plan["n_lots"], "shares": plan["shares"],
                    "close": plan["close"],
                    "score": plan["score"],
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
                slip_bps_fixed = config.buy_slippage_bps.get(bi, 0)
                slip_rate_fixed = slip_bps_fixed / 10000.0
                fill_fixed = float(stock["close"]) * (1.0 + slip_rate_fixed)
                buy_targets[sid] = {
                    "lots": n_lots, "shares": n_lots * config.lot_size,
                    "close": fill_fixed,
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
    _price_lookup: dict | None = None,
    _price_cols: set[str] | None = None,
    industry_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Account-based real backtest: T signal → T close buy → T+1 close sell.

    Returns
    -------
    dict with:
      daily_returns, daily_nav, daily_equity, daily_cash, daily_market_value,
      daily_n_positions, daily_n_buys, daily_n_sells, daily_capital_used,
      daily_filtered, daily_blocked_sells, trade_log, position_snapshots, summary.
    """
    if config is None:
        config = RealBacktestConfig()

    # ── Regime defence ──
    regime_active: dict[pd.Timestamp, dict[str, float]] = {}
    if config.regime_csv_path and config.regime_defense_score is not None:
        reg = pd.read_csv(config.regime_csv_path, parse_dates=["date"])
        required_regime_cols = {"date", "bull", "regime_score"}
        missing_regime_cols = required_regime_cols - set(reg.columns)
        if missing_regime_cols:
            raise ValueError(f"regime CSV missing columns: {missing_regime_cols}")
        reg = reg.sort_values("date").drop_duplicates("date", keep="last")
        reg["score_diff"] = reg["regime_score"].diff()
        for _, row in reg.iterrows():
            ts = pd.Timestamp(row["date"])
            regime_active[ts] = {
                "bull": float(row["bull"]),
                "score": float(row["regime_score"]),
                "falling": bool(float(row["score_diff"]) < -0.01) if pd.notna(row.get("score_diff")) else False,
            }

    defence_active = False
    recovery_left = 0
    cash_ratio_current = config.cash_ratio

    # Inherit regime state from before start_date so a defence triggered on
    # 7/17 is still active (or recovering) when the backtest begins on 7/20.
    if regime_active and config.regime_defense_score is not None:
        warmup_dates = sorted(regime_active)
        if config.start_date is not None:
            warmup_cutoff = pd.Timestamp(config.start_date)
        else:
            signal_times = pd.to_datetime(pred_df[date_col]).dropna()
            warmup_cutoff = (
                pd.Timestamp(signal_times.min()) if len(signal_times) else None
            )
        if warmup_cutoff is None:
            raise ValueError(
                "Regime defence is enabled but no signal dates are available. "
                "Pass a non-empty pred_df or an explicit start_date."
            )
        warmup_dates = [d for d in warmup_dates if d <= warmup_cutoff]
        if not warmup_dates:
            raise ValueError(
                "Regime defence is enabled but the regime CSV has no dates on or "
                f"before {warmup_cutoff.date()}. Extend the regime history or "
                "raise start_date."
            )
        for d in warmup_dates:
            reg = regime_active.get(d)
            if reg is None:
                continue
            danger = (
                reg["bull"] == 0.0
                and reg["score"] < config.regime_defense_score
                and reg["falling"]
            )
            if danger:
                defence_active = True
                recovery_left = 0
                cash_ratio_current = config.regime_danger_ratio
            else:
                if defence_active:
                    defence_active = False
                    recovery_left = config.regime_recovery_steps
                if recovery_left > 0:
                    completed = config.regime_recovery_steps - recovery_left + 1
                    cash_ratio_current = (
                        config.regime_danger_ratio
                        + (completed / config.regime_recovery_steps)
                        * (config.cash_ratio - config.regime_danger_ratio)
                    )
                    recovery_left -= 1
                else:
                    cash_ratio_current = config.cash_ratio

    # ── Core engine ──
    pred = _clean_pred_frame(pred_df, date_col, stock_col, config.pred_col)

    if _price_lookup is not None and _price_cols is not None:
        price_lookup = {
            k: v for k, v in _price_lookup.items()
            if config.start_date is None or k >= pd.Timestamp(config.start_date)
        }
        price_cols = _price_cols
        price_dates = sorted(price_lookup.keys())
        if config.start_date is not None:
            price_dates = [d for d in price_dates if d >= pd.Timestamp(config.start_date)]
    else:
        prices, price_cols = _clean_price_frame(price_df, date_col, stock_col, config)
        price_dates = [pd.Timestamp(d) for d in np.sort(prices[date_col].dropna().unique())]
        if config.start_date is not None:
            start = pd.Timestamp(config.start_date).normalize()
            price_dates = [d for d in price_dates if d >= start]
            pred = pred[pred[date_col] >= start].copy()
        # Price lookup: {date: {stock_id: row_dict}}
        price_lookup = {}
        for d, grp in prices.groupby(date_col, sort=True):
            dd = pd.Timestamp(d)
            grp = grp.loc[:, ~grp.columns.duplicated()].copy()
            price_lookup[dd] = grp.set_index(stock_col).to_dict("index")
    if not price_dates:
        raise ValueError("price_df contains no trading dates on or after start_date")
    date_to_idx = {d: i for i, d in enumerate(price_dates)}

    # Map each signal date T to entry date T (post-market close) and target exit T+1.
    entry_signals: dict[pd.Timestamp, list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]] = {}
    skipped_signal_dates = 0
    for signal_date, day_pred in pred.groupby(date_col, sort=True):
        signal_date = pd.Timestamp(signal_date)
        idx = date_to_idx.get(signal_date)
        if idx is None:
            skipped_signal_dates += 1
            continue
        entry_date = price_dates[idx]       # T: post-market close-price execution
        # The latest signal must still create today's actual holding.  Its T+1
        # date is unknown until the next market update, so keep it open with
        # NaT; the deterministic replay will map it to the real next trading
        # date as soon as that date exists.
        target_exit_date = price_dates[idx + 1] if idx + 1 < len(price_dates) else pd.NaT
        entry_signals.setdefault(entry_date, []).append((signal_date, target_exit_date, day_pred.copy()))

    if not entry_signals:
        raise ValueError("No valid signal dates can be mapped to T/T+1 trading dates")

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
    turnover_by_date: dict[pd.Timestamp, float] = {}
    cash_ratio_by_date: dict[pd.Timestamp, float] = {}
    turnover_deferred_by_date: dict[pd.Timestamp, int] = {}

    trade_log: list[dict[str, Any]] = []
    position_snapshots: list[dict[str, Any]] = []

    last_entry_date = max(entry_signals) if entry_signals else None

    for current_date in price_dates[first_entry_idx:]:
        # Stop early: no more signals and no open positions
        if last_entry_date is not None and current_date > last_entry_date and len(positions) == 0:
            break

        # ── Regime defence per-day ──
        if regime_active and config.regime_defense_score is not None:
            reg = regime_active.get(current_date)
            danger_today = (
                reg is not None
                and reg["bull"] == 0.0
                and reg["score"] < config.regime_defense_score
                and reg["falling"]
            )
            if danger_today:
                if not defence_active:
                    defence_active = True
                recovery_left = 0
                cash_ratio_current = config.regime_danger_ratio
            else:
                if defence_active:
                    defence_active = False
                    recovery_left = config.regime_recovery_steps
                if recovery_left > 0:
                    completed = config.regime_recovery_steps - recovery_left + 1
                    t = completed / config.regime_recovery_steps
                    cash_ratio_current = (
                        config.regime_danger_ratio
                        + t * (config.cash_ratio - config.regime_danger_ratio)
                    )
                    recovery_left -= 1
                else:
                    cash_ratio_current = config.cash_ratio

        today_px = price_lookup.get(current_date, {})
        pre_trade_mv = 0.0
        for pos in positions:
            row = today_px.get(pos["stock_id"])
            if row is not None and _is_finite_positive(row.get("close")):
                mark = float(row["close"])
            else:
                mark = float(pos.get("last_price", pos["entry_open"]))
            pre_trade_mv += pos["shares"] * mark
        pre_trade_equity = cash + pre_trade_mv
        turnover_limit = (
            np.inf
            if config.max_daily_turnover is None
            else config.max_daily_turnover * pre_trade_equity
        )
        gross_buy_today = 0.0
        gross_sell_today = 0.0

        n_buys = 0
        n_sells = 0
        n_filtered = 0
        n_blocked_sells = 0
        n_turnover_deferred = 0
        capital_used_today = 0.0

        # ------------------------------------------------------------------
        # 1) Build today's buy targets.
        # ------------------------------------------------------------------
        buy_targets, n_filtered_today = _build_buy_targets(
            current_date, entry_signals, today_px, positions, cash,
            config, stock_col, price_cols, cash_ratio_override=cash_ratio_current,
            industry_map=industry_map,
        )
        n_filtered += n_filtered_today

        # ------------------------------------------------------------------
        # 2) Smart-hold: sell only delta, not full position, when stock stays in buy list.
        # ------------------------------------------------------------------
        held_stock_ids = {p["stock_id"] for p in positions}
        still_holding: list[dict[str, Any]] = []

        # Forced exits with the weakest scores are processed first.  Surviving
        # names and new buys are then handled from higher to lower target score.
        def _holding_priority(pos: dict[str, Any]) -> tuple[int, float]:
            sid_ = pos["stock_id"]
            target_ = buy_targets.get(sid_)
            if target_ is None:
                return (0, float(pos.get("score", -np.inf)))
            delta_ = target_["lots"] - pos["lots"]
            if delta_ < 0:
                return (1, float(target_["score"]))
            if delta_ > 0:
                return (2, -float(target_["score"]))
            return (3, 0.0)

        for pos in sorted(positions, key=_holding_priority):
            sid = pos["stock_id"]
            target_exit = pos["target_exit_date"]
            due_to_sell = pd.notna(target_exit) and current_date >= target_exit
            row = today_px.get(sid)

            if row is not None and _is_finite_positive(row.get("close")):
                pos["last_price"] = float(row["close"])
                pos["last_price_date"] = current_date

            if not due_to_sell:
                still_holding.append(pos)
                continue

            target = buy_targets.get(sid)
            if target is not None:
                # ── Smart hold: stock still in buy list, only trade delta ──
                delta_lots = _lambda_adjusted_lots(
                    target["lots"] - pos["lots"],
                    config.lambda_weight,
                )
                raw_px = float(row["close"]) if row is not None and _is_finite_positive(row.get("close")) else pos.get("last_price", pos["entry_open"])

                if delta_lots > 0:
                    # Need more: buy delta lots (buy slip applied)
                    buy_slip = config.buy_slippage_bps.get(target.get("bin", 2), 10) / 10000.0
                    px = raw_px * (1.0 + buy_slip)
                    # Need more: buy delta lots at buy commission (万3)
                    remaining_turnover = turnover_limit - gross_buy_today - gross_sell_today
                    delta_lots = _lots_within_turnover(
                        delta_lots,
                        px * config.lot_size,
                        remaining_turnover,
                    )
                    if delta_lots == 0:
                        n_turnover_deferred += 1
                    delta_shares = delta_lots * config.lot_size
                    delta_gross = delta_shares * px
                    delta_fee = _commission(delta_gross, config)
                    if delta_lots > 0 and delta_gross + delta_fee <= cash + 1e-9:
                        cash -= (delta_gross + delta_fee)
                        pos["shares"] += delta_shares
                        pos["lots"] += delta_lots
                        pos["entry_gross"] += delta_gross
                        pos["buy_fee"] += delta_fee
                        pos["entry_open"] = (pos["entry_open"] * (pos["lots"] - delta_lots) + px * delta_lots) / pos["lots"] if pos["lots"] > 0 else px
                        pos["last_buy_date"] = current_date
                        n_buys += 1
                        capital_used_today += delta_gross
                        gross_buy_today += delta_gross
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
                    # Need fewer: sell excess (sell slip applied)
                    can_reduce = row is not None and _is_finite_positive(row.get("close"))
                    if can_reduce and config.skip_suspended and _row_is_suspended(row, price_cols):
                        can_reduce = False
                    if can_reduce and config.block_limit_down_sell and _is_close_limit_down(row, sid, price_cols):
                        can_reduce = False

                    if not can_reduce:
                        pos["delayed_exit_days"] = int(pos.get("delayed_exit_days", 0)) + 1
                        n_blocked_sells += 1
                    else:
                        sell_slip = config.sell_slippage_bps / 10000.0
                        px = raw_px * (1.0 - sell_slip)
                        remaining_turnover = turnover_limit - gross_buy_today - gross_sell_today
                        excess_lots = _lots_within_turnover(
                            -delta_lots,
                            px * config.lot_size,
                            remaining_turnover,
                        )
                        if excess_lots == 0:
                            n_turnover_deferred += 1
                            can_reduce = False
                            still_holding.append(pos)
                            del buy_targets[sid]
                            continue
                        excess_shares = excess_lots * config.lot_size
                        gross_sell = excess_shares * px
                        sell_fee = _commission(gross_sell, config) + gross_sell * config.stamp_tax_rate
                        old_shares = pos["shares"]
                        fraction = excess_shares / old_shares
                        allocated_entry = pos["entry_gross"] * fraction
                        allocated_buy_fee = pos["buy_fee"] * fraction
                        cash += gross_sell - sell_fee
                        pos["shares"] -= excess_shares
                        pos["lots"] -= excess_lots
                        pos["entry_gross"] -= allocated_entry
                        pos["buy_fee"] -= allocated_buy_fee
                        n_sells += 1
                        gross_sell_today += gross_sell
                        trade_log.append({
                            "date": current_date, "action": "SELL",
                            "position_id": pos["position_id"], "stock_id": sid,
                            "shares": excess_shares, "lots": excess_lots,
                            "price": px, "gross": gross_sell, "fee": sell_fee,
                            "cash_after": cash,
                            "signal_date": pos["signal_date"], "entry_date": pos["entry_date"],
                            "target_exit_date": pos["target_exit_date"],
                            "delayed_exit_days": pos.get("delayed_exit_days", 0),
                            "realized_pnl": gross_sell - allocated_entry - allocated_buy_fee - sell_fee,
                            "return_on_gross": (
                                (gross_sell - allocated_entry - allocated_buy_fee - sell_fee) / allocated_entry
                                if allocated_entry > 0 else np.nan
                            ),
                        })

                # Extend exit date
                pos["target_exit_date"] = target["target_exit_date"]
                pos["signal_date"] = target["signal_date"]
                pos["score"] = target["score"]
                pos["rank_pct"] = target["rank_pct"]
                pos["bin"] = target["bin"]
                if delta_lots >= 0 or can_reduce:
                    pos["delayed_exit_days"] = 0
                still_holding.append(pos)
                del buy_targets[sid]
                continue

            # ── Stock NOT in buy list: full sell ──
            can_sell = row is not None and _is_finite_positive(row.get("close"))
            if can_sell and config.skip_suspended and _row_is_suspended(row, price_cols):
                can_sell = False
            if can_sell and config.block_limit_down_sell and _is_close_limit_down(row, sid, price_cols):
                can_sell = False

            if not can_sell:
                pos["delayed_exit_days"] = int(pos.get("delayed_exit_days", 0)) + 1
                still_holding.append(pos)
                n_blocked_sells += 1
                continue

            sell_open = float(row["close"]) * (1.0 - config.sell_slippage_bps / 10000.0)
            gross_sell = pos["shares"] * sell_open
            remaining_turnover = turnover_limit - gross_buy_today - gross_sell_today
            if gross_sell > remaining_turnover + 1e-9:
                # Turnover is a voluntary execution constraint, not a market
                # blockage.  Keep the position and retry on the next day.
                still_holding.append(pos)
                n_turnover_deferred += 1
                continue
            sell_commission = _commission(gross_sell, config)
            stamp_tax = gross_sell * config.stamp_tax_rate
            sell_fee = sell_commission + stamp_tax
            cash += gross_sell - sell_fee

            realized_pnl = gross_sell - pos["entry_gross"] - pos["buy_fee"] - sell_fee
            n_sells += 1
            gross_sell_today += gross_sell
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
        max_slots = max(0, config.max_stocks - len(still_holding))
        new_plans = sorted(
            [(sid, t) for sid, t in buy_targets.items() if sid not in held_stock_ids],
            key=lambda x: x[1]["score"], reverse=True,
        )
        n_new = 0
        for sid, target in new_plans:
            if n_new >= max_slots:
                break

            entry_close = target["close"]
            entry_gross = target.get("entry_gross", target["shares"] * entry_close)
            remaining_turnover = turnover_limit - gross_buy_today - gross_sell_today
            actual_lots = _lots_within_turnover(
                int(target["lots"]),
                float(entry_close) * config.lot_size,
                remaining_turnover,
            )
            if actual_lots == 0:
                n_turnover_deferred += 1
                continue
            entry_gross = actual_lots * config.lot_size * float(entry_close)
            buy_fee = _commission(entry_gross, config)
            cash_needed = entry_gross + buy_fee
            if cash_needed > cash + 1e-9:
                continue

            cash -= cash_needed
            position = {
                "position_id": next_position_id,
                "stock_id": sid,
                "shares": actual_lots * config.lot_size,
                "lots": actual_lots,
                "entry_open": target["close"],
                "entry_gross": entry_gross,
                "buy_fee": buy_fee,
                "signal_date": target["signal_date"],
                "entry_date": current_date,
                "last_buy_date": current_date,
                "target_exit_date": target["target_exit_date"],
                "score": target["score"],
                "rank_pct": target["rank_pct"],
                "bin": target["bin"],
                "last_price": target["close"],
                "last_price_date": current_date,
                "delayed_exit_days": 0,
            }
            positions.append(position)
            held_stock_ids.add(sid)
            n_buys += 1
            n_new += 1
            capital_used_today += entry_gross
            gross_buy_today += entry_gross

            trade_log.append({
                "date": current_date, "action": "BUY",
                "position_id": next_position_id, "stock_id": sid,
                "shares": actual_lots * config.lot_size, "lots": actual_lots,
                "price": entry_close, "gross": entry_gross, "fee": buy_fee,
                "cash_after": cash,
                "signal_date": target["signal_date"], "entry_date": current_date,
                "target_exit_date": target["target_exit_date"],
                "bin": target["bin"], "rank_pct": target["rank_pct"],
                "score": target["score"],
            })
            next_position_id += 1

        # ------------------------------------------------------------------
        # 3) Mark account equity at current close after sell/buy processing.
        # ------------------------------------------------------------------
        market_value = 0.0
        for pos in positions:
            sid = pos["stock_id"]
            row = today_px.get(sid)
            if row is not None and _is_finite_positive(row.get("close")):
                mark_price = float(row["close"])
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
                "last_buy_date": pos.get("last_buy_date", pos["entry_date"]),
                "target_exit_date": pos["target_exit_date"],
                "last_price": mark_price,
                "market_value": pos["shares"] * mark_price,
                "avg_cost": ((pos["entry_gross"] + pos["buy_fee"]) / pos["shares"]
                             if pos["shares"] > 0 else np.nan),
                "unrealized_pnl": (pos["shares"] * mark_price
                                   - pos["entry_gross"] - pos["buy_fee"]),
                "score": pos.get("score"),
                "rank_pct": pos.get("rank_pct"),
                "selected_today": pos.get("signal_date") == current_date,
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
        turnover_by_date[current_date] = (
            (gross_buy_today + gross_sell_today) / (2.0 * pre_trade_equity)
            if pre_trade_equity > 0 else 0.0
        )
        cash_ratio_by_date[current_date] = cash_ratio_current
        turnover_deferred_by_date[current_date] = n_turnover_deferred

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
    daily_turnover = pd.Series(turnover_by_date).sort_index()
    daily_turnover.name = "turnover"
    daily_turnover_deferred = pd.Series(turnover_deferred_by_date).sort_index()
    daily_turnover_deferred.name = "turnover_deferred"

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
    summary["mean_turnover"] = float(daily_turnover.mean())
    summary["max_turnover"] = float(daily_turnover.max())
    summary["mean_turnover_deferred"] = float(daily_turnover_deferred.mean())
    summary["total_turnover_deferred"] = int(daily_turnover_deferred.sum())
    summary["skipped_signal_dates"] = int(skipped_signal_dates)
    summary["open_positions_at_end"] = int(len(positions))

    trade_log_df = pd.DataFrame(trade_log)
    if not trade_log_df.empty:
        trade_log_df = trade_log_df.sort_values(["date", "position_id", "action"]).reset_index(drop=True)

    position_snapshots_df = pd.DataFrame(position_snapshots)
    if not position_snapshots_df.empty:
        position_snapshots_df = position_snapshots_df.sort_values(["date", "position_id"]).reset_index(drop=True)
        totals = position_snapshots_df.groupby("date")["market_value"].transform("sum")
        position_snapshots_df["weight"] = np.where(
            totals > 0, position_snapshots_df["market_value"] / totals, 0.0
        )

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
        "daily_turnover": daily_turnover,
        "daily_turnover_deferred": daily_turnover_deferred,
        "daily_cash_ratio": pd.Series(cash_ratio_by_date).sort_index(),
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

    # Minimal columns for close-close execution.
    price_cols = ["time", "stock_id", "close", "pre_close"]
    panel_path = Path("dataset/processed/unified_daily_panel.parquet")
    prices = pd.read_parquet(panel_path, columns=price_cols)
    prices = normalize_keys(prices)
    prices["time"] = pd.to_datetime(prices["time"])
    print(f"  prices: {len(prices):,} rows, {prices['time'].nunique()} dates")

    config = RealBacktestConfig(
        capital=200_000,
        position_sizing="budget",
        cash_ratio=0.98,
        max_stocks=50,
        commission_rate=0.0003,
        stamp_tax_rate=0.0005,
        min_commission=0.0,
        buy_slippage_bps={1: 3, 2: 5, 3: 8},
        sell_slippage_bps=3,
    )
    result = run_real_backtest(scores, prices, config)
    s = result["summary"]

    print("\n=== Account Real Backtest (T signal → T close buy → T+1 close sell) ===")
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
