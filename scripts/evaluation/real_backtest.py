"""
Standalone real backtest — reads saved Bayes scores + price panel, no re-computation.

Usage:
    python -m scripts.evaluation.real_backtest
    python -m scripts.evaluation.real_backtest --capital 150000 --max-stocks 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.backtest.real_backtest import RealBacktestConfig, run_real_backtest
from src.backtest.ensemble_utils import normalize_keys
from src.backtest.portfolio import PortfolioConfig
from src.backtest.engine import TransactionCostConfig, run_backtest
from src.experiment.returns import align_predictions_to_returns

OUTPUT_DIR = Path("reports/strategy_v1/evidence")


def load_scores(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = normalize_keys(df)
    return df


def load_prices(path: str) -> pd.DataFrame:
    # close+pre_close for close-close execution. Add optional columns if available.
    base_cols = ["time", "stock_id", "close", "pre_close"]
    optional_cols = [
        "up_limit", "down_limit", "high_limit", "low_limit",
        "limit_up", "limit_down", "limit_up_price", "limit_down_price",
        "is_st", "is_paused", "st", "suspend", "suspended",
        "is_suspended", "trade_status", "risk_warning",
    ]
    import pyarrow.parquet as pq
    panel_cols = pq.read_schema(path).names
    load_cols = base_cols + [c for c in optional_cols if c in panel_cols]
    df = pd.read_parquet(path, columns=load_cols)
    df = normalize_keys(df)
    df["time"] = pd.to_datetime(df["time"])
    # Coerce price columns
    for c in ["close", "pre_close"] + [x for x in optional_cols if x in df.columns and x not in ("is_st", "is_paused", "st", "suspend", "suspended", "is_suspended", "trade_status", "risk_warning")]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_returns(path: str) -> pd.DataFrame:
    """Load close-to-close returns for paper comparison."""
    panel = pd.read_parquet(path, columns=["time", "stock_id", "ret_daily"])
    panel = normalize_keys(panel)
    ret = panel.rename(columns={"ret_daily": "return_1d"}).copy()
    ret["return_1d"] = pd.to_numeric(ret["return_1d"], errors="coerce")
    return ret.dropna(subset=["return_1d"]).sort_values(["time", "stock_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=200_000)
    parser.add_argument("--max-stocks", type=int, default=50)
    parser.add_argument("--position-sizing", type=str, default="fixed",
                        choices=["fixed", "budget"])
    parser.add_argument("--cash-ratio", type=float, default=0.80)
    parser.add_argument("--bin-lots", type=str, default="0,1,2",
                        help="comma-separated: bin1_lots,bin2_lots,bin3_lots (fixed mode only)")
    parser.add_argument("--scores-path", type=str,
                        default="reports/strategy_v1/evidence/bayes_scores_test.parquet")
    parser.add_argument("--price-path", type=str,
                        default="dataset/processed/unified_daily_panel.parquet")
    parser.add_argument("--paper", action="store_true",
                        help="also run paper backtest for comparison")
    args = parser.parse_args()

    bin_lots = tuple(int(x) for x in args.bin_lots.split(","))
    if len(bin_lots) != 3:
        raise ValueError("--bin-lots must have exactly 3 values")

    # --- Load ---
    print("Loading data...")
    scores = load_scores(args.scores_path)
    print(f"  scores: {len(scores):,} rows, {scores['time'].nunique()} dates")

    prices = load_prices(args.price_path)
    price_cols_loaded = [c for c in prices.columns if c not in ("time", "stock_id")]
    print(f"  prices: {len(prices):,} rows, {prices['time'].nunique()} dates")
    # Report optional columns found
    extras = [c for c in prices.columns if c not in ("time", "stock_id", "close", "pre_close")]
    if extras:
        print(f"  extra cols: {extras}")

    # --- Real backtest ---
    config = RealBacktestConfig(
        capital=args.capital,
        bin_lots=bin_lots,
        max_stocks=args.max_stocks,
        position_sizing=args.position_sizing,
        cash_ratio=args.cash_ratio,
        commission_rate=0.0003,
        stamp_tax_rate=0.0005,
        min_commission=0.0,
    )

    sizing_label = f"sizing={args.position_sizing}" if args.position_sizing == "budget" else f"lots={bin_lots}"
    print(f"\nRunning real backtest (capital={args.capital:,.0f}, max={args.max_stocks}, {sizing_label})...")
    result = run_real_backtest(scores, prices, config)
    s = result["summary"]

    print(f"\n=== Real Backtest (T signal → T close buy → T+1 close sell) ===")
    print(f"  Sharpe:             {s['sharpe_ratio']:.4f}")
    print(f"  NAV:                {s['final_nav']:.4f}")
    print(f"  Final equity:       {s['final_equity']:,.0f} CNY")
    print(f"  Total PnL:          {s['total_pnl']:,.0f} CNY")
    print(f"  MaxDD:              {s['max_drawdown']:.1%}")
    print(f"  AnnualRet:          {s['annualized_return']:.1%}")
    print(f"  Mean positions:     {s['mean_n_positions']:.1f}  (max {s['max_n_positions']})")
    print(f"  Mean buys/day:      {s['mean_n_buys']:.1f}")
    print(f"  Mean sells/day:     {s['mean_n_sells']:.1f}")
    print(f"  Mean cap used:      {s['mean_capital_used']:,.0f} CNY ({s['mean_capital_used_pct']:.1%})")
    print(f"  Mean market value:  {s['mean_market_value']:,.0f} CNY ({s['mean_market_value_pct']:.1%})")
    print(f"  Mean cash:          {s['mean_cash']:,.0f} CNY ({s['mean_cash_pct']:.1%})")
    print(f"  Filtered buys:      {s['mean_filtered']:.1f}/day ({s['total_filtered']} total)")
    print(f"  Blocked sells:      {s['mean_blocked_sells']:.1f}/day ({s['total_blocked_sells']} total)")
    print(f"  Open positions end: {s['open_positions_at_end']}")

    # --- Save artifacts ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"cap{int(args.capital//1000)}k_s{args.max_stocks}_l{bin_lots[0]}{bin_lots[1]}{bin_lots[2]}"

    # Daily returns
    ret_out = OUTPUT_DIR / f"real_returns_{tag}.parquet"
    result["daily_returns"].to_frame().to_parquet(ret_out)

    # Daily positions (entry-side summary)
    pos_out = OUTPUT_DIR / f"real_positions_{tag}.parquet"
    result["daily_positions"].to_parquet(pos_out, index=False)

    # Trade log (every buy/sell with realized PnL)
    trade_out = OUTPUT_DIR / f"real_trades_{tag}.parquet"
    result["trade_log"].to_parquet(trade_out, index=False)

    # Position snapshots (all open positions at each mark date)
    snap_out = OUTPUT_DIR / f"real_snapshots_{tag}.parquet"
    result["position_snapshots"].to_parquet(snap_out, index=False)

    print(f"\nSaved: {ret_out}, {pos_out}, {trade_out}, {snap_out}")

    # --- Paper comparison (optional) ---
    if args.paper:
        print("\n=== Paper comparison (close-to-close, bin_weighted %-wt) ===")
        returns = load_returns(args.price_path)
        aligned = scores.rename(columns={"bayes_score": "y_pred"})
        aligned2 = align_predictions_to_returns(
            pred_df=aligned, returns_df=returns,
            date_col="time", stock_col="stock_id",
            pred_col="y_pred", return_col="return_1d",
        )
        pfolio = PortfolioConfig(strategy="bin_weighted", top_n=args.max_stocks,
                                 pred_col="y_pred", stock_col="stock_id")
        paper_result = run_backtest(
            pred_df=aligned2, returns_df=returns,
            portfolio_config=pfolio,
            return_col="return_1d", date_col="time", stock_col="stock_id",
            cost_config=TransactionCostConfig(),
        )
        ps = paper_result["summary"]
        print(f"  Paper Sharpe:  {ps['sharpe_ratio']:.4f}")
        print(f"  Paper NAV:     {ps['final_nav']:.4f}")
        print(f"  Paper MaxDD:   {ps['max_drawdown']:.1%}")
        print(f"  Paper Turnover:{ps['mean_turnover']:.1%}")
        print(f"  Real Sharpe:   {s['sharpe_ratio']:.4f}")
        print(f"  Real NAV:      {s['final_nav']:.4f}")
        print(f"  Real MaxDD:    {s['max_drawdown']:.1%}")
        print(f"  Gap:           {ps['sharpe_ratio'] - s['sharpe_ratio']:.4f} Sharpe")


if __name__ == "__main__":
    main()
