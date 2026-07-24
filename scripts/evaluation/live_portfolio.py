"""Thin CLI wrapper — see :func:`src.backtest.live_account.build_ledger`.

Usage: python -m scripts.evaluation.live_portfolio
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.live_account import build_ledger
from src.experiment.config import ExperimentConfig


def main() -> None:
    default_cfg = ExperimentConfig()
    parser = argparse.ArgumentParser(description="Replay and save the live EW-rank-10d account")
    parser.add_argument("--start", default="2026-07-20")
    parser.add_argument("--capital", type=float, default=200_000.0)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--smooth-window", type=int, default=10)
    parser.add_argument("--exp-dir", type=Path, default=Path(default_cfg.output_dir))
    parser.add_argument("--price-path", type=Path, default=Path("dataset/processed/unified_daily_panel.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/strategy_v1/evidence"))
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--min-commission", type=float, default=5.0)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--buy-slippage-bps", type=float, nargs=3, metavar=("TIER1", "TIER2", "TIER3"),
                        default=(0.0, 0.0, 0.0),
                        help="Three rank-tier buy slippages in bps (default: 0 0 0, post-close trading)")
    parser.add_argument("--sell-slippage-bps", type=float, default=0.0)
    parser.add_argument("--cash-ratio", type=float, default=0.80)
    args = parser.parse_args()
    result = build_ledger(
        args.exp_dir, args.price_path, args.output_dir, pd.Timestamp(args.start),
        args.capital, args.top_n, args.smooth_window, args.commission_rate,
        args.min_commission, args.stamp_tax_rate,
        tuple(args.buy_slippage_bps), args.sell_slippage_bps, args.cash_ratio,
    )
    latest = result["daily_account"].iloc[-1]
    print(f"Saved EW10d account through {result['daily_account'].index[-1].date()}: "
          f"NAV={latest['nav']:.6f}, equity={latest['equity']:,.2f}, "
          f"positions={int(latest['n_positions'])}, friction={result['summary']['total_friction_cost']:,.2f}")


if __name__ == "__main__":
    main()
