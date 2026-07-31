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


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay and save the live EW-rank-10d account")
    parser.add_argument("--start", default="2026-07-20")
    parser.add_argument("--capital", type=float, default=200_000.0)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--smooth-window", type=int, default=10)
    parser.add_argument("--exp-dir", type=Path, default=Path("dataset/output/experiment_009"))
    parser.add_argument("--price-path", type=Path, default=Path("dataset/processed/unified_daily_panel.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/strategy_v1/evidence"))
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--min-commission", type=float, default=5.0)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--buy-slippage-bps", type=float, nargs=3, metavar=("TIER1", "TIER2", "TIER3"),
                        default=(0.0, 0.0, 0.0),
                        help="Three rank-tier buy slippages in bps (default: 0 0 0, post-close trading)")
    parser.add_argument("--sell-slippage-bps", type=float, default=0.0)
    parser.add_argument("--cash-ratio", type=float, default=0.98)
    parser.add_argument("--strategy", type=str, default="baseline",
                        choices=("baseline", "defend", "elite"),
                        help="baseline=buf50 (default); defend=buf120+Bear20%%; elite=tier1-only")
    parser.add_argument("--buffer-exit-n", type=int, default=None)
    parser.add_argument("--buffer-mode", type=str, default=None)
    parser.add_argument("--tier1-only", action="store_true", default=None)
    parser.add_argument("--regime-csv", type=Path, default=Path("dataset/input/market_regime.csv"))
    parser.add_argument("--regime-defense-score", type=float, default=None)
    parser.add_argument("--regime-danger-ratio", type=float, default=None)
    parser.add_argument("--regime-recovery-steps", type=int, default=None)
    args = parser.parse_args()

    # Strategy presets
    if args.strategy == "elite":
        buffer_exit_n = args.buffer_exit_n or 50
        buffer_mode = args.buffer_mode or "fixed"
        tier1_only = True if args.tier1_only is None else args.tier1_only
        defence_score = args.regime_defense_score  # None
        danger_ratio = args.regime_danger_ratio or 0.80
        recovery_steps = args.regime_recovery_steps or 3
    elif args.strategy == "defend":
        buffer_exit_n = args.buffer_exit_n or 120
        buffer_mode = args.buffer_mode or "fixed"
        tier1_only = bool(args.tier1_only)
        defence_score = args.regime_defense_score if args.regime_defense_score is not None else -0.35
        danger_ratio = args.regime_danger_ratio if args.regime_danger_ratio is not None else 0.20
        recovery_steps = args.regime_recovery_steps if args.regime_recovery_steps is not None else 3
    else:  # baseline
        buffer_exit_n = args.buffer_exit_n or 50
        buffer_mode = args.buffer_mode or "fixed"
        tier1_only = bool(args.tier1_only)
        defence_score = args.regime_defense_score  # None = disabled
        danger_ratio = args.regime_danger_ratio or 0.80
        recovery_steps = args.regime_recovery_steps or 3

    result = build_ledger(
        args.exp_dir, args.price_path, args.output_dir, pd.Timestamp(args.start),
        args.capital, args.top_n, args.smooth_window, args.commission_rate,
        args.min_commission, args.stamp_tax_rate,
        tuple(args.buy_slippage_bps), args.sell_slippage_bps, args.cash_ratio,
        buffer_exit_n=buffer_exit_n,
        buffer_mode=buffer_mode,
        tier1_only=tier1_only,
        regime_csv_path=str(args.regime_csv) if defence_score is not None else "",
        regime_defense_score=defence_score,
        regime_danger_ratio=danger_ratio,
        regime_recovery_steps=recovery_steps,
    )
    latest = result["daily_account"].iloc[-1]
    print(f"Saved EW10d account through {result['daily_account'].index[-1].date()}: "
          f"NAV={latest['nav']:.6f}, equity={latest['equity']:,.2f}, "
          f"positions={int(latest['n_positions'])}, friction={result['summary']['total_friction_cost']:,.2f}")


if __name__ == "__main__":
    main()
