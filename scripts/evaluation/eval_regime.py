"""
Regime evaluation — Layer 1 (conditional forward statistics) + Layer 2 (stability).

Evaluates regime labels produced by scripts.strategy.market_regime.  All
statistics use only real-time-available information: the regime label at close
of t and forward returns / volatility starting at t+1.

Usage:
    python -m scripts.evaluation.eval_regime
    python -m scripts.evaluation.eval_regime --csv dataset/input/market_regime.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

FORWARD_HORIZONS = (1, 5, 10, 20, 60)
DIRECTION_REGIMES = ("Bull", "Bear")
VOLATILITY_REGIMES = ("HighVol", "LowVol")
ALL_REGIMES = ("Bull", "Bear", "HighVol", "LowVol")


def _load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    return df


def _forward_returns(
    daily_ret: pd.Series, horizon: int
) -> pd.Series:
    """Causal forward N-day log return.  t+N value known at close of t+N."""
    log_ret = np.log1p(daily_ret.clip(lower=-0.999))
    fwd = log_ret.rolling(horizon).sum().shift(-horizon)
    return fwd.rename(f"fwd_{horizon}")


def _forward_volatility(
    daily_ret: pd.Series, horizon: int
) -> pd.Series:
    """Annualised forward N-day volatility (t+1 … t+N)."""
    log_ret = np.log1p(daily_ret.clip(lower=-0.999))
    fwd_vol = log_ret.rolling(horizon).std().shift(-horizon) * np.sqrt(252)
    return fwd_vol.rename(f"fwd_vol_{horizon}")


# ---------------------------------------------------------------------------
# Layer 1 — regime-conditional forward statistics
# ---------------------------------------------------------------------------

def _layer1_table(
    df: pd.DataFrame, mask: pd.Series, horizons: tuple[int, ...]
) -> pd.DataFrame:
    """
    For a boolean *mask* selecting regime dates, return a DataFrame with:
        count, mean fwd return (ann.), mean fwd vol (ann.)
    for each horizon.
    """
    rows = []
    for h in horizons:
        fwd_col = f"fwd_{h}"
        vol_col = f"fwd_vol_{h}"
        r = df.loc[mask, fwd_col].dropna()
        v = df.loc[mask, vol_col].dropna()
        mean_ret = r.mean() / h * 252 if len(r) > 0 else np.nan
        mean_vol = v.mean() if len(v) > 0 else np.nan
        rows.append({
            "horizon_days": h,
            "n_dates": len(r),
            "ann_mean_ret": mean_ret,
            "ann_mean_vol": mean_vol,
        })
    return pd.DataFrame(rows)


def layer1_statistics(df: pd.DataFrame) -> dict:
    """Compute Layer-1 regime-conditional statistics.

    Returns a dict with keys:
        direction_table — multi-horizon return/vol for Bull vs Bear
        vol_table       — multi-horizon return/vol for HighVol vs LowVol
        t_tests         — t-test for Bull mean > Bear mean at each horizon
        consistency     — direction-consistency rate
    """
    # Drop warm-up Unknown
    valid = df[df["regime"] != "Unknown"].copy()
    if len(valid) == 0:
        raise ValueError("No non-Unknown regime dates — is warm-up complete?")

    # Forward returns & vol for each horizon
    for h in FORWARD_HORIZONS:
        valid[f"fwd_{h}"] = _forward_returns(valid["market_ret_1d"], h)
        valid[f"fwd_vol_{h}"] = _forward_volatility(valid["market_ret_1d"], h)

    # Direction table
    bull = _layer1_table(valid, valid["bull"] == 1.0, FORWARD_HORIZONS)
    bear = _layer1_table(valid, valid["bull"] == 0.0, FORWARD_HORIZONS)
    bull.insert(0, "regime", "Bull")
    bear.insert(0, "regime", "Bear")
    direction_table = pd.concat([bull, bear], ignore_index=True)

    # Volatility table
    high = _layer1_table(valid, valid["high_vol"] == 1.0, FORWARD_HORIZONS)
    low = _layer1_table(valid, valid["high_vol"] == 0.0, FORWARD_HORIZONS)
    high.insert(0, "regime", "HighVol")
    low.insert(0, "regime", "LowVol")
    vol_table = pd.concat([high, low], ignore_index=True)

    # t-tests: Bull mean return > Bear mean return? (one-sided)
    t_tests = []
    for h in FORWARD_HORIZONS:
        bull_r = valid.loc[valid["bull"] == 1.0, f"fwd_{h}"].dropna()
        bear_r = valid.loc[valid["bull"] == 0.0, f"fwd_{h}"].dropna()
        if len(bull_r) < 2 or len(bear_r) < 2:
            t_tests.append({"horizon_days": h, "t_stat": np.nan, "p_value": np.nan,
                            "bull_mean_ann": np.nan, "bear_mean_ann": np.nan,
                            "bull_vol_ann": np.nan, "bear_vol_ann": np.nan})
            continue
        bull_mean = bull_r.mean() / h * 252
        bear_mean = bear_r.mean() / h * 252
        bull_vol = bull_r.std() / np.sqrt(h) * np.sqrt(252)
        bear_vol = bear_r.std() / np.sqrt(h) * np.sqrt(252)
        t_stat, p_value = stats.ttest_ind(bull_r, bear_r, alternative="greater")
        t_tests.append({
            "horizon_days": h,
            "t_stat": t_stat,
            "p_value": p_value,
            "bull_mean_ann": bull_mean,
            "bear_mean_ann": bear_mean,
            "bull_vol_ann": bull_vol,
            "bear_vol_ann": bear_vol,
        })
    t_tests_df = pd.DataFrame(t_tests)

    # Direction consistency
    # Bull: what fraction of forward returns are positive?
    # Bear: what fraction of forward returns are negative?
    consistency_rows = []
    for h in FORWARD_HORIZONS:
        bull_fwd = valid.loc[valid["bull"] == 1.0, f"fwd_{h}"].dropna()
        bear_fwd = valid.loc[valid["bull"] == 0.0, f"fwd_{h}"].dropna()
        consistency_rows.append({
            "horizon_days": h,
            "bull_pos_rate": (bull_fwd > 0).mean() if len(bull_fwd) > 0 else np.nan,
            "bear_neg_rate": (bear_fwd < 0).mean() if len(bear_fwd) > 0 else np.nan,
        })
    consistency_df = pd.DataFrame(consistency_rows)

    return {
        "direction_table": direction_table,
        "vol_table": vol_table,
        "t_tests": t_tests_df,
        "consistency": consistency_df,
    }


# ---------------------------------------------------------------------------
# Layer 2 — stability
# ---------------------------------------------------------------------------

def layer2_statistics(df: pd.DataFrame) -> dict:
    """Stability metrics: state duration, switch frequency, asymmetry."""

    valid = df[df["regime"] != "Unknown"].copy()
    n_total = len(valid)

    def _state_runs(series: pd.Series) -> list[int]:
        """Lengths of consecutive True runs."""
        mask = series.to_numpy(dtype=bool)
        runs: list[int] = []
        run_len = 0
        for v in mask:
            if v:
                run_len += 1
            else:
                if run_len > 0:
                    runs.append(run_len)
                run_len = 0
        if run_len > 0:
            runs.append(run_len)
        return runs

    bull_runs = _state_runs(valid["bull"] == 1.0)
    bear_runs = _state_runs(valid["bull"] == 0.0)
    highvol_runs = _state_runs(valid["high_vol"] == 1.0)
    lowvol_runs = _state_runs(valid["high_vol"] == 0.0)

    # Switch counts
    bull_series = (valid["bull"] == 1.0).to_numpy(dtype=int)
    vol_series = (valid["high_vol"] == 1.0).to_numpy(dtype=int)
    dir_switches = int(np.sum(np.diff(bull_series) != 0))
    vol_switches = int(np.sum(np.diff(vol_series) != 0))

    # Asymmetry: Bull→Bear vs Bear→Bull
    bull_to_bear = int(np.sum((np.diff(bull_series) == -1)))
    bear_to_bull = int(np.sum((np.diff(bull_series) == 1)))

    # Unknown %
    unknown_pct = (df["regime"] == "Unknown").mean()

    years = n_total / 252

    rows = {
        "metric": [
            "total_dates", "unknown_pct",
            "direction_switches", "dir_switches_per_year",
            "vol_switches", "vol_switches_per_year",
            "bull_to_bear", "bear_to_bull", "switch_asymmetry_ratio",
            "bull_avg_duration_days", "bear_avg_duration_days",
            "bull_max_duration_days", "bear_max_duration_days",
            "highvol_avg_duration_days", "lowvol_avg_duration_days",
        ],
        "value": [
            n_total,
            f"{unknown_pct:.2%}",
            dir_switches,
            f"{dir_switches / years:.2f}",
            vol_switches,
            f"{vol_switches / years:.2f}",
            bull_to_bear,
            bear_to_bull,
            f"{bear_to_bull / bull_to_bear:.2f}" if bull_to_bear > 0 else "inf",
            f"{np.mean(bull_runs):.1f}" if bull_runs else "N/A",
            f"{np.mean(bear_runs):.1f}" if bear_runs else "N/A",
            max(bull_runs) if bull_runs else "N/A",
            max(bear_runs) if bear_runs else "N/A",
            f"{np.mean(highvol_runs):.1f}" if highvol_runs else "N/A",
            f"{np.mean(lowvol_runs):.1f}" if lowvol_runs else "N/A",
        ],
    }
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate market regime quality")
    parser.add_argument(
        "--csv", default="dataset/input/market_regime.csv",
        help="Path to regime CSV from scripts.strategy.market_regime",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Regime CSV not found: {csv_path}.  "
                                f"Run python -m scripts.strategy.market_regime first.")

    df = _load(str(csv_path))
    print(f"Loaded {len(df)} dates ({df['date'].min().date()} → {df['date'].max().date()})")
    print(f"  Unknown: {(df['regime']=='Unknown').sum()}  "
          f"Valid: {(df['regime']!='Unknown').sum()}\n")

    # ── Layer 2: Stability ──
    l2 = layer2_statistics(df)
    print("=" * 72)
    print("Layer 2 — State Stability")
    print("=" * 72)
    for _, row in l2.iterrows():
        print(f"  {row['metric']:<30s}  {row['value']!s:>12s}")

    # ── Layer 1: Conditional statistics ──
    l1 = layer1_statistics(df)

    print(f"\n{'=' * 72}")
    print("Layer 1 — Direction Regime → Forward Return / Vol")
    print("=" * 72)
    dir_tbl = l1["direction_table"].copy()
    dir_tbl["ann_mean_ret"] = dir_tbl["ann_mean_ret"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")
    dir_tbl["ann_mean_vol"] = dir_tbl["ann_mean_vol"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")
    print(dir_tbl.to_string(index=False))

    print(f"\n{'=' * 72}")
    print("Layer 1 — Volatility Regime → Forward Return / Vol")
    print("=" * 72)
    vol_tbl = l1["vol_table"].copy()
    vol_tbl["ann_mean_ret"] = vol_tbl["ann_mean_ret"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")
    vol_tbl["ann_mean_vol"] = vol_tbl["ann_mean_vol"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")
    print(vol_tbl.to_string(index=False))

    print(f"\n{'=' * 72}")
    print("Layer 1 — Bull vs Bear t-test (one-sided: Bull > Bear)")
    print("=" * 72)
    tt = l1["t_tests"].copy()
    for col in ["t_stat", "p_value", "bull_mean_ann", "bear_mean_ann", "bull_vol_ann", "bear_vol_ann"]:
        tt[col] = tt[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")
    print(tt.to_string(index=False))

    print(f"\n{'=' * 72}")
    print("Layer 1 — Direction Consistency (Bull→pos, Bear→neg)")
    print("=" * 72)
    cons = l1["consistency"].copy()
    cons["bull_pos_rate"] = cons["bull_pos_rate"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")
    cons["bear_neg_rate"] = cons["bear_neg_rate"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")
    print(cons.to_string(index=False))

    # ── Save ──
    out_dir = Path("reports/strategy_v1/evidence")
    out_dir.mkdir(parents=True, exist_ok=True)
    l2.to_csv(out_dir / "regime_stability.csv", index=False)
    l1["direction_table"].to_csv(out_dir / "regime_direction_stats.csv", index=False)
    l1["vol_table"].to_csv(out_dir / "regime_vol_stats.csv", index=False)
    l1["t_tests"].to_csv(out_dir / "regime_ttest.csv", index=False)
    l1["consistency"].to_csv(out_dir / "regime_consistency.csv", index=False)
    print(f"\nSaved CSVs to {out_dir}/")


if __name__ == "__main__":
    main()
