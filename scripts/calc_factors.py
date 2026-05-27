"""
Compute 12 factors + 2 targets from csmar_daily_raw_panel.parquet.

Pipeline:
  1. Load 2012+ raw panel
  2. Group A: daily price/volume factors (8)
  3. Group B: balance-sheet factors (2) — direct from ffill'd fields
  4. Group C: TTM income-statement factors (2) — quarterly-level TTM then ffill
  5. Targets: 1d / 5d forward return
  6. Cross-sectional winsorize (1%/99%) + z-score
  7. Filter to 2018+, save factor_panel_12.parquet
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_factor_panel import (
    FactorPanelConfig,
    _load_financial,
    _shift_to_available_date,
)
from src.utils.seed import set_seed


@dataclass
class FactorConfig:
    input_path: str = "dataset/processed/csmar_daily_raw_panel.parquet"
    output_path: str = "dataset/processed/factor_panel_12_ind.parquet"
    model_start_date: str = "2018-01-01"
    date_col: str = "time"
    stock_col: str = "stock_id"

    # Rolling window sizes
    beta_window: int = 252
    beta_min_periods: int = 60
    liq_window: int = 21
    liq_min_periods: int = 10

    seed: int = 42


# ---- Group A: daily price/volume factors ----

def _build_size(df: pd.DataFrame) -> pd.Series:
    mktcap = df["mktcap_total"].fillna(df["mktcap_float"])
    result = np.log(mktcap)
    result[~np.isfinite(result)] = np.nan
    return result


def _build_sizenl(df: pd.DataFrame, size_raw: pd.Series) -> pd.Series:
    return size_raw.groupby(df["time"]).rank(pct=True) - 0.5


def _build_liquidity(df: pd.DataFrame) -> pd.Series:
    amt_to_total = df["amount"] / df["mktcap_total"].where(
        df["mktcap_total"].fillna(0) > 0
    )
    result = (
        np.log1p(amt_to_total.clip(0))
        .groupby(df["stock_id"])
        .rolling(window=63, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    result[~np.isfinite(result)] = np.nan
    return result


def _build_beta_resvol(df: pd.DataFrame, window: int, min_p: int):
    ri = df["ret_daily"]
    rm = df["mkt_ret_vw"]
    g = df["stock_id"]

    E_ri = ri.groupby(g).rolling(window, min_periods=min_p).mean().reset_index(level=0, drop=True)
    E_rm = rm.groupby(g).rolling(window, min_periods=min_p).mean().reset_index(level=0, drop=True)
    E_ri2 = (ri**2).groupby(g).rolling(window, min_periods=min_p).mean().reset_index(level=0, drop=True)
    E_rm2 = (rm**2).groupby(g).rolling(window, min_periods=min_p).mean().reset_index(level=0, drop=True)
    E_rirm = (ri*rm).groupby(g).rolling(window, min_periods=min_p).mean().reset_index(level=0, drop=True)

    cov_im = E_rirm - E_ri * E_rm
    var_m = E_rm2 - E_rm ** 2
    var_i = E_ri2 - E_ri ** 2

    beta = cov_im / var_m.where(var_m > 1e-10, np.nan)
    res_var = var_i - cov_im**2 / var_m.where(var_m > 1e-10, np.nan)
    resvol = np.sqrt(np.maximum(res_var, 0))

    return beta, resvol


def _build_momentum(df: pd.DataFrame) -> pd.Series:
    g = df.groupby("stock_id")["close_adj"]
    return g.shift(21) / g.shift(252) - 1


def _build_ltrev(df: pd.DataFrame) -> pd.Series:
    g = df.groupby("stock_id")["close_adj"]
    # Three-window log-return ensemble, best corr 0.66 vs teacher
    def _log_rev(lookback):
        return -(np.log(g.shift(252).clip(1e-10)) - np.log(g.shift(lookback).clip(1e-10)))
    parts = []
    for lb in [504, 630, 756]:
        p = _log_rev(lb)
        # Z-score each window per date before averaging
        mu = p.groupby(df["time"]).transform("mean")
        sd = p.groupby(df["time"]).transform("std")
        parts.append((p - mu) / sd.where(sd > 1e-10, 1.0))
    return (parts[0] + parts[1] + parts[2]) / 3.0


def _build_strev(df: pd.DataFrame) -> pd.Series:
    g = df.groupby("stock_id")["close_adj"]
    return -(g.shift(1) / g.shift(21) - 1)


# ---- Group B: balance-sheet factors ----

def _build_leverage(df: pd.DataFrame) -> pd.Series:
    result = df["total_liabilities"] / df["total_assets"].where(df["total_assets"] > 0)
    result[~np.isfinite(result)] = np.nan
    return result


def _build_value(df: pd.DataFrame) -> pd.Series:
    mkt = df["mktcap_total"].fillna(df["mktcap_float"])
    result = df["equity_parent"] / mkt.where(mkt > 0)
    result[~np.isfinite(result)] = np.nan
    return result


# ---- Group C: TTM income-statement factors ----

def _cumulative_to_single_quarter(fin: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Convert cumulative YTD financial fields to single-quarter values.

    Sorted by stock_id + accper (accounting period), NOT available_date.
    Q1 = Q1 cumulative. Q2/Q3/Q4 = cumulative - prior quarter of SAME year.
    Cross-year diffs are rejected (return NaN).
    """
    fin = fin.sort_values(["stock_id", "accper"]).reset_index(drop=True)

    for col in value_cols:
        fin[f"{col}_sq"] = np.nan

    for stock, grp in fin.groupby("stock_id"):
        grp = grp.sort_values("accper")
        idx = grp.index
        for col in value_cols:
            vals = grp[col].values
            accpers = pd.DatetimeIndex(grp["accper"].values)
            sq = np.full(len(vals), np.nan)
            for i in range(len(vals)):
                m = accpers[i].month
                y = accpers[i].year
                if m == 3:
                    sq[i] = vals[i]  # Q1 = Q1 cumulative
                elif m in {6, 9, 12}:
                    if i == 0:
                        continue
                    prev_m = accpers[i - 1].month
                    prev_y = accpers[i - 1].year
                    # Only diff if previous row is same-year prior quarter
                    if prev_y == y and prev_m == {6: 3, 9: 6, 12: 9}.get(m):
                        if np.isfinite(vals[i]) and np.isfinite(vals[i - 1]):
                            sq[i] = vals[i] - vals[i - 1]
                else:
                    sq[i] = vals[i]  # non-standard month: keep as-is (fallback)
            fin.loc[idx, f"{col}_sq"] = sq

    return fin


def _compute_ttm(fin: pd.DataFrame, col_sq: str) -> pd.Series:
    """Compute rolling 4-quarter TTM from single-quarter values."""
    result = pd.Series(np.nan, index=fin.index)
    for stock, grp in fin.groupby("stock_id"):
        grp = grp.sort_values("available_date")
        vals = grp[col_sq].values
        ttm = np.full(len(vals), np.nan)
        for i in range(len(vals)):
            window = vals[max(0, i - 3):i + 1]
            if len(window) >= 4 and np.isfinite(window).all():
                ttm[i] = window.sum()
        result.loc[grp.index] = ttm
    return result


def _build_ttm_factors(df: pd.DataFrame, fc: FactorConfig) -> pd.DataFrame:
    """
    Compute EARNYLD and GROWTH at quarterly level with TTM,
    then forward-fill to daily panel.
    """
    print("  Computing TTM from quarterly financials...")

    # Load raw quarterly financial data
    fp_config = FactorPanelConfig()
    fin = _load_financial(fp_config)

    # Filter non-standard Accper months before TTM
    fin["_accper_m"] = pd.to_datetime(fin["accper"]).dt.month
    fin = fin[fin["_accper_m"].isin({3, 6, 9, 12})].copy()

    # Cumulative → single quarter (sorted by accper internally)
    cumul_cols = ["revenue_total", "net_profit_parent"]
    fin = _cumulative_to_single_quarter(fin, cumul_cols)

    # TTM from single quarter
    fin["revenue_ttm"] = _compute_ttm(fin, "revenue_total_sq")
    fin["np_ttm"] = _compute_ttm(fin, "net_profit_parent_sq")

    # TTM YoY: (TTM_t - TTM_{t-4}) / |TTM_{t-4}|, sorted by accper
    fin = fin.sort_values(["stock_id", "accper"]).reset_index(drop=True)
    for col in ["revenue_ttm", "np_ttm"]:
        fin[f"{col}_yoy"] = fin.groupby("stock_id")[col].transform(
            lambda x: (x - x.shift(4)) / x.shift(4).abs().where(x.shift(4).abs() > 1e-8)
        )

    # Asset & equity YoY (from same quarterly financial data, sorted by accper)
    fin_bs = fin.sort_values(["stock_id", "accper"]).reset_index(drop=True)
    for col in ["total_assets", "equity_parent"]:
        if col in fin_bs.columns:
            fin_bs[f"{col}_yoy"] = fin_bs.groupby("stock_id")[col].transform(
                lambda x: (x - x.shift(4)) / x.shift(4).abs().where(x.shift(4).abs() > 1e-8)
            )

    # Merge BS YoY into fin for unified ffill
    bs_merge = fin_bs[["stock_id", "available_date", "accper",
                        "total_assets_yoy", "equity_parent_yoy"]].dropna(
        subset=["total_assets_yoy", "equity_parent_yoy"], how="all"
    )
    fin = fin.merge(bs_merge, on=["stock_id", "available_date", "accper"], how="left")

    fin_cols = ["stock_id", "available_date", "accper",
                "revenue_ttm", "np_ttm", "revenue_ttm_yoy",
                "total_assets_yoy", "equity_parent_yoy"]
    fin_daily = fin[fin_cols].copy()
    # Sort and keep last (latest accper) per (stock_id, available_date)
    fin_daily = fin_daily.sort_values(
        ["stock_id", "available_date", "accper"]
    ).reset_index(drop=True)
    fin_daily = fin_daily.drop_duplicates(
        subset=["stock_id", "available_date"], keep="last"
    )

    # Merge into daily panel via groupby stock → ffill
    daily = df[[fc.date_col, fc.stock_col]].copy()
    daily = daily.sort_values([fc.stock_col, fc.date_col]).reset_index(drop=True)

    idx_map = {}
    for stock, grp in fin_daily.groupby("stock_id"):
        stock_dates = fin_daily.loc[grp.index, "available_date"].values
        idx_map[stock] = (stock_dates, grp)

    fill_cols = ["revenue_ttm", "np_ttm", "revenue_ttm_yoy",
                 "total_assets_yoy", "equity_parent_yoy"]
    for c in fill_cols:
        daily[c] = np.nan

    fin_parts = []
    for stock, group in daily.groupby(fc.stock_col):
        if stock not in idx_map:
            fin_parts.append(group)
            continue
        stock_dates, grp = idx_map[stock]
        fin_dates = stock_dates
        for c in fill_cols:
            fin_vals = grp[c].values
            daily_vals = np.full(len(group), np.nan)
            if len(fin_vals) > 0:
                idx = np.searchsorted(fin_dates, group[fc.date_col].values, side="right") - 1
                valid = idx >= 0
                daily_vals[valid] = fin_vals[idx[valid]]
            group = group.copy()
            group[c] = daily_vals
        fin_parts.append(group)

    daily = pd.concat(fin_parts, ignore_index=True)

    # EARNYLD = np_ttm / mktcap
    mkt = df["mktcap_total"].fillna(df["mktcap_float"])
    daily["EARNYLD_raw"] = daily["np_ttm"] / mkt.where(mkt > 0)

    # GROWTH = 0.50*z(rev_ttm_yoy) + 0.25*z(asset_yoy) + 0.25*z(equity_yoy)
    # Z-score each component per date, then weighted sum
    def _pdate_z(s):
        mu = s.groupby(daily[fc.date_col]).transform("mean")
        sd = s.groupby(daily[fc.date_col]).transform("std")
        return (s - mu) / sd.where(sd > 1e-10, 1.0)

    daily["GROWTH_raw"] = (
        0.50 * _pdate_z(daily["revenue_ttm_yoy"])
        + 0.25 * _pdate_z(daily["total_assets_yoy"])
        + 0.25 * _pdate_z(daily["equity_parent_yoy"])
    )

    keep_cols = [fc.date_col, fc.stock_col, "EARNYLD_raw", "GROWTH_raw"]
    return daily[[c for c in keep_cols if c in daily.columns]]


# ---- Targets ----

def _build_targets(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("stock_id")["open"]
    result = pd.DataFrame(index=df.index)
    # T+1: buy at t+1 open, sell at t+2 open
    result["1d_next_raw"] = g.shift(-2) / g.shift(-1) - 1
    # T+1 buy at t+1 open, sell at t+7 open (~5 trading days)
    result["5d_next_raw"] = g.shift(-6) / g.shift(-1) - 1
    return result


# ---- Cross-sectional winsorize + z-score ----

def _winsorize_zscore(df: pd.DataFrame, factor_names: list[str],
                      industry_col: str | None = None,
                      neutralize_factors: list[str] | None = None) -> pd.DataFrame:
    """
    Winsorize (1%/99%) then z-score. If neutralize_factors is provided,
    only those factors get industry neutralization before cross-sectional z-score.
    """
    if neutralize_factors is None:
        neutralize_factors = factor_names
    neut_set = set(neutralize_factors)

    for name in factor_names:
        raw_col = f"{name}_raw"
        if raw_col not in df.columns:
            print(f"    [WARNING] {raw_col} not found, skipping")
            continue

        # Winsorize: clip to [1%, 99%] per date
        lo = df.groupby("time")[raw_col].transform(lambda x: x.quantile(0.01))
        hi = df.groupby("time")[raw_col].transform(lambda x: x.quantile(0.99))
        w_col = f"{name}_w"
        df[w_col] = df[raw_col].clip(lo, hi)

        # Step 1: Industry neutralization (only for selected factors)
        if industry_col and industry_col in df.columns and name in neut_set:
            mu_ind = df.groupby(["time", industry_col])[w_col].transform("mean")
            sd_ind = df.groupby(["time", industry_col])[w_col].transform("std")
            df[w_col] = (df[w_col] - mu_ind) / sd_ind.where(sd_ind > 1e-10, 1.0)

        # Step 2: Cross-sectional z-score
        mu = df.groupby("time")[w_col].transform("mean")
        sd = df.groupby("time")[w_col].transform("std")
        df[name] = (df[w_col] - mu) / sd.where(sd > 1e-10, 1.0)

    return df


# ---- Sanity checks ----

def _sanity_checks(df: pd.DataFrame, factor_names: list[str]) -> None:
    print("\n========== Factor Sanity Checks ==========")
    print(f"Rows: {len(df):,}  Stocks: {df['stock_id'].nunique():,}")

    for name in factor_names:
        col = name
        missing = df[col].isna().mean()
        extreme = (df[col].abs() > 5).mean()
        print(f"  {name:12s}  missing={missing:.4f}  extreme(|z|>5)={extreme:.4f}")

    # Cross-sectional mean/std for first 5 dates
    dates = pd.DatetimeIndex(df["time"].unique()).sort_values()[:5]
    print(f"\n  Cross-sectional mean/std for first dates:")
    for d in dates:
        sub = df[df["time"] == d]
        means = [sub[f].mean() for f in factor_names]
        stds = [sub[f].std() for f in factor_names]
        print(f"    {d.date()}: mean range [{min(means):.3f}, {max(means):.3f}]  "
              f"std range [{min(stds):.3f}, {max(stds):.3f}]")

    # Correlation matrix sample
    clean = df[factor_names].dropna()
    sample = clean.sample(min(5000, len(clean)), random_state=42)
    corr = sample.corr()
    print(f"\n  Correlation matrix (sample n={len(sample)}):")
    for fi in factor_names[:6]:
        row = " ".join(f"{corr.loc[fi, fj]:6.2f}" for fj in factor_names[:6])
        print(f"    {fi:12s} {row}")


# ---- Main ----

def main() -> None:
    config = FactorConfig()
    set_seed(config.seed)

    print("Loading input panel...")
    df = pd.read_parquet(config.input_path)
    # Unify stock_id as string (matches financial data merge)
    df[config.stock_col] = df[config.stock_col].astype(str).str.strip()
    print(f"  {len(df):,} rows, {df[config.stock_col].nunique():,} stocks")

    df = df.sort_values([config.stock_col, config.date_col]).reset_index(drop=True)

    # --- Group A ---
    print("\nGroup A: Daily price/volume factors...")

    print("  SIZE...")
    df["SIZE_raw"] = _build_size(df)

    print("  SIZENL...")
    df["SIZENL_raw"] = _build_sizenl(df, df["SIZE_raw"])

    print("  LIQUIDITY...")
    df["LIQUIDITY_raw"] = _build_liquidity(df)

    print("  BETA + RESVOL...")
    beta, resvol = _build_beta_resvol(df, config.beta_window, config.beta_min_periods)
    df["BETA_raw"] = beta
    df["RESVOL_raw"] = resvol

    print("  MOMENTUM...")
    df["MOMENTUM_raw"] = _build_momentum(df)

    print("  LTREV...")
    df["LTREV_raw"] = _build_ltrev(df)

    print("  STREV...")
    df["STREV_raw"] = _build_strev(df)

    # --- Group B ---
    print("\nGroup B: Balance-sheet factors...")
    print("  LEVERAGE...")
    df["LEVERAGE_raw"] = _build_leverage(df)
    print("  VALUE...")
    df["VALUE_raw"] = _build_value(df)

    # --- Group C ---
    print("\nGroup C: TTM income-statement factors...")
    ttm_df = _build_ttm_factors(df, config)
    df = df.merge(ttm_df, on=[config.date_col, config.stock_col], how="left")

    # --- Targets ---
    print("\nBuilding targets...")
    targets = _build_targets(df)
    df["1d_next_raw"] = targets["1d_next_raw"]
    df["5d_next_raw"] = targets["5d_next_raw"]

    # --- Winsorize + Z-score ---
    factor_names = [
        "SIZE", "SIZENL", "LIQUIDITY", "BETA", "RESVOL",
        "MOMENTUM", "LTREV", "STREV", "LEVERAGE", "VALUE",
        "EARNYLD", "GROWTH",
    ]
    print("\nCross-sectional winsorize + z-score...")
    fin_factors = ["SIZE", "SIZENL", "LIQUIDITY", "VALUE", "LEVERAGE", "EARNYLD", "GROWTH"]
    df = _winsorize_zscore(df, factor_names,
                           industry_col="industry_csrc_2012", neutralize_factors=fin_factors)

    # --- Filter to model start date ---
    print(f"\nFiltering to time >= {config.model_start_date}...")
    before = len(df)
    df = df[df[config.date_col] >= config.model_start_date].copy()
    print(f"  {before:,} → {len(df):,} rows")

    # --- Sanity checks ---
    _sanity_checks(df, factor_names)

    # --- Save ---
    # Output: needed columns = date, stock_id, 12 factors, 2 targets, key metadata
    output_cols = [config.date_col, config.stock_col] + factor_names + \
                  ["1d_next_raw", "5d_next_raw"]
    # Add metadata cols if present
    for meta in ["industry_csrc_2012", "list_date", "open", "close", "close_adj",
                 "ret_daily", "mktcap_float", "mktcap_total", "volume",
                 "trade_status", "limit_status", "change_ratio"]:
        if meta in df.columns:
            output_cols.append(meta)

    df_out = df[[c for c in output_cols if c in df.columns]]

    output_path = Path(config.output_path)
    df_out.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"  rows={len(df_out):,}  stocks={df_out[config.stock_col].nunique():,}  cols={len(df_out.columns)}")


if __name__ == "__main__":
    main()
