"""
Apply select_stocks logic to factor_panel_12.parquet.
Output factor_panel_1500.parquet for experiments.

Selection pipeline:
  1. Hard filters: mega-cap, valid_ratio, n_valid_days, low price,
     trade_status, limit_status, ST, negative equity, high leverage, extreme loss
  2. Composite scoring: data_quality(0.20) + liquidity(0.40) + stability(0.20) + size(0.20)
  3. Industry-stratified top-1500 selection

Financial raw fields loaded from csmar_daily_raw_panel.parquet (not in factor_panel_12).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INPUT_FACTORS = "dataset/processed/factor_panel_12_ind.parquet"
INPUT_FINANCIAL = "dataset/processed/csmar_daily_raw_panel.parquet"
OUTPUT = "dataset/processed/factor_panel_1500_ind.parquet"
AUDIT_CSV = "dataset/processed/selected_universe_audit.csv"
TARGET = 1500
MAX_MKTCAP_Q = 0.95
MIN_MEDIAN_PRICE = 2.0
TRADE_NORMAL_MIN_RATIO = 0.80
LIMIT_ABNORMAL_MAX_RATIO = 0.20
MAX_LEVERAGE = 0.95
EARNYLD_BOTTOM_Q = 0.01


def _pct_rank(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    pct = s.rank(pct=True)
    if not higher_is_better:
        pct = 1.0 - pct
    return pct.fillna(0.0)


def main():
    # ---- Load factor panel ----
    df = pd.read_parquet(INPUT_FACTORS)
    df["time"] = pd.to_datetime(df["time"]).dt.normalize()
    df["stock_id"] = df["stock_id"].astype(str).str.strip()
    print(f"Loaded factor panel: {len(df):,} rows, {df['stock_id'].nunique()} stocks")

    # ---- Load financial raw fields ----
    print("Loading financial raw fields...")
    fin = pd.read_parquet(INPUT_FINANCIAL)
    fin["time"] = pd.to_datetime(fin["time"]).dt.normalize()
    fin["stock_id"] = fin["stock_id"].astype(str).str.strip()
    fin_fields = {}
    for col in ["total_assets", "total_liabilities", "equity_parent",
                "revenue_total", "net_profit_parent"]:
        if col in fin.columns:
            fin_fields[col] = fin.groupby("stock_id")[col].median()
            print(f"  {col}: available ({fin_fields[col].notna().mean():.1%} non-NaN)")
        else:
            print(f"  {col}: NOT FOUND in raw panel")

    # ---- Per-stock metrics from factor panel ----
    grouped = df.groupby("stock_id")
    n_valid = grouped["SIZE"].count()
    n_dates_appeared = grouped["time"].nunique()
    valid_ratio = n_valid / n_dates_appeared
    mktcap = grouped["mktcap_total"].median()
    liq = grouped["LIQUIDITY"].median()
    resvol = grouped["RESVOL"].median()
    industry = grouped["industry_csrc_2012"].last()
    median_close = grouped["close"].median()

    # Trade status: 1=normal, 3=ST, 2=*ST, others uncommon
    trade_normal_ratio = grouped["trade_status"].apply(lambda x: (x == 1).mean())
    # Limit status: 0=normal, 1=涨停, -1=跌停
    limit_abnormal_ratio = grouped["limit_status"].apply(lambda x: (x != 0).mean())
    # ST: trade_status == 3 (ST) or 2 (*ST)
    st_ratio = grouped["trade_status"].apply(lambda x: ((x == 2) | (x == 3)).mean())

    metrics = pd.DataFrame({
        "stock_id": n_valid.index,
        "n_valid_days": n_valid.values,
        "valid_ratio": valid_ratio.values,
        "mktcap_median": mktcap.values,
        "liq_median": liq.values,
        "resvol_median": resvol.values,
        "industry": industry.values,
        "median_close": median_close.values,
        "trade_normal_ratio": trade_normal_ratio.values,
        "limit_abnormal_ratio": limit_abnormal_ratio.values,
        "st_ratio": st_ratio.values,
    })

    # Merge financial raw fields
    for col_name, col_series in fin_fields.items():
        metrics = metrics.merge(
            col_series.reset_index().rename(columns={col_series.name: col_name + "_median"}),
            on="stock_id", how="left"
        )

    n_total = len(metrics)
    print(f"\nInitial stocks: {n_total}")

    # ---- Hard filters ----
    keep = np.ones(len(metrics), dtype=bool)
    stats = {}

    def apply_filter(mask, name):
        stats[name] = int(mask.sum())
        nonlocal_idx = np.where(mask)[0]
        keep[nonlocal_idx] = False

    # 1. Mega-cap
    cutoff = metrics["mktcap_median"].quantile(MAX_MKTCAP_Q)
    apply_filter(metrics["mktcap_median"] > cutoff, "mega_cap")

    # 2. valid_ratio
    apply_filter(metrics["valid_ratio"] < 0.75, "valid_ratio")

    # 3. n_valid_days
    apply_filter(metrics["n_valid_days"] < 180, "n_valid_days")

    # 4. Low price
    apply_filter(metrics["median_close"] < MIN_MEDIAN_PRICE, "low_price")

    # 5. Trade status abnormal (if normal status = 1 confirmed)
    apply_filter(metrics["trade_normal_ratio"] < TRADE_NORMAL_MIN_RATIO, "trade_status_abnormal")

    # 6. Limit status abnormal
    apply_filter(metrics["limit_abnormal_ratio"] > LIMIT_ABNORMAL_MAX_RATIO, "limit_abnormal")

    # 7. ST / risk warning
    apply_filter(metrics["st_ratio"] > 0.05, "st_risk_warning")

    # 8. Negative equity
    if "equity_parent_median" in metrics.columns:
        apply_filter(metrics["equity_parent_median"].isna() | (metrics["equity_parent_median"] <= 0),
                     "negative_equity")
    else:
        print("  [WARNING] equity_parent not available; negative equity filter skipped")

    # 9. High leverage
    if "total_assets_median" in metrics.columns and "total_liabilities_median" in metrics.columns:
        leverage = metrics["total_liabilities_median"] / metrics["total_assets_median"].where(
            metrics["total_assets_median"] > 0
        )
        apply_filter((leverage.notna() & (leverage > MAX_LEVERAGE)), "high_leverage")
    else:
        print("  [WARNING] raw leverage unavailable; high leverage filter skipped")

    # 10. Extreme loss
    # EARNYLD from factor panel is cross-sectionally normalized; find raw proxy from raw panel
    if "net_profit_parent_median" in metrics.columns and "mktcap_median" in metrics.columns:
        e_yld_raw = metrics["net_profit_parent_median"] / metrics["mktcap_median"]
        e_yld_cutoff = e_yld_raw.quantile(EARNYLD_BOTTOM_Q)
        apply_filter(e_yld_raw.notna() & (e_yld_raw < e_yld_cutoff), "extreme_loss")
    else:
        print("  [WARNING] np/mktcap not available; extreme loss filter skipped")

    # ---- Apply filters ----
    metrics = metrics.loc[keep].copy().reset_index(drop=True)
    print(f"\nAfter hard filters: {len(metrics)} stocks")
    for k, v in stats.items():
        print(f"  {k}: removed {v}")

    if len(metrics) < TARGET:
        print(f"\n[WARNING] Only {len(metrics)} stocks remain, below target {TARGET}. "
              f"Consider relaxing filters.")

    # ---- Composite score ----
    metrics["score"] = (
        _pct_rank(metrics["valid_ratio"]) * 0.10
        + _pct_rank(metrics["n_valid_days"]) * 0.10
        + _pct_rank(metrics["liq_median"]) * 0.40
        + _pct_rank(metrics["resvol_median"], higher_is_better=False) * 0.20
        + _pct_rank(metrics["mktcap_median"]) * 0.20
    )

    # ---- Industry-stratified selection ----
    ind_counts = metrics["industry"].value_counts()
    ind_weights = ind_counts / ind_counts.sum()
    allocation = (ind_weights * min(TARGET, len(metrics))).round().astype(int)
    diff = min(TARGET, len(metrics)) - allocation.sum()
    for i in range(abs(diff)):
        allocation.iloc[allocation.argmax()] += np.sign(diff)

    selected = []
    for ind, quota in allocation.items():
        if quota <= 0:
            continue
        pool = metrics[metrics["industry"] == ind]
        top = pool.nlargest(quota, "score")
        selected.extend(top["stock_id"].tolist())

    print(f"\nSelected: {len(selected)} stocks")

    # ---- Output factor panel ----
    out = df[df["stock_id"].isin(selected)].copy()
    print(f"Output: {len(out):,} rows, {out['stock_id'].nunique()} stocks")
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT, index=False)
    print(f"Saved: {OUTPUT}")

    # ---- Audit CSV ----
    audit = metrics.copy()
    audit["selected"] = audit["stock_id"].isin(selected)
    # Add filter flags
    for k, v in stats.items():
        if k not in audit.columns:
            audit[k] = 0
    audit.to_csv(AUDIT_CSV, index=False)
    print(f"Saved: {AUDIT_CSV}")

    # ---- Distribution audit ----
    print(f"\n=== Market Cap Distribution ===")
    full_mc = df.groupby("stock_id")["mktcap_total"].median() / 1e8
    sel_mc = out.groupby("stock_id")["mktcap_total"].median() / 1e8
    for q in [0.01, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95]:
        print(f"  p{int(q*100):>3d}:  full={full_mc.quantile(q):.0f}亿  sel={sel_mc.quantile(q):.0f}亿")

    bins = [0, 20, 50, 100, 200, 500, 9999]
    labels = ["<20亿", "20-50亿", "50-100亿", "100-200亿", "200-500亿", ">500亿"]
    fb = pd.cut(full_mc, bins=bins, labels=labels)
    sb = pd.cut(sel_mc, bins=bins, labels=labels)
    print(f"\n  {'':>12s}  {'Full':>8s}  {'Pct':>6s}  {'Selected':>10s}  {'Pct':>6s}")
    for b in labels:
        print(f"  {b:>12s}  {(fb==b).sum():>8d}  {(fb==b).mean():>6.1%}  "
              f"{(sb==b).sum():>10d}  {(sb==b).mean():>6.1%}")

    below_p10 = (sel_mc < full_mc.quantile(0.10)).mean()
    below_p20 = (sel_mc < full_mc.quantile(0.20)).mean()
    print(f"\nSelected below full p10: {below_p10:.2%}")
    print(f"Selected below full p20: {below_p20:.2%}")

    print("\n[WARNING] Universe metrics computed on full available panel. "
          "For strict OOS testing, consider train-period-only selection.")


if __name__ == "__main__":
    main()
