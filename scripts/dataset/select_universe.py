"""
Select stock universe from factor panel.

This script keeps the original selection logic as much as possible:
    1. Hard filters
    2. Composite score
    3. Industry-stratified top-N selection

Changes for Tushare-based base panel:
    - trade_status is optional.
    - If trade_status is missing, trade-status filter is skipped.
    - active_ratio is added to replace part of the old trade-status coverage logic.
    - ST risk is checked by trade_status if available, otherwise by stock_name_short if available.
    - Audit includes board distribution to inspect ChiNext / STAR / main-board bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class UniverseSelectionConfig:
    input_factors: str = "dataset/processed/factor_panel_54_ind.parquet"
    input_base_panel: str = "dataset/processed/unified_daily_panel.parquet"

    output_path: str = "dataset/processed/factor_panel_1500_54_ind.parquet"
    audit_csv: str = "dataset/processed/selected_universe_1500_audit.csv"

    target: int = 1500

    date_col: str = "time"
    stock_col: str = "stock_id"

    # New 24-factor column names.
    valid_factor_col: str = "F001SIZE"
    liquidity_factor_col: str = "F003LIQUIDITY"
    resvol_factor_col: str = "F005RESVOL"

    # Metadata columns.
    industry_col: str = "industry_sw"
    trade_status_col: str | None = "trade_status"
    stock_name_col: str | None = "stock_name_short"

    # Optional selection window.
    # Keep None to preserve old behavior: use full available panel.
    selection_start_date: str | None = None
    selection_end_date: str | None = None

    # Original hard filter thresholds.
    max_mktcap_q: float = 0.95
    min_median_price: float = 2.0
    min_active_ratio: float = 0.60
    min_valid_ratio: float = 0.75
    min_valid_days: int = 180
    trade_normal_min_ratio: float = 0.80
    limit_abnormal_max_ratio: float = 0.20
    st_max_ratio: float = 0.05
    max_leverage: float = 0.95
    earnyld_bottom_q: float = 0.01

    # Original score weights.
    valid_ratio_weight: float = 0.15
    n_valid_days_weight: float = 0.15
    liquidity_weight: float = 0.30
    resvol_weight: float = 0.20
    mktcap_weight: float = 0.20


def run_universe_selection(config: UniverseSelectionConfig) -> dict:
    factor_df, base_df = load_inputs(config)

    metric_factor_df = apply_selection_window(factor_df, config)
    metric_base_df = apply_selection_window(base_df, config)

    metrics = compute_stock_metrics(
        factor_df=metric_factor_df,
        base_df=metric_base_df,
        config=config,
    )

    filtered_metrics, audit_metrics, filter_stats = apply_hard_filters(metrics, config)
    filtered_metrics = compute_composite_score(filtered_metrics, config)

    # Split: STAR gets fixed quota, rest gets remaining slots
    star_mask = filtered_metrics[config.stock_col].astype(str).str.startswith("688")
    star_metrics = filtered_metrics[star_mask].copy()
    main_metrics = filtered_metrics[~star_mask].copy()
    # Exclude BSE from main pool
    main_metrics = main_metrics[
        ~main_metrics[config.stock_col].astype(str).str.startswith(("4", "8", "92"))
    ]

    star_quota = min(100, len(star_metrics))
    main_quota = config.target - star_quota

    star_selected = select_industry_stratified(star_metrics, config, target=star_quota)
    main_selected = select_industry_stratified(main_metrics, config, target=main_quota)
    selected = list(main_selected) + list(star_selected)

    selected_df = factor_df[factor_df[config.stock_col].isin(selected)].copy()

    save_outputs(
        selected_df=selected_df,
        audit_metrics=audit_metrics,
        scored_metrics=filtered_metrics,
        selected=selected,
        filter_stats=filter_stats,
        config=config,
    )

    print_distribution_audit(
        full_factor_df=factor_df,
        selected_df=selected_df,
        scored_metrics=filtered_metrics,
        selected=selected,
        config=config,
    )

    return {
        "selected": selected,
        "n_selected": len(selected),
        "output_path": config.output_path,
        "audit_csv": config.audit_csv,
        "filter_stats": filter_stats,
    }


def load_inputs(config: UniverseSelectionConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    factor_df = pd.read_parquet(config.input_factors)
    factor_df[config.date_col] = pd.to_datetime(factor_df[config.date_col]).dt.normalize()
    factor_df[config.stock_col] = (
        factor_df[config.stock_col].astype(str).str.strip().str.zfill(6)
    )

    base_df = pd.read_parquet(config.input_base_panel)
    base_df[config.date_col] = pd.to_datetime(base_df[config.date_col]).dt.normalize()
    base_df[config.stock_col] = (
        base_df[config.stock_col].astype(str).str.strip().str.zfill(6)
    )

    print(
        f"Loaded factor panel: {len(factor_df):,} rows, "
        f"{factor_df[config.stock_col].nunique():,} stocks"
    )
    print(
        f"Loaded base panel: {len(base_df):,} rows, "
        f"{base_df[config.stock_col].nunique():,} stocks"
    )

    return factor_df, base_df


def apply_selection_window(
    df: pd.DataFrame,
    config: UniverseSelectionConfig,
) -> pd.DataFrame:
    out = df.copy()

    if config.selection_start_date is not None:
        out = out[out[config.date_col] >= pd.Timestamp(config.selection_start_date)]

    if config.selection_end_date is not None:
        out = out[out[config.date_col] <= pd.Timestamp(config.selection_end_date)]

    return out


def compute_stock_metrics(
    factor_df: pd.DataFrame,
    base_df: pd.DataFrame,
    config: UniverseSelectionConfig,
) -> pd.DataFrame:
    required = [
        config.valid_factor_col,
        config.liquidity_factor_col,
        config.resvol_factor_col,
        "mktcap_total",
        "close",
        "limit_status",
    ]
    missing = [col for col in required if col not in factor_df.columns]
    if missing:
        raise ValueError(f"factor_df missing required columns: {missing}")

    grouped = factor_df.groupby(config.stock_col)

    n_valid = grouped[config.valid_factor_col].count()
    n_dates_appeared = grouped[config.date_col].nunique()

    total_dates = factor_df[config.date_col].nunique()
    active_ratio = n_dates_appeared / max(total_dates, 1)

    valid_ratio = n_valid / n_dates_appeared.where(n_dates_appeared > 0)

    mktcap = grouped["mktcap_total"].median()

    # Keep old logic:
    # old script used -LIQUIDITY and -RESVOL before scoring.
    liq = grouped[config.liquidity_factor_col].median()
    resvol = grouped[config.resvol_factor_col].median()  # higher_is_better=False handles direction

    if config.industry_col in factor_df.columns:
        industry = grouped[config.industry_col].last()
    else:
        industry = pd.Series("UNKNOWN", index=n_valid.index)

    median_close = grouped["close"].median()

    limit_status = pd.to_numeric(factor_df["limit_status"], errors="coerce")
    temp = factor_df[[config.stock_col]].copy()
    temp["_limit_status"] = limit_status

    temp_grouped = temp.groupby(config.stock_col)
    limit_abnormal_ratio = temp_grouped["_limit_status"].apply(lambda x: (x != 0).mean())

    trade_normal_ratio, st_ratio = compute_trade_status_metrics(
        factor_df=factor_df,
        base_df=base_df,
        config=config,
        stock_index=n_valid.index,
    )

    stock_name = get_stock_name_by_stock(
        factor_df=factor_df,
        base_df=base_df,
        config=config,
        stock_index=n_valid.index,
    )
    st_by_name = stock_name.astype(str).str.contains("ST", case=False, na=False)

    metrics = pd.DataFrame(
        {
            config.stock_col: n_valid.index,
            "n_valid_days": n_valid.values,
            "n_dates_appeared": n_dates_appeared.values,
            "active_ratio": active_ratio.values,
            "valid_ratio": valid_ratio.values,
            "mktcap_median": mktcap.values,
            "liq_median": liq.values,
            "resvol_median": resvol.values,
            "industry": industry.reindex(n_valid.index).values,
            "median_close": median_close.values,
            "limit_abnormal_ratio": limit_abnormal_ratio.reindex(n_valid.index).values,
            "trade_normal_ratio": trade_normal_ratio.reindex(n_valid.index).values,
            "st_ratio": st_ratio.reindex(n_valid.index).values,
            "stock_name_short": stock_name.reindex(n_valid.index).values,
            "st_by_name": st_by_name.reindex(n_valid.index).fillna(False).values,
        }
    )

    metrics["board"] = metrics[config.stock_col].map(classify_board)

    financial_fields = load_financial_medians(base_df, config)
    for col_name, col_series in financial_fields.items():
        metrics = metrics.merge(
            col_series.reset_index().rename(
                columns={col_series.name: f"{col_name}_median"}
            ),
            on=config.stock_col,
            how="left",
        )

    print(f"\nInitial stocks for selection metrics: {len(metrics):,}")
    print(f"Total trading dates in selection window: {total_dates:,}")

    if config.trade_status_col not in factor_df.columns and config.trade_status_col not in base_df.columns:
        print("  [WARNING] trade_status not found; trade-status filter will be skipped.")
        print("  [INFO] active_ratio filter will partially replace old trading-coverage logic.")

    if config.stock_name_col and config.stock_name_col not in factor_df.columns and config.stock_name_col not in base_df.columns:
        print("  [WARNING] stock_name_short not found; ST-by-name filter will be unavailable.")

    return metrics


def compute_trade_status_metrics(
    factor_df: pd.DataFrame,
    base_df: pd.DataFrame,
    config: UniverseSelectionConfig,
    stock_index: pd.Index,
) -> tuple[pd.Series, pd.Series]:
    if not config.trade_status_col:
        nan_series = pd.Series(np.nan, index=stock_index)
        return nan_series, nan_series

    source = None
    if config.trade_status_col in factor_df.columns:
        source = factor_df[[config.stock_col, config.trade_status_col]].copy()
    elif config.trade_status_col in base_df.columns:
        source = base_df[[config.stock_col, config.trade_status_col]].copy()

    if source is None:
        nan_series = pd.Series(np.nan, index=stock_index)
        return nan_series, nan_series

    source["_trade_status"] = pd.to_numeric(
        source[config.trade_status_col],
        errors="coerce",
    )

    grouped = source.groupby(config.stock_col)["_trade_status"]

    trade_normal_ratio = grouped.apply(lambda x: (x == 1).mean())
    st_ratio = grouped.apply(lambda x: ((x == 2) | (x == 3)).mean())

    return trade_normal_ratio, st_ratio


def get_stock_name_by_stock(
    factor_df: pd.DataFrame,
    base_df: pd.DataFrame,
    config: UniverseSelectionConfig,
    stock_index: pd.Index,
) -> pd.Series:
    if not config.stock_name_col:
        return pd.Series("", index=stock_index)

    if config.stock_name_col in factor_df.columns:
        return (
            factor_df.groupby(config.stock_col)[config.stock_name_col]
            .last()
            .reindex(stock_index)
            .fillna("")
        )

    if config.stock_name_col in base_df.columns:
        return (
            base_df.groupby(config.stock_col)[config.stock_name_col]
            .last()
            .reindex(stock_index)
            .fillna("")
        )

    return pd.Series("", index=stock_index)


def load_financial_medians(
    base_df: pd.DataFrame,
    config: UniverseSelectionConfig,
) -> dict[str, pd.Series]:
    print("\nLoading financial raw fields from base panel...")

    fields = {}
    for col in [
        "total_assets",
        "total_liabilities",
        "equity_parent",
        "revenue_total",
        "net_profit_parent",
    ]:
        if col in base_df.columns:
            series = base_df.groupby(config.stock_col)[col].median()
            fields[col] = series
            print(f"  {col}: available ({series.notna().mean():.1%} non-NaN)")
        else:
            print(f"  {col}: NOT FOUND in base panel")

    return fields


def apply_hard_filters(
    metrics: pd.DataFrame,
    config: UniverseSelectionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    metrics = metrics.copy()

    keep = pd.Series(True, index=metrics.index)
    stats: dict[str, int] = {}

    def apply_filter(mask: pd.Series | np.ndarray, name: str) -> None:
        nonlocal keep

        mask_series = to_bool_series(mask, metrics.index)
        newly_removed = mask_series & keep

        metrics[f"fail_{name}"] = mask_series
        stats[name] = int(newly_removed.sum())
        keep.loc[newly_removed] = False

    cutoff = metrics["mktcap_median"].quantile(config.max_mktcap_q)
    apply_filter(metrics["mktcap_median"] > cutoff, "mega_cap")

    apply_filter(metrics["active_ratio"] < config.min_active_ratio, "active_ratio")
    apply_filter(metrics["valid_ratio"] < config.min_valid_ratio, "valid_ratio")
    apply_filter(metrics["n_valid_days"] < config.min_valid_days, "n_valid_days")
    apply_filter(metrics["median_close"] < config.min_median_price, "low_price")

    if metrics["trade_normal_ratio"].notna().any():
        apply_filter(
            metrics["trade_normal_ratio"] < config.trade_normal_min_ratio,
            "trade_status_abnormal",
        )
    else:
        print("  [WARNING] trade_status filter skipped.")
        metrics["fail_trade_status_abnormal"] = False
        stats["trade_status_abnormal"] = 0

    apply_filter(
        metrics["limit_abnormal_ratio"] > config.limit_abnormal_max_ratio,
        "limit_abnormal",
    )

    if metrics["st_ratio"].notna().any():
        st_mask = (metrics["st_ratio"] > config.st_max_ratio) | metrics["st_by_name"]
    else:
        st_mask = metrics["st_by_name"]

    apply_filter(st_mask, "st_risk_warning")

    if "equity_parent_median" in metrics.columns:
        apply_filter(
            metrics["equity_parent_median"].isna()
            | (metrics["equity_parent_median"] <= 0),
            "negative_equity",
        )
    else:
        print("  [WARNING] equity_parent not available; negative equity filter skipped")
        metrics["fail_negative_equity"] = False
        stats["negative_equity"] = 0

    if (
        "total_assets_median" in metrics.columns
        and "total_liabilities_median" in metrics.columns
    ):
        leverage = metrics["total_liabilities_median"] / metrics[
            "total_assets_median"
        ].where(metrics["total_assets_median"] > 0)

        metrics["leverage_raw"] = leverage

        apply_filter(
            leverage.notna() & (leverage > config.max_leverage),
            "high_leverage",
        )
    else:
        print("  [WARNING] raw leverage unavailable; high leverage filter skipped")
        metrics["fail_high_leverage"] = False
        stats["high_leverage"] = 0

    if "net_profit_parent_median" in metrics.columns:
        e_yld_raw = metrics["net_profit_parent_median"] / metrics["mktcap_median"]
        metrics["earn_yield_raw"] = e_yld_raw

        e_yld_cutoff = e_yld_raw.quantile(config.earnyld_bottom_q)
        apply_filter(
            e_yld_raw.notna() & (e_yld_raw < e_yld_cutoff),
            "extreme_loss",
        )
    else:
        print("  [WARNING] np/mktcap unavailable; extreme loss filter skipped")
        metrics["fail_extreme_loss"] = False
        stats["extreme_loss"] = 0

    metrics["passed_hard_filters"] = keep

    before = len(metrics)
    filtered = metrics.loc[keep].copy().reset_index(drop=True)

    print(f"\nAfter hard filters: {len(filtered):,} / {before:,} stocks")
    for name, n_removed in stats.items():
        print(f"  {name}: removed {n_removed:,}")

    if len(filtered) < config.target:
        print(
            f"\n[WARNING] Only {len(filtered):,} stocks remain, "
            f"below target {config.target:,}. Consider relaxing filters."
        )

    return filtered, metrics, stats


def compute_composite_score(
    metrics: pd.DataFrame,
    config: UniverseSelectionConfig,
) -> pd.DataFrame:
    metrics = metrics.copy()

    # Keep original score logic.
    metrics["score"] = (
        pct_rank(metrics["valid_ratio"]) * config.valid_ratio_weight
        + pct_rank(metrics["n_valid_days"]) * config.n_valid_days_weight
        + pct_rank(metrics["liq_median"]) * config.liquidity_weight
        + pct_rank(metrics["resvol_median"], higher_is_better=False)
        * config.resvol_weight
        + pct_rank(metrics["mktcap_median"]) * config.mktcap_weight
    )

    return metrics


def select_industry_stratified(
    metrics: pd.DataFrame,
    config: UniverseSelectionConfig,
    target: int | None = None,
) -> list[str]:
    if target is None:
        target = config.target
    target = min(target, len(metrics))

    industry_series = metrics["industry"].fillna("UNKNOWN")
    ind_counts = industry_series.value_counts()
    ind_weights = ind_counts / ind_counts.sum()
    allocation = (ind_weights * target).round().astype(int)

    diff = target - allocation.sum()
    if diff != 0:
        step = int(np.sign(diff))
        for _ in range(abs(diff)):
            idx = allocation.argmax()
            allocation.iloc[idx] = max(0, allocation.iloc[idx] + step)

    selected: list[str] = []

    for industry, quota in allocation.items():
        if quota <= 0:
            continue

        pool = metrics[industry_series == industry]
        top = pool.nlargest(quota, "score")
        selected.extend(top[config.stock_col].tolist())

    # In rare rounding cases, fill any shortfall globally by score.
    if len(selected) < target:
        selected_set = set(selected)
        rest = metrics[~metrics[config.stock_col].isin(selected_set)]
        fill = rest.nlargest(target - len(selected), "score")[config.stock_col].tolist()
        selected.extend(fill)

    # In rare over-allocation cases, trim globally by score.
    if len(selected) > target:
        score_map = metrics.set_index(config.stock_col)["score"]
        selected = sorted(selected, key=lambda sid: score_map.get(sid, -np.inf), reverse=True)[:target]

    print(f"\nSelected: {len(selected):,} stocks")

    return selected


def save_outputs(
    selected_df: pd.DataFrame,
    audit_metrics: pd.DataFrame,
    scored_metrics: pd.DataFrame,
    selected: list[str],
    filter_stats: dict[str, int],
    config: UniverseSelectionConfig,
) -> None:
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_parquet(output_path, index=False)

    print(
        f"Output factor panel: {len(selected_df):,} rows, "
        f"{selected_df[config.stock_col].nunique():,} stocks"
    )
    print(f"Saved: {output_path}")

    audit = audit_metrics.copy()

    score_cols = scored_metrics[[config.stock_col, "score"]].copy()
    audit = audit.merge(score_cols, on=config.stock_col, how="left")

    audit["selected"] = audit[config.stock_col].isin(selected)

    for name, n_removed in filter_stats.items():
        audit[f"filter_removed_count_{name}"] = n_removed

    audit_path = Path(config.audit_csv)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)

    print(f"Saved audit: {audit_path}")


def print_distribution_audit(
    full_factor_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    scored_metrics: pd.DataFrame,
    selected: list[str],
    config: UniverseSelectionConfig,
) -> None:
    print("\n========== Distribution Audit ==========")

    print_market_cap_distribution(full_factor_df, selected_df, config)
    print_board_distribution(full_factor_df, selected_df, config)
    print_industry_distribution(scored_metrics, selected, config)


def print_market_cap_distribution(
    full_factor_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    config: UniverseSelectionConfig,
) -> None:
    print("\n=== Market Cap Distribution ===")

    full_mc = full_factor_df.groupby(config.stock_col)["mktcap_total"].median() / 1e8
    sel_mc = selected_df.groupby(config.stock_col)["mktcap_total"].median() / 1e8

    for q in [0.01, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95]:
        print(
            f"  p{int(q * 100):>3d}: "
            f"full={full_mc.quantile(q):.0f}亿 "
            f"sel={sel_mc.quantile(q):.0f}亿"
        )

    bins = [0, 20, 50, 100, 200, 500, 9999]
    labels = ["<20亿", "20-50亿", "50-100亿", "100-200亿", "200-500亿", ">500亿"]

    full_bins = pd.cut(full_mc, bins=bins, labels=labels)
    sel_bins = pd.cut(sel_mc, bins=bins, labels=labels)

    print(f"\n  {'':>12s}  {'Full':>8s}  {'Pct':>6s}  {'Selected':>10s}  {'Pct':>6s}")
    for label in labels:
        print(
            f"  {label:>12s}  "
            f"{(full_bins == label).sum():>8d}  {(full_bins == label).mean():>6.1%}  "
            f"{(sel_bins == label).sum():>10d}  {(sel_bins == label).mean():>6.1%}"
        )

    below_p10 = (sel_mc < full_mc.quantile(0.10)).mean()
    below_p20 = (sel_mc < full_mc.quantile(0.20)).mean()

    print(f"\nSelected below full p10: {below_p10:.2%}")
    print(f"Selected below full p20: {below_p20:.2%}")


def print_board_distribution(
    full_factor_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    config: UniverseSelectionConfig,
) -> None:
    print("\n=== Board Distribution ===")

    full_stock = (
        full_factor_df[[config.stock_col]]
        .drop_duplicates()
        .assign(board=lambda x: x[config.stock_col].map(classify_board))
    )
    selected_stock = (
        selected_df[[config.stock_col]]
        .drop_duplicates()
        .assign(board=lambda x: x[config.stock_col].map(classify_board))
    )

    full_dist = full_stock["board"].value_counts()
    selected_dist = selected_stock["board"].value_counts()

    boards = sorted(set(full_dist.index).union(set(selected_dist.index)))

    print(f"  {'Board':<12s} {'Full':>8s} {'Pct':>8s} {'Selected':>10s} {'Pct':>8s}")
    for board in boards:
        full_n = int(full_dist.get(board, 0))
        sel_n = int(selected_dist.get(board, 0))
        full_pct = full_n / len(full_stock) if len(full_stock) else 0.0
        sel_pct = sel_n / len(selected_stock) if len(selected_stock) else 0.0

        print(
            f"  {board:<12s} "
            f"{full_n:>8d} {full_pct:>8.2%} "
            f"{sel_n:>10d} {sel_pct:>8.2%}"
        )


def print_industry_distribution(
    scored_metrics: pd.DataFrame,
    selected: list[str],
    config: UniverseSelectionConfig,
) -> None:
    print("\n=== Selected Industry Distribution ===")

    selected_metrics = scored_metrics[
        scored_metrics[config.stock_col].isin(selected)
    ].copy()

    dist = selected_metrics["industry"].fillna("UNKNOWN").value_counts()

    for industry, count in dist.head(30).items():
        print(f"  {str(industry):>8s}: {count:>4d}")


def classify_board(stock_id: str) -> str:
    sid = str(stock_id).strip().zfill(6)

    if sid.startswith(("300", "301")):
        return "ChiNext"
    if sid.startswith("688"):
        return "STAR"
    if sid.startswith(("8", "4")) or sid.startswith("92"):
        return "BSE"
    if sid.startswith(("000", "001", "002", "003")):
        return "SZ_Main"
    if sid.startswith(("600", "601", "603", "605")):
        return "SH_Main"

    return "Other"


def pct_rank(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    pct = s.rank(pct=True)
    if not higher_is_better:
        pct = 1.0 - pct
    return pct.fillna(0.0)


def to_bool_series(mask: pd.Series | np.ndarray, index: pd.Index) -> pd.Series:
    if isinstance(mask, pd.Series):
        return mask.reindex(index).fillna(False).astype(bool)

    return pd.Series(mask, index=index).fillna(False).astype(bool)


def main() -> None:
    config = UniverseSelectionConfig(
        input_factors="dataset/processed/factor_panel_54_ind.parquet",
        input_base_panel="dataset/processed/unified_daily_panel.parquet",

        output_path="dataset/processed/factor_panel_1500_54_ind.parquet",
        audit_csv="dataset/processed/selected_universe_1500_audit.csv",

        target=1500,

        # Keep None to preserve old behavior.
        selection_start_date=None,
        selection_end_date=None,
    )

    run_universe_selection(config)

    print(
        "\n[WARNING] Universe metrics computed on the configured selection window. "
        "If selection_start_date/selection_end_date are None, this uses the full available panel. "
        "For strict OOS testing, use train-period-only selection."
    )


if __name__ == "__main__":
    main()