"""
Base panel construction and extension utilities.

This module builds the reusable unified daily base panel.

Design
------
1. One-time historical build:
   - Tushare cleaned daily panel is the daily master table.
   - CSMAR financials are the main financial source.
   - Tushare financial data only fills missing quarterly records.
   - Tushare disclosure dates provide actual available_date.

2. Row extension:
   - Append new daily rows.
   - Reuse existing latest financial records by forward fill.
   - Run sentinel checks before accepting the update.

3. Column extension:
   - Merge newly computed columns by (stock_id, time).
   - Existing columns are not overwritten unless explicitly allowed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass
class BasePanelConfig:
    """Configuration for building the unified base panel."""

    tushare_daily_path: str | Path = "dataset/input/tushare/daily_raw_panel.parquet"
    csmar_data_dir: str | Path = "dataset/input/original_data"
    tushare_financial_dir: str | Path = "dataset/input/tushare/financial"
    disclosure_dates_path: str | Path = "dataset/input/tushare/disclosure_dates.parquet"
    industry_path: str | Path | None = "dataset/input/Aindustry.xlsx"

    output_dir: str | Path = "dataset/processed"
    unified_panel_name: str = "unified_daily_panel.parquet"
    financial_panel_name: str = "financial_quarterly_panel.parquet"
    audit_name: str = "base_panel_audit.json"

    start_date: str = "2012-01-01"
    end_date: str = "2026-05-29"

    date_col: str = "time"
    stock_col: str = "stock_id"

    # Whether to drop CSMAR Accper=01-01 duplicate labels.
    # These records duplicate previous Q4 values and should not participate
    # in TTM or financial availability logic.
    drop_accper_0101_duplicates: bool = True

    # Daily panel fields that must be valid.
    required_daily_cols: tuple[str, ...] = (
        "stock_id",
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "ret_daily",
        "mktcap_total",
        "mktcap_float",
        "close_adj",
    )

    # Keep adj_factor for adjustment audit and future daily updates.
    keep_adj_factor: bool = True


# ---------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------


def build_base_panel(config: BasePanelConfig | None = None) -> dict[str, Any]:
    """
    Build the full historical base panel once.

    Outputs
    -------
    unified_daily_panel.parquet
        Daily panel with financial fields forward-filled.

    financial_quarterly_panel.parquet
        Quarterly financial panel with actual available_date.

    base_panel_audit.json
        Basic audit summary.
    """

    if config is None:
        config = BasePanelConfig()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading Tushare cleaned daily panel...")
    daily = load_tushare_daily_panel(config)

    print("Step 2: Building quarterly financial panel...")
    financial = build_financial_quarterly_panel(config)

    financial_path = output_dir / config.financial_panel_name
    financial.to_parquet(financial_path, index=False)
    print(f"Saved financial quarterly panel: {financial_path}")

    print("Step 3: Forward-filling financials to daily panel...")
    panel = forward_fill_financial_to_daily(
        daily=daily,
        financial=financial,
        config=config,
    )

    print("Step 4: Merging industry metadata...")
    panel = _merge_industry_metadata(panel, config)

    print("Step 5: Merging list date from CSMAR...")
    panel = _merge_list_date(panel, config)

    print("Step 6: Running base panel checks...")
    audit = run_base_panel_checks(
        panel=panel,
        financial=financial,
        config=config,
    )

    unified_path = output_dir / config.unified_panel_name
    panel.to_parquet(unified_path, index=False)
    print(f"Saved unified daily panel: {unified_path}")

    audit_path = output_dir / config.audit_name
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Saved audit: {audit_path}")

    return {
        "unified_panel_path": str(unified_path),
        "financial_panel_path": str(financial_path),
        "audit_path": str(audit_path),
        "audit": audit,
    }


def append_base_panel_rows(
    base_panel: pd.DataFrame,
    new_daily_rows: pd.DataFrame,
    financial_quarterly: pd.DataFrame,
    config: BasePanelConfig | None = None,
    allow_overlap_replace: bool = False,
) -> pd.DataFrame:
    """
    Row extension API for daily updates.

    This appends new date-stock rows to an existing unified_daily_panel.

    Parameters
    ----------
    base_panel:
        Existing unified daily panel.

    new_daily_rows:
        New cleaned daily rows. They should already contain daily fields such
        as open, close, ret_daily, close_adj, mktcap_total, etc.

    financial_quarterly:
        Existing quarterly financial panel. The latest available financial
        record is forward-filled into the new daily rows.

    allow_overlap_replace:
        If False, overlapping date-stock rows raise an error.
        If True, new rows replace existing rows on duplicated date-stock keys.
    """

    if config is None:
        config = BasePanelConfig()

    base = _standardize_panel_keys(base_panel, config)
    new_daily = _standardize_panel_keys(new_daily_rows, config)
    financial = _standardize_financial_keys(financial_quarterly, config)

    _validate_required_columns(new_daily, config.required_daily_cols, "new_daily_rows")
    _validate_no_duplicate_keys(new_daily, config, "new_daily_rows")

    new_panel = forward_fill_financial_to_daily(
        daily=new_daily,
        financial=financial,
        config=config,
    )

    key_cols = [config.stock_col, config.date_col]

    if allow_overlap_replace:
        overlap_keys = new_panel[key_cols].drop_duplicates()
        marker = overlap_keys.assign(_new_key=1)
        base = base.merge(marker, on=key_cols, how="left")
        base = base[base["_new_key"].isna()].drop(columns=["_new_key"])
    else:
        duplicated = base[key_cols].merge(
            new_panel[key_cols],
            on=key_cols,
            how="inner",
        )
        if not duplicated.empty:
            sample = duplicated.head(10)
            raise ValueError(
                "new_daily_rows overlap existing base_panel keys. "
                f"Sample:\n{sample}"
            )

    combined = pd.concat([base, new_panel], ignore_index=True)
    combined = combined.sort_values(key_cols).reset_index(drop=True)

    run_base_panel_checks(
        panel=combined,
        financial=financial,
        config=config,
        is_incremental=True,
    )

    return combined


def merge_base_panel_columns(
    base_panel: pd.DataFrame,
    new_columns: pd.DataFrame,
    config: BasePanelConfig | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Column extension API.

    Merge new columns into the base panel by stock_id + time.

    Use cases
    ---------
    - Add industry fields.
    - Add new audit fields.
    - Add future source flags.
    - Add precomputed helper columns.

    Existing columns are protected by default.
    """

    if config is None:
        config = BasePanelConfig()

    base = _standardize_panel_keys(base_panel, config)
    cols = _standardize_panel_keys(new_columns, config)

    key_cols = [config.stock_col, config.date_col]
    _validate_required_columns(cols, key_cols, "new_columns")
    _validate_no_duplicate_keys(cols, config, "new_columns")

    candidate_cols = [c for c in cols.columns if c not in key_cols]
    existing = [c for c in candidate_cols if c in base.columns]

    if existing and not overwrite:
        raise ValueError(
            "new_columns contains columns already in base_panel. "
            f"Existing columns: {existing}. "
            "Pass overwrite=True to replace them."
        )

    if overwrite and existing:
        base = base.drop(columns=existing)

    merged = base.merge(cols, on=key_cols, how="left")
    return merged


# ---------------------------------------------------------------------
# Daily panel
# ---------------------------------------------------------------------


def load_tushare_daily_panel(config: BasePanelConfig) -> pd.DataFrame:
    """
    Load cleaned Tushare daily panel.

    The input is expected to already combine:
        daily
        daily_basic
        adj_factor

    Required fields are checked here.
    """

    path = Path(config.tushare_daily_path)
    if not path.exists():
        raise FileNotFoundError(f"Tushare daily panel not found: {path}")

    df = pd.read_parquet(path)
    df = _standardize_panel_keys(df, config)

    mask = (
        (df[config.date_col] >= pd.Timestamp(config.start_date))
        & (df[config.date_col] <= pd.Timestamp(config.end_date))
    )
    df = df.loc[mask].copy()

    _validate_required_columns(df, config.required_daily_cols, "tushare_daily_panel")
    _validate_no_duplicate_keys(df, config, "tushare_daily_panel")

    # Verify close_adj is working: most rows should differ from close (后复权生效)
    if "close_adj" in df.columns and "close" in df.columns:
        diverge = (abs(df["close_adj"] - df["close"]) > df["close"] * 0.01).mean()
        print(f"  close_adj divergence from close (>1%): {diverge:.1%}")
        if diverge < 0.5:
            print("  WARNING: close_adj may not be properly calculated (low divergence)")

    # Keep daily_source distinct from financial_source.
    if "daily_source" not in df.columns:
        df["daily_source"] = "tushare"

    return df.sort_values([config.stock_col, config.date_col]).reset_index(drop=True)


# ---------------------------------------------------------------------
# Financial quarterly panel
# ---------------------------------------------------------------------


def build_financial_quarterly_panel(config: BasePanelConfig) -> pd.DataFrame:
    """
    Build quarterly financial panel.

    Rules
    -----
    - CSMAR is the main source.
    - Tushare financial gaps are appended only when CSMAR lacks stock_id+accper.
    - Tushare disclosure dates override theoretical available dates.
    """

    csmar_fin = load_csmar_financial_quarterly(config)
    gap_fin = load_tushare_financial_gaps(config)

    if gap_fin.empty:
        financial = csmar_fin.copy()
    else:
        key_cols = [config.stock_col, "accper"]
        existing_keys = set(zip(csmar_fin[config.stock_col], csmar_fin["accper"]))
        is_new = ~gap_fin.apply(
            lambda row: (row[config.stock_col], row["accper"]) in existing_keys,
            axis=1,
        )
        gap_new = gap_fin.loc[is_new].copy()
        financial = pd.concat([csmar_fin, gap_new], ignore_index=True)

    financial = _standardize_financial_keys(financial, config)
    financial = apply_disclosure_dates(financial, config)

    # Sort and deduplicate. Prefer standard quarter-end labels over 01-01 duplicates.
    financial = _handle_accper_0101_duplicates(financial, config)

    financial = financial.sort_values(
        [config.stock_col, "accper", "available_date"]
    ).reset_index(drop=True)

    # Drop duplicated exact keys after cleaning.
    financial = financial.drop_duplicates(
        subset=[config.stock_col, "accper"],
        keep="last",
    ).reset_index(drop=True)

    return financial


def load_csmar_financial_quarterly(config: BasePanelConfig) -> pd.DataFrame:
    """
    Load and standardize CSMAR quarterly financials.

    This function intentionally inherits the old build_factor_panel.py mapping.
    """

    data_dir = Path(config.csmar_data_dir)

    bas = pd.read_csv(data_dir / "FS_Combas.csv")
    ins = pd.read_csv(data_dir / "FS_Comins.csv")
    scf = pd.read_csv(data_dir / "FS_Comscfi.csv")
    scf_direct = pd.read_csv(data_dir / "FS_Comscfd.csv")

    for table in (bas, ins, scf, scf_direct):
        table["Stkcd"] = table["Stkcd"].astype(str).str.strip().str.zfill(6)
        table["Accper"] = pd.to_datetime(table["Accper"], errors="coerce")
        if "Typrep" in table.columns:
            table.dropna(subset=["Accper"], inplace=True)

    bas = bas[bas["Typrep"] == "A"].copy()
    ins = ins[ins["Typrep"] == "A"].copy()
    scf = scf[scf["Typrep"] == "A"].copy()
    scf_direct = scf_direct[scf_direct["Typrep"] == "A"].copy()

    bas = bas.rename(
        columns={
            "Stkcd": config.stock_col,
            "Accper": "accper",
            "A001000000": "total_assets",
            "A002000000": "total_liabilities",
            "A003000000": "total_equity",
            "A003100000": "equity_parent",
            "A001100000": "current_assets",
            "A001200000": "noncurrent_assets",
            "A002100000": "current_liabilities",
            "A002200000": "noncurrent_liabilities",
            "A001101000": "cash",
            "A001111000": "receivables",
            "A001123000": "inventory",
            "A002107000": "payables_trade",
        }
    )

    ins = ins.rename(
        columns={
            "Stkcd": config.stock_col,
            "Accper": "accper",
            "B001100000": "revenue_total",
            "B001101000": "revenue",
            "B001201000": "cost_revenue",
            "B001209000": "sell_expense",
            "B001210000": "admin_expense",
            "B001216000": "rd_expense",
            "B001211000": "finance_expense",
            "B001300000": "operating_profit",
            "B002000000": "net_profit",
            "B002000101": "net_profit_parent",
            "B002100000": "income_tax",
            "B003000000": "eps_basic",
        }
    )

    scf = scf.rename(
        columns={
            "Stkcd": config.stock_col,
            "Accper": "accper",
            "D000100000": "cf_operating",
        }
    )

    scf_direct = scf_direct.rename(
        columns={
            "Stkcd": config.stock_col,
            "Accper": "accper",
            "C001000000": "cf_operating_direct",
            "C001001000": "cf_sales_cash",
            "C001020000": "cf_wages_cash",
            "C001021000": "cf_taxes_cash",
            "C002006000": "cf_capex",
            "C002007000": "cf_invest_paid",
            "C003002000": "cf_borrow",
            "C003005000": "cf_repay_debt",
            "C003006000": "cf_dividend_paid",
            "C003008000": "cf_equity_issue",
        }
    )

    bas_cols = _available_columns(
        bas,
        [
            config.stock_col,
            "accper",
            "total_assets",
            "total_liabilities",
            "total_equity",
            "equity_parent",
            "current_assets",
            "noncurrent_assets",
            "current_liabilities",
            "noncurrent_liabilities",
            "cash",
            "receivables",
            "inventory",
            "payables_trade",
        ],
    )

    ins_cols = _available_columns(
        ins,
        [
            config.stock_col,
            "accper",
            "revenue_total",
            "revenue",
            "cost_revenue",
            "sell_expense",
            "admin_expense",
            "rd_expense",
            "finance_expense",
            "operating_profit",
            "net_profit",
            "net_profit_parent",
            "income_tax",
            "eps_basic",
        ],
    )

    scf_cols = _available_columns(
        scf,
        [
            config.stock_col,
            "accper",
            "cf_operating",
        ],
    )

    scfd_cols = _available_columns(
        scf_direct,
        [
            config.stock_col,
            "accper",
            "cf_operating_direct",
            "cf_sales_cash",
            "cf_wages_cash",
            "cf_taxes_cash",
            "cf_capex",
            "cf_invest_paid",
            "cf_borrow",
            "cf_repay_debt",
            "cf_dividend_paid",
            "cf_equity_issue",
        ],
    )

    fin = bas[bas_cols].merge(
        ins[ins_cols],
        on=[config.stock_col, "accper"],
        how="inner",
    )
    fin = fin.merge(
        scf[scf_cols],
        on=[config.stock_col, "accper"],
        how="left",
    )
    fin = fin.merge(
        scf_direct[scfd_cols],
        on=[config.stock_col, "accper"],
        how="left",
    )

    if "cf_operating_direct" in fin.columns:
        fin["cf_operating"] = fin["cf_operating"].fillna(fin["cf_operating_direct"])
        fin = fin.drop(columns=["cf_operating_direct"])

    fin["financial_source"] = "csmar"
    fin["theoretical_available_date"] = shift_to_theoretical_available_date(fin["accper"])

    # Only filter end date — keep history needed for TTM warm-up (4 prior quarters)
    end_ts = pd.Timestamp(config.end_date)
    fin = fin[fin["accper"] <= end_ts].copy()

    return fin


def load_tushare_financial_gaps(config: BasePanelConfig) -> pd.DataFrame:
    """
    Load Tushare financial gap rows and map them to standard columns.

    Tushare financial statement amounts are already in yuan.
    """

    fin_dir = Path(config.tushare_financial_dir)
    if not fin_dir.exists():
        return pd.DataFrame()

    paths = {
        "income": fin_dir / "income.parquet",
        "bs": fin_dir / "bs.parquet",
        "cf": fin_dir / "cf.parquet",
    }

    frames = []
    for path in paths.values():
        if path.exists():
            frames.append(pd.read_parquet(path))

    if not frames:
        return pd.DataFrame()

    ts = frames[0]
    for frame in frames[1:]:
        ts = ts.merge(
            frame,
            on=["_sid", "_period"],
            how="outer",
            suffixes=("", "_dup"),
        )
        dup_cols = [col for col in ts.columns if col.endswith("_dup")]
        if dup_cols:
            ts = ts.drop(columns=dup_cols)

    mappings = {
        # BS
        "total_assets": "total_assets",
        "total_liab": "total_liabilities",
        "total_hldr_eqy_exc_min_int": "equity_parent",
        "total_hldr_eqy_inc_min_int": "total_equity",
        "total_cur_assets": "current_assets",
        "total_nca": "noncurrent_assets",
        "total_cur_liab": "current_liabilities",
        "total_ncl": "noncurrent_liabilities",
        "inventories": "inventory",
        "accounts_receiv": "receivables",
        "acct_payable": "payables_trade",
        "money_cap": "cash",
        # IS
        "total_revenue": "revenue_total",
        "revenue": "revenue",
        "n_income_attr_p": "net_profit_parent",
        "n_income": "net_profit",
        "operate_profit": "operating_profit",
        "sell_exp": "sell_expense",
        "admin_exp": "admin_expense",
        "fin_exp": "finance_expense",
        "rd_exp": "rd_expense",
        "income_tax": "income_tax",
        "basic_eps": "eps_basic",
        # CF
        "n_cashflow_act": "cf_operating",
        "c_fr_sale_sg": "cf_sales_cash",
        "c_pay_acq_const_fiolta": "cf_capex",
        "c_paid_to_for_empl": "cf_wages_cash",
    }

    out = pd.DataFrame()
    out[config.stock_col] = ts["_sid"].astype(str).str.strip().str.zfill(6)
    out["accper"] = pd.to_datetime(ts["_period"], errors="coerce")

    for source_col, target_col in mappings.items():
        if source_col in ts.columns:
            out[target_col] = pd.to_numeric(ts[source_col], errors="coerce")

    out["financial_source"] = "tushare_gap"
    out["theoretical_available_date"] = shift_to_theoretical_available_date(out["accper"])

    return out.dropna(subset=[config.stock_col, "accper"]).reset_index(drop=True)


def apply_disclosure_dates(
    financial: pd.DataFrame,
    config: BasePanelConfig,
) -> pd.DataFrame:
    """
    Apply Tushare disclosure dates.

    available_date = disclosure_date if exists, otherwise theoretical_available_date.
    """

    fin = _standardize_financial_keys(financial, config)

    disclosure_path = Path(config.disclosure_dates_path)
    if not disclosure_path.exists():
        fin["disclosure_date"] = pd.NaT
        fin["available_date"] = fin["theoretical_available_date"]
        fin["disclosure_source"] = "theoretical_fallback"
        return fin

    dd = pd.read_parquet(disclosure_path)
    dd[config.stock_col] = dd[config.stock_col].astype(str).str.strip().str.zfill(6)
    dd["accper"] = pd.to_datetime(dd["end_date"], errors="coerce")
    dd["disclosure_date"] = pd.to_datetime(dd["disclosure_date"], errors="coerce")

    dd = dd[[config.stock_col, "accper", "disclosure_date"]].dropna(
        subset=[config.stock_col, "accper"]
    )
    dd = dd.drop_duplicates(subset=[config.stock_col, "accper"], keep="last")

    fin = fin.merge(dd, on=[config.stock_col, "accper"], how="left")

    has_disclosure = fin["disclosure_date"].notna()
    fin["available_date"] = fin["theoretical_available_date"]
    fin.loc[has_disclosure, "available_date"] = fin.loc[
        has_disclosure,
        "disclosure_date",
    ]
    fin["disclosure_source"] = np.where(
        has_disclosure,
        "tushare",
        "theoretical_fallback",
    )

    fin["available_date"] = pd.to_datetime(fin["available_date"], errors="coerce")
    return fin


def shift_to_theoretical_available_date(accper: pd.Series) -> pd.Series:
    """
    Map accounting period end to conservative theoretical available date.

    Standard rules:
        Q1 03-31 -> 05-01
        Q2 06-30 -> 09-01
        Q3 09-30 -> 11-01
        Q4 12-31 -> next year 05-01

    Non-standard periods use accper + 4 months as fallback.
    Accper=01-01 is handled elsewhere as duplicate Q4 label.
    """

    accper = pd.to_datetime(accper, errors="coerce")
    month = accper.dt.month
    year = accper.dt.year

    result = accper.copy()

    result.loc[month == 3] = pd.to_datetime(year[month == 3].astype(str) + "-05-01")
    result.loc[month == 6] = pd.to_datetime(year[month == 6].astype(str) + "-09-01")
    result.loc[month == 9] = pd.to_datetime(year[month == 9].astype(str) + "-11-01")
    result.loc[month == 12] = pd.to_datetime(
        (year[month == 12] + 1).astype(str) + "-05-01"
    )

    non_standard = ~month.isin([3, 6, 9, 12])
    result.loc[non_standard] = accper.loc[non_standard] + pd.DateOffset(months=4)

    return result


def _handle_accper_0101_duplicates(
    financial: pd.DataFrame,
    config: BasePanelConfig,
) -> pd.DataFrame:
    """
    Handle CSMAR Accper=01-01 duplicate labels.

    These rows duplicate previous-year Q4 values and must not participate
    as independent quarterly records.
    """

    fin = financial.copy()
    fin["is_accper_0101_duplicate"] = (
        (fin["accper"].dt.month == 1)
        & (fin["accper"].dt.day == 1)
    )

    n_dup = int(fin["is_accper_0101_duplicate"].sum())
    if n_dup > 0:
        print(f"  Accper=01-01 duplicate labels detected: {n_dup:,}")

    if config.drop_accper_0101_duplicates:
        fin = fin[~fin["is_accper_0101_duplicate"]].copy()

    return fin


# ---------------------------------------------------------------------
# Forward fill
# ---------------------------------------------------------------------


def forward_fill_financial_to_daily(
    daily: pd.DataFrame,
    financial: pd.DataFrame,
    config: BasePanelConfig,
) -> pd.DataFrame:
    """
    Forward-fill quarterly financials onto daily panel by available_date.

    For each stock and trading day, use the latest financial record whose
    available_date <= time.
    """

    daily = _standardize_panel_keys(daily, config)
    financial = _standardize_financial_keys(financial, config)

    audit_cols_drop_for_daily = {
        "disclosure_date",
        "theoretical_available_date",
        "disclosure_source",
        "is_accper_0101_duplicate",
    }

    fin_cols = [
        col for col in financial.columns
        if col not in {config.stock_col, "available_date"}
        and col not in audit_cols_drop_for_daily
    ]

    daily = daily.sort_values([config.stock_col, config.date_col]).reset_index(drop=True)
    financial = financial.sort_values([config.stock_col, "available_date"]).reset_index(drop=True)

    result_parts = []

    for stock_id, group in daily.groupby(config.stock_col, sort=False):
        group = group.copy()
        stock_fin = financial[financial[config.stock_col] == stock_id]

        if stock_fin.empty:
            for col in fin_cols:
                if col not in group.columns:
                    group[col] = np.nan
            result_parts.append(group)
            continue

        stock_fin = stock_fin.sort_values("available_date").reset_index(drop=True)
        stock_dates = group[config.date_col].to_numpy()
        fin_dates = stock_fin["available_date"].to_numpy()

        idx = np.searchsorted(fin_dates, stock_dates, side="right") - 1
        valid = idx >= 0

        for col in fin_cols:
            src = stock_fin[col]
            numeric = pd.api.types.is_numeric_dtype(src)
            values = np.full(len(group), np.nan, dtype=np.float64 if numeric else object)
            source_values = src.to_numpy(dtype=np.float64 if numeric else object, na_value=np.nan)
            values[valid] = source_values[idx[valid]]
            group[col] = values

        result_parts.append(group)

    panel = pd.concat(result_parts, ignore_index=True)
    panel = panel.sort_values([config.stock_col, config.date_col]).reset_index(drop=True)

    return panel


def _merge_list_date(
    panel: pd.DataFrame,
    config: BasePanelConfig,
) -> pd.DataFrame:
    """Merge listing date from CSMAR TRD_Co.csv."""
    csmar_dir = Path(config.csmar_data_dir)
    co_path = csmar_dir / "TRD_Co.csv"
    if not co_path.exists():
        print(f"  TRD_Co.csv not found at {co_path}, skipping list_date")
        panel["list_date"] = pd.NaT
        return panel

    co = pd.read_csv(co_path, low_memory=False)
    co["_sid"] = co["Stkcd"].astype(str).str.strip().str.zfill(6)
    co["_list"] = pd.to_datetime(co["Listdt"], errors="coerce")
    ld = co.groupby("_sid")["_list"].first()

    panel["list_date"] = panel[config.stock_col].map(ld).where(
        lambda x: x.notna(), pd.NaT
    )
    match = panel["list_date"].notna().mean()
    print(f"  list_date match rate: {match:.1%}")
    return panel


def _merge_industry_metadata(
    panel: pd.DataFrame,
    config: BasePanelConfig,
) -> pd.DataFrame:
    """Merge 申万 industry classification from Aindustry.xlsx if available."""
    if config.industry_path is None:
        return panel

    industry_path = Path(config.industry_path)
    if not industry_path.exists():
        print(f"  Industry file not found: {industry_path}")
        return panel

    xl = pd.read_excel(industry_path)
    xl["_code"] = xl["stock_code"].astype(str).str.strip().str.zfill(6)
    panel["_code"] = panel[config.stock_col].astype(str).str.strip().str.zfill(6)

    if "first_level_code" in xl.columns:
        ind_map = xl.groupby("_code")["first_level_code"].first()
        panel["industry_sw"] = panel["_code"].map(ind_map).fillna(-1).astype(int)
        match_rate = (panel["industry_sw"] != -1).mean()
        print(f"  industry_sw match rate: {match_rate:.1%}")

    panel = panel.drop(columns=["_code"], errors="ignore")
    return panel


# ---------------------------------------------------------------------
# Checks and helpers
# ---------------------------------------------------------------------


def run_base_panel_checks(
    panel: pd.DataFrame,
    financial: pd.DataFrame,
    config: BasePanelConfig,
    is_incremental: bool = False,
) -> dict[str, Any]:
    """Run base panel quality checks and return audit dict."""

    audit: dict[str, Any] = {}

    _validate_no_duplicate_keys(panel, config, "base_panel")
    _validate_required_columns(panel, config.required_daily_cols, "base_panel")

    audit["is_incremental"] = is_incremental
    audit["n_rows"] = int(len(panel))
    audit["n_stocks"] = int(panel[config.stock_col].nunique())
    audit["start_date"] = str(panel[config.date_col].min())
    audit["end_date"] = str(panel[config.date_col].max())

    daily_missing = {}
    for col in config.required_daily_cols:
        if col in panel.columns:
            daily_missing[col] = float(panel[col].isna().mean())
    audit["required_daily_missing_rate"] = daily_missing

    if "pct_chg" in panel.columns and "ret_daily" in panel.columns:
        diff = (panel["ret_daily"] - panel["pct_chg"] / 100.0).abs()
        audit["ret_pct_chg_max_abs_diff"] = float(diff.dropna().max())

    if "adj_factor" in panel.columns and "close_adj" in panel.columns:
        diff = (panel["close_adj"] - panel["close"] * panel["adj_factor"]).abs()
        audit["close_adj_check_max_abs_diff"] = float(diff.dropna().max())

    if "mktcap_total" in panel.columns:
        audit["mktcap_total_non_positive"] = int((panel["mktcap_total"] <= 0).sum())

    if "mktcap_float" in panel.columns:
        audit["mktcap_float_non_positive"] = int((panel["mktcap_float"] <= 0).sum())

    if "available_date" in financial.columns and "accper" in financial.columns:
        bad = financial["available_date"] <= financial["accper"]
        audit["financial_available_date_le_accper"] = int(bad.sum())

    key_fin_cols = [
        "total_assets",
        "total_liabilities",
        "equity_parent",
        "revenue_total",
        "net_profit_parent",
        "cf_operating",
    ]
    fin_cov = {}
    for col in key_fin_cols:
        if col in panel.columns:
            fin_cov[col] = float(panel[col].notna().mean())
    audit["financial_daily_coverage"] = fin_cov

    if "industry_sw" in panel.columns:
        audit["industry_sw_coverage"] = float(panel["industry_sw"].notna().mean())

    return audit


def _standardize_panel_keys(
    df: pd.DataFrame,
    config: BasePanelConfig,
) -> pd.DataFrame:
    out = df.copy()
    out[config.stock_col] = out[config.stock_col].astype(str).str.strip().str.zfill(6)
    out[config.date_col] = pd.to_datetime(out[config.date_col], errors="raise").dt.normalize()
    return out


def _standardize_financial_keys(
    df: pd.DataFrame,
    config: BasePanelConfig,
) -> pd.DataFrame:
    out = df.copy()
    out[config.stock_col] = out[config.stock_col].astype(str).str.strip().str.zfill(6)
    out["accper"] = pd.to_datetime(out["accper"], errors="raise").dt.normalize()

    if "available_date" in out.columns:
        out["available_date"] = pd.to_datetime(
            out["available_date"],
            errors="coerce",
        ).dt.normalize()

    if "theoretical_available_date" in out.columns:
        out["theoretical_available_date"] = pd.to_datetime(
            out["theoretical_available_date"],
            errors="coerce",
        ).dt.normalize()

    return out


def _validate_required_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _validate_no_duplicate_keys(
    df: pd.DataFrame,
    config: BasePanelConfig,
    name: str,
) -> None:
    key_cols = [config.stock_col, config.date_col]
    dup = df.duplicated(subset=key_cols)
    if dup.any():
        sample = df.loc[dup, key_cols].head(10)
        raise ValueError(
            f"{name} contains duplicated stock-date rows: {int(dup.sum())}. "
            f"Sample:\n{sample}"
        )


def _available_columns(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [col for col in columns if col in df.columns]