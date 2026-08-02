"""
Tushare daily data puller — provides pull_trade_dates + pull_one_date.

Used by scripts.dataset.daily_update for incremental panel extension.

Usage:
    from src.pipeline.tushare_client import pull_trade_dates, pull_one_date
"""
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts

SLEEP = 0.35


def _load_dotenv_token() -> None:
    """Set TUSHARE_TOKEN from .env if not already set."""
    if os.environ.get("TUSHARE_TOKEN"):
        return
    module_dir = Path(__file__).resolve()
    env_path = next(
        (parent / ".env" for parent in module_dir.parents if (parent / ".env").exists()),
        module_dir.parents[2] / ".env",
    )
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("TUSHARE_TOKEN="):
            os.environ["TUSHARE_TOKEN"] = line.split("=", 1)[1].strip()
            break


_load_dotenv_token()
_TOKEN = os.environ.get("TUSHARE_TOKEN", "").strip()
pro = ts.pro_api(_TOKEN) if _TOKEN else ts.pro_api()


def pull_trade_dates(start: str, end: str) -> list[str]:
    """Return sorted list of open SSE trading dates between *start* and *end*.

    Parameters
    ----------
    start, end: "YYYYMMDD" strings (inclusive)
    """
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
    return sorted(cal[cal["is_open"] == 1]["cal_date"].tolist())


def pull_one_date(trade_date: str) -> pd.DataFrame | None:
    """Pull full daily row for every stock on one trading date.

    Merges ``daily`` + ``daily_basic`` + ``adj_factor`` into a single
    DataFrame whose columns match what :func:`append_base_panel_rows` expects.

    Returns None when Tushare returns no rows for the date.
    """
    d = pro.daily(trade_date=trade_date,
                  fields="ts_code,trade_date,open,high,low,close,pre_close,"
                         "vol,amount,pct_chg,ah_vol,ah_amount")
    time.sleep(SLEEP)
    db = pro.daily_basic(trade_date=trade_date)
    time.sleep(SLEEP)
    af = pro.adj_factor(trade_date=trade_date)
    time.sleep(SLEEP)

    if d is None or len(d) == 0:
        return None

    d = d.copy()
    d["stock_id"] = d["ts_code"].str.strip().str.split(".").str[0].str.zfill(6)

    out = pd.DataFrame({
        "stock_id": d["stock_id"],
        "time": pd.to_datetime(trade_date),
    })

    # OHLCV
    out["open"] = pd.to_numeric(d["open"], errors="coerce")
    out["close"] = pd.to_numeric(d["close"], errors="coerce")
    out["high"] = pd.to_numeric(d["high"], errors="coerce")
    out["low"] = pd.to_numeric(d["low"], errors="coerce")
    out["pre_close"] = pd.to_numeric(d["pre_close"], errors="coerce")
    out["volume"] = pd.to_numeric(d["vol"], errors="coerce") * 100
    out["amount"] = pd.to_numeric(d["amount"], errors="coerce") * 1000
    out["pct_chg"] = pd.to_numeric(d["pct_chg"], errors="coerce")
    out["change"] = out["close"] - out["pre_close"]

    # After-hours trading volume/amount (available from 2026-07-06 onward).
    # Match the canonical base-panel units: shares and CNY.
    if "ah_vol" in d.columns:
        out["ah_vol"] = pd.to_numeric(d["ah_vol"], errors="coerce") * 100
    if "ah_amount" in d.columns:
        out["ah_amount"] = pd.to_numeric(d["ah_amount"], errors="coerce") * 1000

    # ret_daily = close / pre_close - 1  (not pct_chg which can have rounding)
    out["ret_daily"] = out["close"] / out["pre_close"] - 1.0

    # close_adj starts from raw close; adj_factor overrides it
    out["close_adj"] = out["close"]
    if af is not None and len(af) > 0:
        af = af.copy()
        af["_sid"] = af["ts_code"].str.strip().str.split(".").str[0].str.zfill(6)
        af_map = dict(zip(af["_sid"], pd.to_numeric(af["adj_factor"], errors="coerce")))
        out["close_adj"] = out["close"] * out["stock_id"].map(af_map).fillna(1.0)

    # daily_basic → mktcap, turnover, valuation
    if db is not None and len(db) > 0:
        db = db.copy()
        db["_sid"] = db["ts_code"].str.strip().str.split(".").str[0].str.zfill(6)
        db_map = db.set_index("_sid")

        for tcol, ocol, mult in [
            ("total_mv", "mktcap_total", 10000),
            ("circ_mv", "mktcap_float", 10000),
            ("total_share", "total_share", 10000),
            ("float_share", "float_share", 10000),
        ]:
            if tcol in db_map.columns:
                out[ocol] = out["stock_id"].map(
                    lambda x, tc=tcol, m=mult: (
                        pd.to_numeric(db_map[tc].get(x, np.nan), errors="coerce") * m
                        if x in db_map.index else np.nan
                    )
                )

        for tcol in ["turnover_rate", "turnover_rate_f", "volume_ratio",
                     "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm"]:
            if tcol in db_map.columns:
                out[tcol] = out["stock_id"].map(
                    lambda x, tc=tcol: (
                        pd.to_numeric(db_map[tc].get(x, np.nan), errors="coerce")
                        if x in db_map.index else np.nan
                    )
                )

    # Market VW return — used by base panel / factor panel
    valid = out["ret_daily"].notna() & out["mktcap_total"].notna() & (out["mktcap_total"] > 0)
    if valid.any():
        out["mkt_ret_vw"] = np.average(
            out.loc[valid, "ret_daily"],
            weights=out.loc[valid, "mktcap_total"],
        )
    else:
        out["mkt_ret_vw"] = out["ret_daily"].mean() if out["ret_daily"].notna().any() else 0.0

    out["daily_source"] = "tushare"

    return out


def pull_index_daily(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Pull index daily data from Tushare.

    Returns a DataFrame with columns matching the existing zz500/1000 parquet
    files: ts_code, time, close, open, high, low, pre_close, change, pct_chg,
    vol, amount.  Sorted by trade_date ascending.

    Requires ≥2000 Tushare points.
    """
    if start_date > end_date:
        return None
    df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    time.sleep(SLEEP)
    if df is None or df.empty:
        return None
    df = df.rename(columns={"trade_date": "time"})
    df["time"] = pd.to_datetime(df["time"])
    keep = ["ts_code", "time", "close", "open", "high", "low",
            "pre_close", "change", "pct_chg", "vol", "amount"]
    return df[keep].sort_values("time").reset_index(drop=True)


if __name__ == "__main__":
    import sys

    test_dates = []
    if "--smoke" in sys.argv:
        from datetime import date
        test_dates = [date.today().strftime("%Y%m%d")]
    elif "--start" in sys.argv:
        i = sys.argv.index("--start")
        start = sys.argv[i + 1]
        end = sys.argv[i + 3] if len(sys.argv) > i + 3 and sys.argv[i + 2] == "--end" else start
        from src.pipeline.tushare_client import pull_trade_dates
        test_dates = pull_trade_dates(start, end)
    else:
        print("Usage: python -m src.pipeline.tushare_client --smoke")
        print("       python -m src.pipeline.tushare_client --start 20260611 --end 20260710")
        sys.exit()

    print(f"Testing {len(test_dates)} dates: {test_dates}")
    for td in test_dates:
        df = pull_one_date(td)
        if df is not None:
            print(f"  {td}: {len(df):,} rows, cols={list(df.columns)}")
        else:
            print(f"  {td}: no data")
