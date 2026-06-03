"""
Strategy V1 portfolio-construction experiments.

This script starts from an already-selected ensemble score and evaluates a small
set of lightweight strategy variants:

1. TopN sensitivity
2. Holding buffer / hysteresis
3. Industry exposure cap
4. Optional bull/bear dynamic TopN if index data is available

Important conventions:
    - The input score is a signal-date score, usually produced by ensemble.py.
    - Daily portfolio return is computed from the next available return_1d date.
    - Transaction cost follows the existing project convention:
          net_return = raw_return - turnover * (buy_cost + sell_cost) / 2
      with first-day turnover set to 0.
    - Industry cap values are percentages of portfolio weight. Since the
      portfolio is equal-weighted, a 10% cap with Top50 means max 5 holdings
      per industry.

Default inputs:
    reports/strategy_v1/best_ridge_score_test.parquet
    dataset/processed/unified_daily_panel.parquet
    dataset/processed/factor_panel_54_ind.parquet

Outputs:
    reports/strategy_v1/strategy_results.csv
    reports/strategy_v1/strategy_summary.txt
    reports/strategy_v1/strategy_daily_returns.csv
    reports/strategy_v1/strategy_holdings.parquet
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyConfig:
    score_test_path: Path = Path("reports/strategy_v1/best_ridge_score_test.parquet")
    score_valid_path: Path | None = None
    score_col: str = "_best_ridge"
    returns_path: Path = Path("dataset/processed/unified_daily_panel.parquet")
    factor_path: Path = Path("dataset/processed/factor_panel_54_ind.parquet")
    output_dir: Path = Path("reports/strategy_v1")

    topn_list: tuple[int, ...] = (50, 100, 150)
    default_top_n: int = 100
    buffer_rank: int = 150
    industry_cap_pcts: tuple[float, ...] = (0.0, 10.0, 20.0)

    apply_limit_filter: bool = True
    buy_cost: float = 0.0003
    sell_cost: float = 0.0008

    # Ensemble baseline path for comparison rows in the output summary.
    ensemble_results_path: Path = Path("reports/strategy_v1/ensemble_results.csv")

    # Optional bull/bear experiment. If the file does not exist, it is skipped.
    index_path: Path = Path("dataset/input/original_data/TRD_Index.csv")
    index_code: int = 300
    bull_lookback: int = 60
    bull_top_n: int | None = 100
    bear_top_n: int = 60
    run_bull_bear: bool = True

    def __post_init__(self) -> None:
        if not self.topn_list:
            raise ValueError("topn_list must be non-empty")
        if any(n <= 0 for n in self.topn_list):
            raise ValueError(f"topn_list must be positive, got {self.topn_list}")
        if self.default_top_n <= 0:
            raise ValueError(f"default_top_n must be positive, got {self.default_top_n}")
        if self.buffer_rank < self.default_top_n:
            raise ValueError("buffer_rank should be >= default_top_n")
        if any(p < 0 for p in self.industry_cap_pcts):
            raise ValueError(f"industry_cap_pcts must be non-negative, got {self.industry_cap_pcts}")


# ---------------------------------------------------------------------------
# IO and normalization
# ---------------------------------------------------------------------------


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"]).dt.normalize()
    out["stock_id"] = out["stock_id"].astype(str).str.strip().str.zfill(6)
    return out


def load_score_file(path: Path, score_col: str, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing score file: {path}")
    df = pd.read_parquet(path)
    required = {"time", "stock_id", score_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    keep_cols = ["time", "stock_id", score_col]
    if "y_true" in df.columns:
        keep_cols.append("y_true")
    out = normalize_keys(df[keep_cols])
    out = out.rename(columns={score_col: "score"})
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out["split"] = split
    out = out.dropna(subset=["score"]).sort_values(["time", "stock_id"]).reset_index(drop=True)
    return out


def load_scores(config: StrategyConfig) -> dict[str, pd.DataFrame]:
    scores = {"test": load_score_file(config.score_test_path, config.score_col, "test")}
    if config.score_valid_path is not None:
        scores["valid"] = load_score_file(config.score_valid_path, config.score_col, "valid")
    return scores


def load_returns(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing returns file: {path}")
    panel = pd.read_parquet(path)
    required = {"time", "stock_id", "ret_daily"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    ret = panel[["time", "stock_id", "ret_daily"]].rename(columns={"ret_daily": "return_1d"}).copy()
    ret = normalize_keys(ret)
    ret["return_1d"] = pd.to_numeric(ret["return_1d"], errors="coerce")
    ret = ret.dropna(subset=["return_1d"])
    ret = ret.drop_duplicates(["time", "stock_id"], keep="last")
    return ret.sort_values(["time", "stock_id"]).reset_index(drop=True)


def load_metadata(path: Path, apply_limit_filter: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing factor/metadata file: {path}")
    cols_needed = [
        "time", "stock_id", "industry_sw", "limit_status", "amount", "mktcap_total",
        "F046LIMITUP20", "F047LIMITDN20", "F048LIMITSTREAKUP",
    ]
    panel = pd.read_parquet(path)
    existing = [c for c in cols_needed if c in panel.columns]
    missing_core = {"time", "stock_id", "industry_sw"} - set(existing)
    if missing_core:
        raise ValueError(f"{path} is missing core metadata columns: {sorted(missing_core)}")
    if apply_limit_filter:
        missing_limit = {"F046LIMITUP20", "F047LIMITDN20", "F048LIMITSTREAKUP"} - set(existing)
        if missing_limit:
            raise ValueError(f"{path} is missing limit-filter columns: {sorted(missing_limit)}")

    meta = normalize_keys(panel[existing])
    meta["industry_sw"] = meta["industry_sw"].fillna("UNKNOWN").astype(str)
    for c in ["F046LIMITUP20", "F047LIMITDN20", "F048LIMITSTREAKUP"]:
        if c not in meta.columns:
            meta[c] = 0.0
        meta[c] = pd.to_numeric(meta[c], errors="coerce").fillna(0.0)
    meta = meta.drop_duplicates(["time", "stock_id"], keep="last")
    return meta.sort_values(["time", "stock_id"]).reset_index(drop=True)


def attach_metadata_and_returns(
    score_df: pd.DataFrame,
    meta: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Merge metadata and next-day realized returns into score rows.

    The returned `time` is still signal date. `return_date` is the next available
    return date used for realized return_1d.
    """
    df = score_df.merge(meta, on=["time", "stock_id"], how="left", validate="many_to_one")

    ret_dates = pd.DatetimeIndex(returns["time"].dropna().unique()).sort_values()
    signal_dates = pd.DatetimeIndex(df["time"].dropna().unique()).sort_values()
    next_map: dict[pd.Timestamp, pd.Timestamp] = {}
    for sd in signal_dates:
        idx = ret_dates.searchsorted(sd, side="right")
        if idx < len(ret_dates):
            next_map[pd.Timestamp(sd)] = pd.Timestamp(ret_dates[idx])
    df["return_date"] = df["time"].map(next_map)
    df = df.dropna(subset=["return_date"])
    df["return_date"] = pd.to_datetime(df["return_date"]).dt.normalize()

    ret = returns.rename(columns={"time": "return_date"})
    df = df.merge(ret, on=["return_date", "stock_id"], how="inner", validate="many_to_one")
    return df.sort_values(["time", "stock_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Candidate filters and portfolio selection
# ---------------------------------------------------------------------------


def apply_limit_filter(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (df["F046LIMITUP20"] <= 0)
        & (df["F047LIMITDN20"] <= 0)
        & (df["F048LIMITSTREAKUP"] <= 0)
    )
    return df[mask].copy()


def topn_select(day: pd.DataFrame, top_n: int) -> list[str]:
    return day.sort_values("score", ascending=False).head(top_n)["stock_id"].tolist()


def industry_cap_max_count(top_n: int, industry_cap_pct: float) -> int | None:
    if industry_cap_pct <= 0:
        return None
    return max(1, int(np.floor(top_n * industry_cap_pct / 100.0)))


def topn_with_industry_cap(day: pd.DataFrame, top_n: int, industry_cap_pct: float) -> list[str]:
    max_count = industry_cap_max_count(top_n, industry_cap_pct)
    if max_count is None:
        return topn_select(day, top_n)

    selected: list[str] = []
    counts: dict[str, int] = {}
    ranked = day.sort_values("score", ascending=False)
    for row in ranked.itertuples(index=False):
        industry = str(getattr(row, "industry_sw", "UNKNOWN"))
        if counts.get(industry, 0) >= max_count:
            continue
        selected.append(row.stock_id)
        counts[industry] = counts.get(industry, 0) + 1
        if len(selected) >= top_n:
            break
    return selected


def buffer_select(
    day: pd.DataFrame,
    previous_holdings: set[str],
    target_top_n: int,
    buffer_rank: int,
) -> list[str]:
    ranked = day.sort_values("score", ascending=False).reset_index(drop=True)
    ranked["rank_no"] = np.arange(1, len(ranked) + 1)
    eligible = set(ranked["stock_id"])

    keep = []
    if previous_holdings:
        keep_df = ranked[(ranked["stock_id"].isin(previous_holdings)) & (ranked["rank_no"] <= buffer_rank)]
        keep = keep_df.sort_values("score", ascending=False)["stock_id"].tolist()

    selected = list(dict.fromkeys(keep))[:target_top_n]
    selected_set = set(selected)
    for sid in ranked["stock_id"]:
        if sid in selected_set:
            continue
        selected.append(sid)
        selected_set.add(sid)
        if len(selected) >= target_top_n:
            break
    return [sid for sid in selected if sid in eligible]


def equal_weights(stock_ids: list[str]) -> dict[str, float]:
    if not stock_ids:
        return {}
    w = 1.0 / len(stock_ids)
    return {sid: w for sid in stock_ids}


# ---------------------------------------------------------------------------
# Lightweight custom backtest
# ---------------------------------------------------------------------------


Selector = Callable[[pd.DataFrame, set[str]], list[str]]


def calculate_turnover(prev_w: dict[str, float], cur_w: dict[str, float], is_first: bool) -> float:
    if is_first:
        return 0.0
    keys = set(prev_w) | set(cur_w)
    raw = float(sum(abs(cur_w.get(k, 0.0) - prev_w.get(k, 0.0)) for k in keys))
    return raw / 2.0


def run_custom_backtest(
    data: pd.DataFrame,
    strategy_name: str,
    selector: Selector,
    *,
    buy_cost: float,
    sell_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run a custom equal-weight backtest from signal-date candidates.

    Parameters
    ----------
    data:
        DataFrame with signal-date rows and already merged return_1d for the
        next available return date.
    selector:
        Function taking (day_df, previous_holdings_set) and returning selected
        stock_id list for the current signal date.
    """
    rows: list[dict] = []
    holdings_rows: list[dict] = []
    prev_weights: dict[str, float] = {}
    prev_holdings: set[str] = set()
    cost_rate = 0.5 * (buy_cost + sell_cost)

    grouped = data.sort_values(["time", "score"], ascending=[True, False]).groupby("time", sort=True)
    for i, (signal_date, day) in enumerate(grouped):
        selected = selector(day.copy(), prev_holdings)
        cur_weights = equal_weights(selected)
        turnover = calculate_turnover(prev_weights, cur_weights, is_first=(i == 0))

        if cur_weights:
            ret_map = day.set_index("stock_id")["return_1d"].to_dict()
            raw_return = float(sum(w * ret_map.get(sid, 0.0) for sid, w in cur_weights.items()))
        else:
            raw_return = 0.0
        net_return = raw_return - turnover * cost_rate
        return_date = day["return_date"].iloc[0] if len(day) else pd.NaT

        rows.append({
            "strategy": strategy_name,
            "time": signal_date,
            "return_date": return_date,
            "raw_return": raw_return,
            "net_return": net_return,
            "turnover": turnover,
            "n_holdings": len(cur_weights),
        })
        for sid, weight in cur_weights.items():
            h = day.loc[day["stock_id"] == sid].head(1)
            industry = h["industry_sw"].iloc[0] if len(h) and "industry_sw" in h.columns else "UNKNOWN"
            score = h["score"].iloc[0] if len(h) else np.nan
            holdings_rows.append({
                "strategy": strategy_name,
                "time": signal_date,
                "return_date": return_date,
                "stock_id": sid,
                "weight": weight,
                "score": score,
                "industry_sw": industry,
            })

        prev_weights = cur_weights
        prev_holdings = set(cur_weights)

    daily = pd.DataFrame(rows)
    holdings = pd.DataFrame(holdings_rows)
    summary = summarize_daily_returns(daily, strategy_name)
    return daily, holdings, summary


def summarize_daily_returns(daily: pd.DataFrame, strategy_name: str) -> dict:
    if daily.empty:
        return {
            "strategy": strategy_name,
            "n_days": 0,
            "sharpe": np.nan,
            "annret": np.nan,
            "maxdd": np.nan,
            "nav": np.nan,
            "turnover": np.nan,
            "winrate": np.nan,
            "avg_n_holdings": np.nan,
        }
    ret = daily["net_return"].astype(float)
    nav = (1.0 + ret).cumprod()
    daily = daily.copy()
    daily["nav"] = nav

    ret_std = ret.std(ddof=1)
    sharpe = float(ret.mean() / ret_std * np.sqrt(252)) if ret_std and ret_std > 0 else 0.0
    annret = float(nav.iloc[-1] ** (252.0 / len(ret)) - 1.0) if len(ret) else np.nan
    maxdd = float((nav / nav.cummax() - 1.0).min()) if len(nav) else np.nan
    return {
        "strategy": strategy_name,
        "n_days": int(len(ret)),
        "sharpe": sharpe,
        "annret": annret,
        "maxdd": maxdd,
        "nav": float(nav.iloc[-1]),
        "turnover": float(daily["turnover"].mean()),
        "winrate": float((ret > 0).mean()),
        "avg_n_holdings": float(daily["n_holdings"].mean()),
    }


# ---------------------------------------------------------------------------
# Optional market regime
# ---------------------------------------------------------------------------


def load_bull_bear_map(index_path: Path, index_code: int, lookback: int) -> dict[pd.Timestamp, bool]:
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    idx = pd.read_csv(index_path)
    if {"Indexcd", "Trddt", "Clsindex"}.issubset(idx.columns):
        idx = idx[idx["Indexcd"] == index_code].copy()
        idx["time"] = pd.to_datetime(idx["Trddt"]).dt.normalize()
        idx["close"] = pd.to_numeric(idx["Clsindex"], errors="coerce")
    elif {"time", "close"}.issubset(idx.columns):
        idx = idx.copy()
        idx["time"] = pd.to_datetime(idx["time"]).dt.normalize()
        idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
    else:
        raise ValueError(
            f"{index_path} must contain either Indexcd/Trddt/Clsindex or time/close columns"
        )
    idx = idx.dropna(subset=["time", "close"]).sort_values("time")
    idx["bull"] = idx["close"].pct_change(lookback) > 0
    return idx.set_index("time")["bull"].to_dict()


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------


def prepare_split_data(
    score_df: pd.DataFrame,
    meta: pd.DataFrame,
    returns: pd.DataFrame,
    apply_limit: bool,
) -> pd.DataFrame:
    data = attach_metadata_and_returns(score_df, meta, returns)
    if apply_limit:
        before = len(data)
        data = apply_limit_filter(data)
        dropped = before - len(data)
        print(f"    limit filter dropped {dropped:,} rows ({dropped / before:.2%})" if before else "    no rows before limit filter")
    return data.sort_values(["time", "score"], ascending=[True, False]).reset_index(drop=True)


def evaluate_strategy_on_splits(
    split_data: dict[str, pd.DataFrame],
    strategy_name: str,
    selector_factory: Callable[[str], Selector],
    config: StrategyConfig,
) -> tuple[list[dict], list[pd.DataFrame], list[pd.DataFrame]]:
    summaries: list[dict] = []
    daily_frames: list[pd.DataFrame] = []
    holdings_frames: list[pd.DataFrame] = []
    for split, data in split_data.items():
        selector = selector_factory(split)
        daily, holdings, summary = run_custom_backtest(
            data,
            strategy_name=strategy_name,
            selector=selector,
            buy_cost=config.buy_cost,
            sell_cost=config.sell_cost,
        )
        summary["split"] = split
        summaries.append(summary)
        daily["split"] = split
        holdings["split"] = split
        daily_frames.append(daily)
        holdings_frames.append(holdings)
    return summaries, daily_frames, holdings_frames


def choose_best_topn(results: pd.DataFrame, default_top_n: int) -> int:
    valid = results[(results["family"] == "topn") & (results["split"] == "valid")].copy()
    if valid.empty:
        return default_top_n
    best = valid.sort_values("sharpe", ascending=False).iloc[0]
    return int(best["top_n"])


def load_ensemble_baseline(path: Path) -> pd.DataFrame:
    """Load ensemble result rows as baseline family for comparison."""
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    keep_sections = {"single", "equal_weight_raw", "equal_weight_rank", "rank_ridge", "ridge_raw"}
    base = raw[raw["section"].isin(keep_sections) & (raw["split"] == "test")].copy()
    if base.empty:
        return pd.DataFrame()
    base = base.rename(columns={
        "sharpe": "sharpe", "maxdd": "maxdd", "turnover": "turnover",
        "nav": "nav", "winrate": "winrate",
    })
    base["family"] = "baseline"
    base["strategy"] = base["section"] + "__" + base["strategy"] + "__w" + base["window"].astype(int).astype(str)
    base["split"] = "test"
    base["top_n"] = float("nan")
    base["buffer_rank"] = float("nan")
    base["industry_cap_pct"] = float("nan")
    base["avg_n_holdings"] = 50.0
    base["annret"] = float("nan")
    cols = ["family", "strategy", "split", "sharpe", "annret", "maxdd", "nav",
            "turnover", "winrate", "avg_n_holdings", "top_n", "buffer_rank", "industry_cap_pct"]
    return base[[c for c in cols if c in base.columns]]


def build_strategy_experiments(config: StrategyConfig, split_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    all_summaries: list[dict] = []
    all_daily: list[pd.DataFrame] = []
    all_holdings: list[pd.DataFrame] = []

    # 1) TopN sensitivity.
    print("\n[1/4] TopN sensitivity...")
    for top_n in config.topn_list:
        strategy_name = f"Top{top_n}"

        def factory(_split: str, n: int = top_n) -> Selector:
            return lambda day, _prev: topn_select(day, n)

        summaries, daily_frames, holdings_frames = evaluate_strategy_on_splits(
            split_data, strategy_name, factory, config
        )
        for s in summaries:
            s.update({"family": "topn", "top_n": top_n, "buffer_rank": np.nan, "industry_cap_pct": np.nan})
        all_summaries.extend(summaries)
        all_daily.extend(daily_frames)
        all_holdings.extend(holdings_frames)

    temp_results = pd.DataFrame(all_summaries)
    selected_top_n = choose_best_topn(temp_results, config.default_top_n)
    print(f"    selected_top_n={selected_top_n} (valid Sharpe if valid split exists; otherwise default)")

    # 2) Holding buffer.
    print("\n[2/4] Holding buffer...")
    buffer_strategy = f"Top{selected_top_n}_Buffer{config.buffer_rank}"

    def buffer_factory(_split: str) -> Selector:
        return lambda day, prev: buffer_select(day, prev, selected_top_n, config.buffer_rank)

    summaries, daily_frames, holdings_frames = evaluate_strategy_on_splits(
        split_data, buffer_strategy, buffer_factory, config
    )
    for s in summaries:
        s.update({"family": "buffer", "top_n": selected_top_n, "buffer_rank": config.buffer_rank, "industry_cap_pct": np.nan})
    all_summaries.extend(summaries)
    all_daily.extend(daily_frames)
    all_holdings.extend(holdings_frames)

    # 3) Industry cap, using selected TopN.
    print("\n[3/4] Industry cap...")
    for cap_pct in config.industry_cap_pcts:
        label = "NoCap" if cap_pct <= 0 else f"IndCap{cap_pct:g}pct"
        strategy_name = f"Top{selected_top_n}_{label}"

        def cap_factory(_split: str, pct: float = cap_pct) -> Selector:
            return lambda day, _prev: topn_with_industry_cap(day, selected_top_n, pct)

        summaries, daily_frames, holdings_frames = evaluate_strategy_on_splits(
            split_data, strategy_name, cap_factory, config
        )
        for s in summaries:
            s.update({"family": "industry_cap", "top_n": selected_top_n, "buffer_rank": np.nan, "industry_cap_pct": cap_pct})
        all_summaries.extend(summaries)
        all_daily.extend(daily_frames)
        all_holdings.extend(holdings_frames)

    # 4) Optional bull/bear dynamic TopN.
    print("\n[4/4] Bull/Bear dynamic TopN...")
    if config.run_bull_bear and config.index_path.exists():
        bull_map = load_bull_bear_map(config.index_path, config.index_code, config.bull_lookback)
        bull_top_n = config.bull_top_n or selected_top_n
        bear_top_n = config.bear_top_n
        strategy_name = f"Bull{bull_top_n}_Bear{bear_top_n}"

        def bb_factory(_split: str) -> Selector:
            def selector(day: pd.DataFrame, _prev: set[str]) -> list[str]:
                signal_date = pd.Timestamp(day["time"].iloc[0])
                bull = bool(bull_map.get(signal_date, True))
                n = bull_top_n if bull else bear_top_n
                return topn_select(day, n)
            return selector

        summaries, daily_frames, holdings_frames = evaluate_strategy_on_splits(
            split_data, strategy_name, bb_factory, config
        )
        for s in summaries:
            s.update({"family": "bull_bear", "top_n": np.nan, "buffer_rank": np.nan, "industry_cap_pct": np.nan})
            s["bull_top_n"] = bull_top_n
            s["bear_top_n"] = bear_top_n
        all_summaries.extend(summaries)
        all_daily.extend(daily_frames)
        all_holdings.extend(holdings_frames)
    else:
        print(f"    skipped: index file not found or disabled ({config.index_path})")

    results = pd.DataFrame(all_summaries)
    daily = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    holdings = pd.concat(all_holdings, ignore_index=True) if all_holdings else pd.DataFrame()
    return results, daily, holdings, selected_top_n


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def build_summary(results: pd.DataFrame, config: StrategyConfig, selected_top_n: int) -> str:
    lines: list[str] = []
    lines.append("Strategy V1 Portfolio Construction Summary")
    lines.append("==========================================")
    lines.append(f"Score test path:  {config.score_test_path}")
    lines.append(f"Score valid path: {config.score_valid_path if config.score_valid_path else '(not provided)'}")
    lines.append(f"Score column:     {config.score_col}")
    lines.append(f"Returns path:     {config.returns_path}")
    lines.append(f"Factor path:      {config.factor_path}")
    lines.append(f"Limit filter:     {config.apply_limit_filter}")
    lines.append(f"Costs:            buy={config.buy_cost:.4g}, sell={config.sell_cost:.4g}, avg={(config.buy_cost + config.sell_cost)/2:.4g}")
    lines.append(f"TopN list:        {list(config.topn_list)}")
    lines.append(f"Selected TopN:    {selected_top_n}")
    lines.append(f"Buffer rank:      {config.buffer_rank}")
    lines.append(f"Industry caps:    {list(config.industry_cap_pcts)} pct")
    lines.append("")
    lines.append("Conventions")
    lines.append("-----------")
    lines.append("- time is signal date; return_date is the next available trading date.")
    lines.append("- Portfolio is equal-weighted within selected holdings.")
    lines.append("- Industry cap is specified as portfolio weight percentage, implemented as max holdings per industry.")
    lines.append("- First-day turnover is set to 0, matching the existing project convention.")
    lines.append("- family=baseline rows are from ensemble.py evaluation for direct comparison.")
    lines.append("")

    cols = [
        "family", "strategy", "split", "sharpe", "annret", "maxdd", "nav", "turnover",
        "winrate", "avg_n_holdings", "top_n", "buffer_rank", "industry_cap_pct",
    ]
    shown = [c for c in cols if c in results.columns]
    lines.append("Results")
    lines.append("-------")
    if results.empty:
        lines.append("(no results)")
    else:
        lines.append(results[shown].sort_values(["family", "strategy", "split"]).to_string(index=False))
    lines.append("")
    return "\n".join(lines)


def print_results(results: pd.DataFrame) -> None:
    cols = ["family", "strategy", "split", "sharpe", "maxdd", "turnover", "nav", "avg_n_holdings"]
    shown = [c for c in cols if c in results.columns]
    with pd.option_context("display.width", 160, "display.max_rows", 200, "display.max_colwidth", 60):
        print("\n=== Strategy Results ===")
        print(results[shown].sort_values(["family", "strategy", "split"]).to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight Strategy V1 portfolio-construction variants.")
    parser.add_argument("--score-test-path", type=Path, default=Path("reports/strategy_v1/best_ridge_score_test.parquet"))
    parser.add_argument("--score-valid-path", type=Path, default=None)
    parser.add_argument("--score-col", type=str, default="_best_ridge")
    parser.add_argument("--returns-path", type=Path, default=Path("dataset/processed/unified_daily_panel.parquet"))
    parser.add_argument("--factor-path", type=Path, default=Path("dataset/processed/factor_panel_54_ind.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/strategy_v1"))
    parser.add_argument("--topn-list", type=parse_int_tuple, default=(50, 100, 150))
    parser.add_argument("--default-top-n", type=int, default=100)
    parser.add_argument("--buffer-rank", type=int, default=150)
    parser.add_argument("--industry-cap-pcts", type=parse_float_tuple, default=(0.0, 10.0, 20.0))
    parser.add_argument("--no-limit-filter", action="store_true")
    parser.add_argument("--buy-cost", type=float, default=0.0003)
    parser.add_argument("--sell-cost", type=float, default=0.0008)
    parser.add_argument("--index-path", type=Path, default=Path("dataset/input/original_data/TRD_Index.csv"))
    parser.add_argument("--index-code", type=int, default=300)
    parser.add_argument("--bull-lookback", type=int, default=60)
    parser.add_argument("--bull-top-n", type=int, default=100)
    parser.add_argument("--bear-top-n", type=int, default=60)
    parser.add_argument("--no-bull-bear", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StrategyConfig(
        score_test_path=args.score_test_path,
        score_valid_path=args.score_valid_path,
        score_col=args.score_col,
        returns_path=args.returns_path,
        factor_path=args.factor_path,
        output_dir=args.output_dir,
        topn_list=args.topn_list,
        default_top_n=args.default_top_n,
        buffer_rank=args.buffer_rank,
        industry_cap_pcts=args.industry_cap_pcts,
        apply_limit_filter=not args.no_limit_filter,
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
        index_path=args.index_path,
        index_code=args.index_code,
        bull_lookback=args.bull_lookback,
        bull_top_n=args.bull_top_n,
        bear_top_n=args.bear_top_n,
        run_bull_bear=not args.no_bull_bear,
        ensemble_results_path=Path("reports/strategy_v1/ensemble_results.csv"),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading scores...")
    scores = load_scores(config)
    for split, df in scores.items():
        print(f"  {split}: {len(df):,} rows, {df['time'].nunique()} dates")

    print("Loading returns and metadata...")
    returns = load_returns(config.returns_path)
    meta = load_metadata(config.factor_path, apply_limit_filter=config.apply_limit_filter)
    print(f"  returns: {len(returns):,} rows, {returns['time'].nunique()} dates")
    print(f"  metadata: {len(meta):,} rows, {meta['time'].nunique()} dates")

    print("Preparing split data...")
    split_data: dict[str, pd.DataFrame] = {}
    for split, score_df in scores.items():
        print(f"  {split}:")
        split_data[split] = prepare_split_data(score_df, meta, returns, config.apply_limit_filter)
        print(f"    prepared: {len(split_data[split]):,} rows, {split_data[split]['time'].nunique()} dates")

    results, daily, holdings, selected_top_n = build_strategy_experiments(config, split_data)

    baseline = load_ensemble_baseline(config.ensemble_results_path)
    if not baseline.empty:
        results = pd.concat([baseline, results], ignore_index=True)

    results_path = config.output_dir / "strategy_results.csv"
    summary_path = config.output_dir / "strategy_summary.txt"
    daily_path = config.output_dir / "strategy_daily_returns.csv"
    holdings_path = config.output_dir / "strategy_holdings.parquet"

    results.to_csv(results_path, index=False)
    daily.to_csv(daily_path, index=False)
    holdings.to_parquet(holdings_path, index=False)
    summary = build_summary(results, config, selected_top_n)
    summary_path.write_text(summary, encoding="utf-8")

    print_results(results)
    print(f"\nSaved: {results_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {daily_path}")
    print(f"Saved: {holdings_path}")


if __name__ == "__main__":
    main()
