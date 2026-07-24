"""
Market regime detection — multi-horizon trend + market breadth + hysteresis.

The signal dated ``t`` uses information available at the close of ``t`` and is
intended for the post-close T execution assumed by this project.

Outputs:
    dataset/input/market_regime.csv   — scores and auditable hard states
    reports/pics/market_regime.png    — three-panel visualization

Usage:
    python -m scripts.strategy.market_regime
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeConfig:
    """V1 defaults. All windows are trading days."""

    trend_windows: tuple[int, int, int] = (20, 60, 120)
    trend_weights: tuple[float, float, float] = (0.20, 0.45, 0.35)
    breadth_weights: tuple[float, float, float] = (0.15, 0.35, 0.50)
    trend_share: float = 0.60
    breadth_share: float = 0.40

    # Causal normalization: today's value is compared only with prior values.
    norm_window: int = 756
    norm_min_periods: int = 60
    score_temperature: float = 2.0

    bull_entry: float = 0.50
    bear_entry: float = -0.35
    bull_confirm_days: int = 5
    bear_confirm_days: int = 3

    vol_window: int = 20
    vol_history: int = 756
    vol_min_periods: int = 60
    high_vol_entry_pct: float = 0.60
    high_vol_exit_pct: float = 0.40

    min_stock_observations: int = 20


def _validate_weights(values: tuple[float, ...], name: str) -> None:
    if not np.isclose(sum(values), 1.0):
        raise ValueError(f"{name} must sum to 1, got {sum(values):.6f}")


def _causal_robust_score(
    values: pd.Series,
    window: int,
    min_periods: int,
    temperature: float,
    neutral: float = 0.0,
) -> pd.Series:
    """Map a direction signal to [-1, 1] around an economic neutral point.

    Prior data estimates scale but never moves the zero point: zero return is
    neutral for momentum and 50% is neutral for breadth. Median absolute
    distance is much less sensitive than std to A-share crash days.
    """
    history = values.shift(1)
    absolute_distance = (history - neutral).abs()
    scale = absolute_distance.rolling(window, min_periods=min_periods).median()

    # A zero scale is possible in short/rounded series. Fall back to causal RMS.
    fallback = np.sqrt(
        ((history - neutral) ** 2).rolling(window, min_periods=min_periods).mean()
    )
    scale = scale.where(scale > 1e-12, fallback).where(lambda x: x > 1e-12)
    z = ((values - neutral) / scale).clip(-8.0, 8.0)
    return np.tanh(z / temperature).rename(values.name)


def _rolling_percentile(
    values: pd.Series, window: int, min_periods: int
) -> pd.Series:
    """Percentile of today's value relative to prior observations only."""
    arr = values.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    for i, value in enumerate(arr):
        if not np.isfinite(value):
            continue
        previous = arr[max(0, i - window):i]
        previous = previous[np.isfinite(previous)]
        if previous.size < min_periods:
            continue
        # Mid-rank gives stable behaviour when volatility values are tied.
        less = np.count_nonzero(previous < value)
        equal = np.count_nonzero(previous == value)
        out[i] = (less + 0.5 * equal) / previous.size
    return pd.Series(out, index=values.index, name="vol_percentile")


def _direction_hysteresis(score: pd.Series, cfg: RegimeConfig) -> pd.DataFrame:
    """Convert a continuous score into a sticky, asymmetric Bull/Bear state.

    NaN scores reset confirmation counters but do NOT flip an already-
    established state.  Only the initial warm-up period produces NaN labels.
    """
    state: int | None = None
    age = 0
    bull_run = 0
    bear_run = 0
    rows: list[tuple[float, float, float, float, float]] = []

    for value in score.to_numpy(dtype=float):
        switched = 0
        if not np.isfinite(value):
            # Data glitch — kill any pending confirmation streak but keep the
            # current direction so downstream ensemble weights stay stable.
            bull_run = 0
            bear_run = 0
            if state is None:
                rows.append((np.nan, np.nan, 0.0, 0.0, 0.0))
            else:
                age += 1
                rows.append((float(state), float(age), 0.0, 0.0, 0.0))
            continue

        if state is None:
            state = int(value > 0.0)
            age = 1
            switched = 0
        else:
            bull_run = bull_run + 1 if value > cfg.bull_entry else 0
            bear_run = bear_run + 1 if value < cfg.bear_entry else 0

            if state == 0 and bull_run >= cfg.bull_confirm_days:
                state, age, switched = 1, 1, 1
                bull_run = bear_run = 0
            elif state == 1 and bear_run >= cfg.bear_confirm_days:
                state, age, switched = 0, 1, 1
                bull_run = bear_run = 0
            else:
                age += 1

        rows.append((float(state), float(age), float(switched), bull_run, bear_run))

    return pd.DataFrame(
        rows,
        index=score.index,
        columns=["bull", "state_age", "switch_flag", "pending_bull_days", "pending_bear_days"],
    )


def _vol_hysteresis(percentile: pd.Series, cfg: RegimeConfig) -> pd.Series:
    state: int | None = None
    output: list[float] = []
    for value in percentile.to_numpy(dtype=float):
        if not np.isfinite(value):
            output.append(np.nan)
            continue
        if state is None:
            state = int(value > 0.5)
        elif state == 0 and value > cfg.high_vol_entry_pct:
            state = 1
        elif state == 1 and value < cfg.high_vol_exit_pct:
            state = 0
        output.append(float(state))
    return pd.Series(output, index=percentile.index, name="high_vol")


def build_regime(
    panel_path: str = "dataset/processed/factor_panel_1500_54_ind.parquet",
    warmup_panel_path: str = "dataset/processed/unified_daily_panel.parquet",
    stock_col: str = "stock_id",
    config: RegimeConfig | None = None,
    # Legacy keyword arguments remain accepted for callers of the old function.
    ma_window: int | None = None,
    vol_window: int | None = None,
) -> pd.DataFrame:
    """Build causal market-direction and volatility-regime signals.

    The *panel_path* defines the trading universe (ZZ500+1000, 1500 stocks).
    *warmup_panel_path* provides earlier data (2012–2015, broader universe) to
    pre-warm the causal normalization so regime scores are valid from the first
    day of the trading-universe panel.
    """
    cfg = config or RegimeConfig()
    if ma_window is not None or vol_window is not None:
        cfg_values = dict(cfg.__dict__)
        if ma_window is not None:
            cfg_values["trend_windows"] = (20, int(ma_window), 120)
        if vol_window is not None:
            cfg_values["vol_window"] = int(vol_window)
        cfg = RegimeConfig(**cfg_values)

    _validate_weights(cfg.trend_weights, "trend_weights")
    _validate_weights(cfg.breadth_weights, "breadth_weights")
    if not np.isclose(cfg.trend_share + cfg.breadth_share, 1.0):
        raise ValueError("trend_share + breadth_share must equal 1")

    required = ["time", stock_col, "ret_daily"]

    # ── Main panel (trading universe) ──
    fp = pd.read_parquet(panel_path, columns=required)
    missing = set(required).difference(fp.columns)
    if missing:
        raise KeyError(f"Panel is missing required columns: {sorted(missing)}")
    fp["time"] = pd.to_datetime(fp["time"])
    fp["ret_daily"] = pd.to_numeric(fp["ret_daily"], errors="coerce")
    fp = fp.dropna(subset=["time", stock_col])
    panel_start = fp["time"].min()

    # ── Warm-up panel (earlier data, broader universe) ──
    warmup_path = Path(warmup_panel_path)
    if warmup_path.exists():
        wu = pd.read_parquet(warmup_path, columns=required)
        wu["time"] = pd.to_datetime(wu["time"])
        wu["ret_daily"] = pd.to_numeric(wu["ret_daily"], errors="coerce")
        wu = wu.dropna(subset=["time", stock_col])
        wu = wu[wu["time"] < panel_start]
        if len(wu) > 0:
            fp = pd.concat([wu, fp], ignore_index=True)
            print(f"  Warm-up: {wu['time'].nunique()} pre-panel dates appended "
                  f"({wu['time'].min().date()} → {wu['time'].max().date()})")

    fp = fp.sort_values([stock_col, "time"])
    if fp.duplicated(["time", stock_col]).any():
        raise ValueError(f"Duplicate (time, {stock_col}) rows found in panel")

    valid_ret = fp["ret_daily"].notna() & (fp["ret_daily"] > -1.0)
    fp["log_ret"] = np.where(valid_ret, np.log1p(fp["ret_daily"]), np.nan)

    # Market series and daily coverage use only actually observed valid returns.
    daily = fp.loc[valid_ret].groupby("time").agg(
        market_ret_1d=("ret_daily", "mean"),
        n_stocks=(stock_col, "nunique"),
        breadth_up_1d_raw=("ret_daily", lambda x: float((x > 0).mean())),
    ).sort_index()
    max_stocks_to_date = daily["n_stocks"].cummax()
    daily["breadth_coverage"] = daily["n_stocks"] / max_stocks_to_date
    daily["market_nav"] = np.exp(np.log1p(daily["market_ret_1d"]).cumsum())
    daily["drawdown"] = daily["market_nav"] / daily["market_nav"].cummax() - 1.0

    # Per-stock cumulative returns for breadth. min_periods=window prevents new
    # IPOs with only a few observations from being called positive/negative.
    for horizon in (20, 60):
        rolling_log = fp.groupby(stock_col, sort=False)["log_ret"].transform(
            lambda s, h=horizon: s.rolling(h, min_periods=h).sum()
        )
        fp[f"positive_{horizon}d"] = np.where(
            rolling_log.notna(), (rolling_log > 0).astype(float), np.nan
        )
        grouped_breadth = fp.groupby("time")[f"positive_{horizon}d"]
        breadth = grouped_breadth.mean()
        breadth_count = grouped_breadth.count()
        breadth = breadth.where(breadth_count >= cfg.min_stock_observations)
        daily[f"breadth_pos_{horizon}d_raw"] = breadth.reindex(daily.index)
        daily[f"breadth_n_{horizon}d"] = breadth_count.reindex(daily.index).fillna(0).astype(int)

    # Risk-adjusted log momentum: cumulative return / matching realized risk.
    market_log = np.log1p(daily["market_ret_1d"])
    trend_parts: list[pd.Series] = []
    for horizon, weight in zip(cfg.trend_windows, cfg.trend_weights):
        cumulative = market_log.rolling(horizon, min_periods=horizon).sum()
        risk = market_log.rolling(horizon, min_periods=horizon).std() * np.sqrt(horizon)
        raw = (cumulative / risk.where(risk > 1e-12)).rename(f"trend_{horizon}_raw")
        daily[raw.name] = raw
        scored = _causal_robust_score(
            raw, cfg.norm_window, cfg.norm_min_periods, cfg.score_temperature
        ).rename(f"trend_{horizon}_score")
        daily[scored.name] = scored
        trend_parts.append(weight * scored)
    daily["trend_score"] = sum(trend_parts)

    breadth_inputs = [
        ("breadth_up_1d_raw", cfg.breadth_weights[0]),
        ("breadth_pos_20d_raw", cfg.breadth_weights[1]),
        ("breadth_pos_60d_raw", cfg.breadth_weights[2]),
    ]
    breadth_parts: list[pd.Series] = []
    for column, weight in breadth_inputs:
        scored_name = column.replace("_raw", "_score")
        scored = _causal_robust_score(
            daily[column], cfg.norm_window, cfg.norm_min_periods,
            cfg.score_temperature, neutral=0.5,
        ).rename(scored_name)
        daily[scored_name] = scored
        breadth_parts.append(weight * scored)
    daily["breadth_score"] = sum(breadth_parts)

    daily["regime_score"] = (
        cfg.trend_share * daily["trend_score"]
        + cfg.breadth_share * daily["breadth_score"]
    ).clip(-1.0, 1.0)
    daily["bull_score"] = (daily["regime_score"] + 1.0) / 2.0
    daily = daily.join(_direction_hysteresis(daily["regime_score"], cfg))

    # Independent volatility axis. Percentiles compare t only with observations
    # available through t-1; the label itself may use volatility observed at t.
    daily["vol20"] = (
        daily["market_ret_1d"].rolling(cfg.vol_window, min_periods=cfg.vol_window).std()
        * np.sqrt(252)
    )
    daily["vol_median"] = daily["vol20"].shift(1).rolling(
        cfg.vol_history, min_periods=cfg.vol_min_periods
    ).median()
    daily["vol_percentile"] = _rolling_percentile(
        daily["vol20"], cfg.vol_history, cfg.vol_min_periods
    )
    daily["vol_score"] = 2.0 * daily["vol_percentile"] - 1.0
    daily["high_vol"] = _vol_hysteresis(daily["vol_percentile"], cfg)

    # Legacy diagnostic name: mean daily return over the middle trend horizon.
    middle_window = cfg.trend_windows[1]
    daily["ma60"] = daily["market_ret_1d"].rolling(
        middle_window, min_periods=middle_window
    ).mean()

    bull_text = daily["bull"].map({1.0: "Bull", 0.0: "Bear"})
    vol_text = daily["high_vol"].map({1.0: "HighVol", 0.0: "LowVol"})
    daily["regime"] = bull_text + "+" + vol_text
    daily.loc[bull_text.isna() | vol_text.isna(), "regime"] = "Unknown"

    # Date first and stable, human-auditable columns before detailed components.
    daily.index.name = "date"
    result = daily.reset_index()
    first = [
        "date", "market_ret_1d", "market_nav", "n_stocks", "breadth_coverage",
        "regime_score", "bull_score", "bull", "state_age", "switch_flag",
        "vol20", "vol_percentile", "vol_score", "high_vol", "regime",
    ]
    return result[first + [c for c in result.columns if c not in first]]


def plot_regime(regime: pd.DataFrame, save_path: str) -> None:
    """Direction + Volatility in one figure, four rows.  Warm-up excluded."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    frame = regime[regime["regime"] != "Unknown"].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    nav = frame["market_nav"] if "market_nav" in frame else (1 + frame["market_ret_1d"]).cumprod()

    bull_color = "#2ca02c"
    bear_color = "#d62728"

    fig, axes = plt.subplots(
        4, 1, figsize=(16, 11),
        gridspec_kw={"height_ratios": [2.5, 1.2, 1.5, 0.8]},
    )

    # ── Row 1: NAV + Bull/Bear shading ──
    ax = axes[0]
    bull_mask = frame["bull"].eq(1.0).to_numpy()
    bear_mask = frame["bull"].eq(0.0).to_numpy()
    if bull_mask.any():
        ax.fill_between(frame["date"], float(nav.min()), float(nav.max()),
                        where=bull_mask, color=bull_color, alpha=0.12, label="Bull")
    if bear_mask.any():
        ax.fill_between(frame["date"], float(nav.min()), float(nav.max()),
                        where=bear_mask, color=bear_color, alpha=0.12, label="Bear")
    ax.plot(frame["date"], nav, "k-", linewidth=0.8)
    ax.set_ylabel("Market NAV (EW)")
    ax.set_title("Market Regime — Direction (rows 1–2)  /  Volatility (rows 3–4)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 2: regime / trend / breadth scores ──
    ax = axes[1]
    ax.plot(frame["date"], frame["regime_score"], color="#111111", lw=1.0,
            label="Regime score")
    ax.plot(frame["date"], frame["trend_score"], color="#1f77b4", lw=0.7,
            alpha=0.75, label="Trend")
    ax.plot(frame["date"], frame["breadth_score"], color="#2ca02c", lw=0.7,
            alpha=0.75, label="Breadth")
    ax.axhline(0.50, color="green", ls="--", lw=0.7, alpha=0.5, label="Bull entry")
    ax.axhline(-0.35, color="red", ls="--", lw=0.7, alpha=0.5, label="Bear entry")
    ax.axhline(0, color="grey", lw=0.4)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Direction score")
    ax.legend(loc="upper left", ncol=3, fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Row 3: vol20 + median + HighVol shading ──
    ax = axes[2]
    ax.plot(frame["date"], frame["vol20"] * 100, color="purple", lw=0.8,
            label="Vol20 (ann.)")
    if "vol_median" in frame:
        ax.plot(frame["date"], frame["vol_median"] * 100, "k--", lw=0.6,
                alpha=0.5, label="Prior 3y median")
    high = frame["high_vol"].eq(1).to_numpy()
    ax.fill_between(frame["date"], 0, frame["vol20"] * 100,
                    where=high, color="orange", alpha=0.18)
    ax.set_ylabel("Volatility (%)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 4: vol_percentile + high_vol state ──
    ax = axes[3]
    ax.plot(frame["date"], frame["vol_percentile"], color="#111111", lw=0.8,
            label="Vol percentile")
    ax.axhline(0.60, color="orange", ls="--", lw=0.6, alpha=0.5, label="HighVol entry")
    ax.axhline(0.40, color="blue", ls="--", lw=0.6, alpha=0.5, label="HighVol exit")
    ax.axhline(0.50, color="grey", lw=0.4)
    ax.fill_between(frame["date"], 0, 1, where=high, color="orange", alpha=0.12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Vol pct")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ― Each axis controls its own view; last row carries the x-label.
    for ax in axes[:3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlabel("")
    axes[3].set_xlabel("Date")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    regime_df = build_regime()
    csv_path = Path("dataset/input/market_regime.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    regime_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    print(regime_df["regime"].value_counts(dropna=False).to_string())

    plot_regime(regime_df, "reports/pics/market_regime.png")
    print("Saved: reports/pics/market_regime.png")
