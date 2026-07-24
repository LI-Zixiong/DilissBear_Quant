"""Declarative dependencies for the live factor update path.

The research implementation remains the numerical source of truth.  This
module describes how much daily history each factor needs and which shared
state family owns its reusable rolling inputs.  Live orchestration uses the
registry to build one projected tail cache instead of letting every factor
read the panel independently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from src.pipeline.factor_panel import DEFAULT_FACTOR_NAMES, FACTOR_BATCH_SPECS


@dataclass(frozen=True)
class LiveFactorDependency:
    factor: str
    daily_history: int
    state_family: str
    columns: tuple[str, ...]
    financial_history: bool = False


# Values include the current observation.  A small boundary cushion is added
# by ``required_tail_rows`` rather than being repeated in every entry.
_HISTORY_OVERRIDES: dict[str, int] = {
    "F001SIZE": 1, "F002SIZENL": 1, "F003LIQUIDITY": 63,
    "F004BETA": 252, "F005RESVOL": 252, "F006MOMENTUM": 253,
    "F007LTREV": 757, "F008STREV": 22, "F009LEVERAGE": 1,
    "F010VALUE": 1, "F011EARNYLD": 1, "F012GROWTH": 1,
    # CAPM residual itself needs 252 rows; downstream residual windows extend it.
    "F013REV5": 256, "F014MOM120_20": 371, "F015VOLREV": 61,
    "F016MAXRET": 20, "F017IVOL": 311, "F018AMIHUD": 60,
    "F019COSTDEV": 120, "F020LIMITUP_RECENCY": 21,
    "F021CFP": 1, "F022GPTA": 1, "F023ACCRUAL": 1, "F024ASSETGR": 1,
    "F025GAP": 2, "F026KLEN": 1, "F027KUP": 1, "F028KLOW": 1,
    "F029KSFT": 1, "F030RSV20": 20, "F031RSV60": 60,
    "F032RANGEZ20": 20, "F033GAPREV5": 6, "F034HIGHDEV20": 20,
    "F035LOWDEV20": 20, "F036VOLSHOCK5": 5, "F037VOLSHOCK20": 20,
    "F038TURNZ20": 20, "F039VSTD20": 20, "F040PVCORR20": 21,
    "F041RETVOLCORR20": 21, "F042AMTCORR20": 20,
    "F043SLOPE20": 20, "F044RSQR20": 20, "F045RESI20": 20,
    "F046LIMITUP20": 20, "F047LIMITDN20": 20, "F048LIMITSTREAKUP": 20,
    "F049ROE": 1, "F050ROA": 1, "F051GPM": 1, "F052CFOA": 1,
    "F053RD_INTENSITY": 1, "F054RECEIVABLE_RATIO": 1, "F055IND": 1,
    "F056GAP_UP_FAIL": 2, "F057INTRA1": 1, "F058O2O_RET5": 6,
    "F059GK_VOL20": 20, "F060ON_INTRA_DIV5": 6,
    "F061GAP_UP_HOLD": 2, "F062GAP_DN_RECOVER": 2,
    "F063RET5D_SKIP1": 7, "F064RET_ACCEL20": 42,
    "F065MAXDD20": 20, "F066EFFICIENCY20": 21,
    "F067TAIL_LOSS20": 20, "F068SKEW20": 20, "F069DNVOL20": 20,
    "F070UP_DN_VOL": 20, "F071VOL_OF_VOL": 24,
    "F072CORR_60D": 60, "F073KURT_60D": 60, "F074BETA_20D": 20,
    "F075VOLUME_RATIO": 1, "F076SIGNED_AMT20": 20,
    "F077AMP_VOL20": 20, "F078TURN_SIZE": 1, "F079TURN_ACCEL": 20,
    "F080VWAP_DEV": 1, "F081STRONG_CLOSE": 1, "F082LOCKED_PCT": 1,
    "F083TURN_FREE": 1, "F084AMT_FREE20": 20, "F085SP_TTM": 1,
    "F086DIV_TTM": 1, "F087LIST_AGE": 1,
    "F088CF_SALES_Q": 1, "F089CASH_PROFIT": 1, "F090CRR": 1,
    "F091CF_VOL": 1, "F092EARN_STAB": 1, "F093FCF_YIELD": 1,
    "F094CAPEX_INT": 1, "F095NET_FIN": 1, "F096DILUTION": 1,
    "F097INT_BURDEN": 1, "F098DIV_PAYOUT": 1,
    # These implementations use a 4*63 daily lag after PIT financial mapping.
    "F099AR_MINUS_REV": 253, "F100INV_MINUS_REV": 253,
}

_FAMILY_BY_PRECOMPUTE = {
    None: "direct",
    "capm": "capm_252",
    "old_ttm": "financial_pit",
    "adj_ohlc": "ohlc_tail",
    "rolling_corr": "moments_20",
    "regression20": "regression_20",
    "o2o": "o2o_tail",
    "path": "path_tail",
    "ret_stats": "return_moments",
    "volume_vwap": "volume_moments",
    "float_value_age": "float_value",
    "v2_financial": "financial_pit",
}

_FINANCIAL_FACTORS = {
    "F009LEVERAGE", "F010VALUE", "F011EARNYLD", "F012GROWTH",
    "F021CFP", "F022GPTA", "F023ACCRUAL", "F024ASSETGR",
    "F049ROE", "F050ROA", "F051GPM", "F052CFOA", "F053RD_INTENSITY",
    "F054RECEIVABLE_RATIO", *[f"F{i:03d}" for i in range(88, 101)],
}


def build_live_factor_registry() -> dict[str, LiveFactorDependency]:
    registry: dict[str, LiveFactorDependency] = {}
    for spec in FACTOR_BATCH_SPECS:
        family = _FAMILY_BY_PRECOMPUTE[spec.precompute]
        cols = tuple(dict.fromkeys(("time", "stock_id", *spec.keep_cols_extra)))
        for factor in spec.factors:
            registry[factor] = LiveFactorDependency(
                factor=factor,
                daily_history=_HISTORY_OVERRIDES[factor],
                state_family=family,
                columns=cols,
                financial_history=factor in _FINANCIAL_FACTORS,
            )
    # F020 is intentionally outside the production batch list and is emitted
    # by the scheduler's compatibility fallback.
    registry["F020LIMITUP_RECENCY"] = LiveFactorDependency(
        factor="F020LIMITUP_RECENCY",
        daily_history=_HISTORY_OVERRIDES["F020LIMITUP_RECENCY"],
        state_family="limit_tail",
        columns=("time", "stock_id", "limit_status"),
    )
    missing = set(DEFAULT_FACTOR_NAMES) - set(registry)
    if missing:
        raise RuntimeError(f"Live factor registry is incomplete: {sorted(missing)}")
    return registry


LIVE_FACTOR_REGISTRY = build_live_factor_registry()


def required_tail_rows(
    factor_names: Sequence[str],
    new_dates: int = 1,
    boundary_cushion: int = 2,
) -> int:
    """Return the number of distinct trading dates needed by a live update."""
    if not factor_names:
        return max(1, new_dates)
    unknown = set(factor_names) - set(LIVE_FACTOR_REGISTRY)
    if unknown:
        raise ValueError(f"Unknown live factors: {sorted(unknown)}")
    history = max(LIVE_FACTOR_REGISTRY[f].daily_history for f in factor_names)
    return history + max(0, new_dates - 1) + boundary_cushion


def required_columns(factor_names: Iterable[str]) -> tuple[str, ...]:
    cols: set[str] = {"time", "stock_id"}
    for factor in factor_names:
        cols.update(LIVE_FACTOR_REGISTRY[factor].columns)
    return tuple(sorted(cols))


def state_schema() -> dict[str, dict[str, object]]:
    """Machine-readable schema grouped by shared state family."""
    grouped: dict[str, dict[str, object]] = {}
    for dep in LIVE_FACTOR_REGISTRY.values():
        item = grouped.setdefault(
            dep.state_family,
            {"factors": [], "columns": set(), "max_history": 0},
        )
        item["factors"].append(dep.factor)
        item["columns"].update(dep.columns)
        item["max_history"] = max(int(item["max_history"]), dep.daily_history)
    return {
        name: {
            "factors": sorted(value["factors"]),
            "columns": sorted(value["columns"]),
            "max_history": value["max_history"],
        }
        for name, value in sorted(grouped.items())
    }
