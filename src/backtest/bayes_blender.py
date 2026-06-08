"""
Industry-Conditional Rolling Bayes Model Blender.

Replaces Rank-Ridge / equal-weight fusion: instead of fitting a single weight
per model, this calibrates per-industry per-rank-bin likelihood ratios that
answer "when model M puts a stock in rank bin B within industry G, how likely
is that stock to be a future head-winner?"

Design:
    Global models (no retrain) -> daily rank_pct per model
    -> rolling-window LLR calibration at global / sector / industry levels
    -> Naive Bayes score with shrinkage, clip, and temperature

Key choices:
    1. H = future 1d return in top 5% (handled by the runner)
    2. Rolling calibration avoids fixed-valid-regime overfitting
    3. No industry prior by default (no direct sector timing)
    4. Per-size-class Laplace alpha, LLR clip=1.0
       (tau removed — applied post-sum, equal for all stocks → no-op for ranking)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Sector & industry classification
# ------------------------------------------------------------------

SECTOR_MAP: dict[str, str] = {
    "63": "TMT", "61": "TMT", "62": "TMT",
    "64": "MFG", "71": "MFG", "24": "MFG", "65": "MFG",
    "37": "CYCLICAL", "22": "CYCLICAL", "28": "CYCLICAL", "34": "CYCLICAL",
    "41": "PROPERTY", "73": "PROPERTY",
    "11": "STABLE", "49": "STABLE", "72": "STABLE",
    "27": "CONS_HEALTH", "42": "CONS_HEALTH", "23": "CONS_HEALTH",
    "77": "GLOBAL", "-1": "GLOBAL",
}

STRONG = frozenset({"27", "37", "63", "64", "71", "22", "24", "28"})
WEAK = frozenset({"72", "65", "41", "42", "49", "73", "34"})
SMALL = frozenset({"23", "62", "11", "61"})
GLOBAL_ONLY = frozenset({"77", "-1"})

W_STRONG = (0.30, 0.20, 0.50)       # global, sector, industry
W_WEAK = (0.45, 0.30, 0.25)
W_SMALL = (0.55, 0.45, 0.00)
W_GLOBAL_ONLY = (1.00, 0.00, 0.00)


ALPHA_STRONG = 10.0   # Laplace smoothing for strong industries
ALPHA_OTHER  = 20.0   # Laplace smoothing for weak / small / global

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

@dataclass
class BayesBlenderConfig:
    models: Tuple[str, ...] = ("lightgbm", "xgboost", "dlinear", "gated_dwtcn")

    # Current default keeps valid-split tuning feasible. For final long walk-forward
    # runs, override to 168 or 252 from the runner if desired.
    rolling_window: int = 63

    rank_bins: Tuple[float, ...] = (
        0.00, 0.50, 0.70, 0.85, 0.90, 0.95, 0.97, 0.99, 1.00,
    )

    laplace_alpha_strong: float = 10.0
    laplace_alpha_other: float = 20.0

    # V1: raised from 1.0 to 2.0 — 63-78% of industry bin7 groups were
    # saturated at +1.0 (diagnostic report, 2026-06-06).
    llr_clip: float = 2.0

    min_group_total: int = 300
    min_group_head: int = 20

    # V1: per-model (global, sector, industry) weights.
    # XGBoost/DLinear have strong global + industry head prediction.
    # LGBM/GatedDW have near-zero global LLR but strong industry LLR.
    model_weights: dict = field(default_factory=lambda: {
        "xgboost": {
            "strong":       (0.25, 0.20, 0.55),
            "weak":         (0.40, 0.30, 0.30),
            "small":        (0.55, 0.45, 0.00),
            "global_only":  (1.00, 0.00, 0.00),
        },
        "dlinear": {
            "strong":       (0.25, 0.20, 0.55),
            "weak":         (0.40, 0.30, 0.30),
            "small":        (0.55, 0.45, 0.00),
            "global_only":  (1.00, 0.00, 0.00),
        },
        "lightgbm": {
            "strong":       (0.05, 0.25, 0.70),
            "weak":         (0.15, 0.40, 0.45),
            "small":        (0.20, 0.80, 0.00),
            "global_only":  (1.00, 0.00, 0.00),
        },
        "gated_dwtcn": {
            "strong":       (0.05, 0.25, 0.70),
            "weak":         (0.15, 0.40, 0.45),
            "small":        (0.20, 0.80, 0.00),
            "global_only":  (1.00, 0.00, 0.00),
        },
    })


# ------------------------------------------------------------------
# Calibrator
# ------------------------------------------------------------------

class IndustryBayesCalibrator:
    """Rolling-window LLR calibrator — no base-model retraining."""

    GLOBAL_KEY = "__GLOBAL__"

    def __init__(self, cfg: BayesBlenderConfig | None = None):
        self.cfg = cfg or BayesBlenderConfig()
        self.models = list(self.cfg.models)
        self.rank_bins = np.asarray(self.cfg.rank_bins, dtype=float)
        if len(self.rank_bins) < 2 or not np.all(np.diff(self.rank_bins) > 0):
            raise ValueError("rank_bins must be strictly increasing.")
        if self.rank_bins[0] > 0.0 or self.rank_bins[-1] < 1.0:
            raise ValueError("rank_bins should cover [0, 1].")
        self.n_bins = len(self.rank_bins) - 1
        self._reset()

    def _reset(self) -> None:
        self._t: dict[str, dict[str, tuple[np.ndarray | None, list[str]]]] = {
            "global": {}, "sector": {}, "industry": {},
        }
        self._meta: dict[str, dict[str, dict[str, dict[str, int]]]] = {
            "global": {}, "sector": {}, "industry": {},
        }
        # Incremental: running totals per (level, model, group)
        # _rt[level][model][group] = (head_arr, not_arr)  each shape (n_bins,)
        self._rt: dict[str, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]] = {
            "global": {}, "sector": {}, "industry": {},
        }
        # Day-by-day snapshots for eviction
        from collections import deque
        self._daily_counts: deque[list] = deque()
        self._current_window_size: int = 0

    # ── public API ────────────────────────────────────────────

    def fit_window(self, window_ranks: dict[str, pd.DataFrame]) -> None:
        """
        Parameters
        ----------
        window_ranks:
            dict[model -> DataFrame(time, stock_id, industry_sw, rank_pct, H)]
        """
        self._reset()
        for m in self.models:
            if m not in window_ranks:
                raise KeyError(f"Missing window_ranks for model: {m}")

            df = window_ranks[m].copy()
            required = {"industry_sw", "rank_pct", "H"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"window_ranks[{m}] missing columns: {missing}")

            df["industry_sw"] = _clean_industry(df["industry_sw"])
            df["sector"] = df["industry_sw"].map(_sector_for)

            for level, gcol in (
                ("global", None),
                ("sector", "sector"),
                ("industry", "industry_sw"),
            ):
                table, groups, meta = self._estimate(df, level=level, group_col=gcol)
                self._t[level][m] = (table, groups)
                self._meta[level][m] = meta

    # ── incremental update ────────────────────────────────────

    def update_incremental(
        self, new_day_ranks: dict[str, pd.DataFrame],
        drop_day_ranks: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """+new_day counts, -drop_day counts, recompute all LLR tables."""
        if not self._rt["global"]:
            self._init_from_window(new_day_ranks)
            return
        for m in self.models:
            new_df = self._day_df(new_day_ranks, m)
            drop_df = self._day_df(drop_day_ranks, m) if drop_day_ranks else None
            for level, gcol in [("global", None), ("sector", "sector"),
                                 ("industry", "industry_sw")]:
                self._update_level(level, m, new_df, drop_df, gcol)
        self._recompute_all_tables()

    def _init_from_window(self, ranks: dict[str, pd.DataFrame]) -> None:
        for m in self.models:
            df = self._day_df(ranks, m)
            if df is None: continue
            for level, gcol in [("global", None), ("sector", "sector"),
                                 ("industry", "industry_sw")]:
                rt_m = self._rt[level].setdefault(m, {})
                groups = [self.GLOBAL_KEY] if gcol is None else sorted(df[gcol].astype(str).unique())
                for grp in groups:
                    g = df if gcol is None else df[df[gcol].astype(str) == grp]
                    h, b = g["H"].to_numpy(dtype=int), g["_bin"].to_numpy(dtype=int)
                    head = np.bincount(b[h == 1], minlength=self.n_bins).astype(float)
                    not_h = np.bincount(b[h == 0], minlength=self.n_bins).astype(float)
                    rt_m[str(grp)] = (head, not_h)
        self._recompute_all_tables()

    def _day_df(self, ranks_dict, model):
        if ranks_dict is None or model not in ranks_dict: return None
        df = ranks_dict[model].copy()
        if {"rank_pct", "H"} - set(df.columns): return None
        df["_bin"] = self._rank_to_bin(df["rank_pct"].to_numpy(dtype=float))
        df["H"] = df["H"].astype(int).clip(0, 1)
        df["industry_sw"] = _clean_industry(df.get("industry_sw", pd.Series(["-1"]*len(df))))
        df["sector"] = df["industry_sw"].map(_sector_for)
        return df

    def _update_level(self, level, model, new_df, drop_df, group_col):
        rt_m = self._rt[level].get(model, {})
        for df, sign in [(new_df, +1), (drop_df, -1)]:
            if df is None: continue
            h, b = df["H"].to_numpy(dtype=int), df["_bin"].to_numpy(dtype=int)
            groups = np.full(len(df), self.GLOBAL_KEY) if group_col is None else df[group_col].astype(str).to_numpy()
            for grp in np.unique(groups):
                msk = (groups == grp)
                head = np.bincount(b[msk & (h == 1)], minlength=self.n_bins).astype(float) * sign
                not_h = np.bincount(b[msk & (h == 0)], minlength=self.n_bins).astype(float) * sign
                k = str(grp)
                if k not in rt_m: rt_m[k] = (np.zeros(self.n_bins), np.zeros(self.n_bins))
                rt_m[k] = (np.maximum(0, rt_m[k][0] + head),
                            np.maximum(0, rt_m[k][1] + not_h))
        self._rt[level][model] = rt_m

    def _recompute_all_tables(self) -> None:
        for level in ["global", "sector", "industry"]:
            for m in self.models:
                rm = self._rt[level].get(m, {})
                if not rm: continue
                grps = sorted(rm.keys())
                n_g = len(grps)
                tbl = np.zeros((self.n_bins, n_g), dtype=float)
                mt: dict[str, dict] = {}
                for gi, grp in enumerate(grps):
                    hd, nh = rm[grp]
                    hn, nn = int(hd.sum()), int(nh.sum())
                    total = hn + nn
                    alpha = ALPHA_STRONG if (level == "industry" and grp in STRONG and total > 5000) else ALPHA_OTHER
                    ph = np.maximum(1e-12, (hd + alpha) / (hn + alpha * self.n_bins))
                    pn = np.maximum(1e-12, (nh + alpha) / (nn + alpha * self.n_bins))
                    llr = np.log(ph / pn)
                    llr = np.clip(llr, -self.cfg.llr_clip, self.cfg.llr_clip)
                    tbl[:, gi] = llr
                    mt[grp] = {"total": total, "head": hn, "not_head": nn}
                self._t[level][m] = (tbl, grps)
                self._meta[level][m] = mt

    def score(self, today: pd.DataFrame, today_ranks: dict[str, pd.DataFrame]) -> pd.Series:
        """
        Parameters
        ----------
        today:
            DataFrame(time, stock_id, industry_sw)
        today_ranks:
            dict[model -> DataFrame(time, stock_id, rank_pct)]

        Returns
        -------
        pd.Series indexed by stock_id, named bayes_score.
        """
        required = {"time", "stock_id", "industry_sw"}
        missing = required - set(today.columns)
        if missing:
            raise ValueError(f"today missing columns: {missing}")

        base = today[["time", "stock_id", "industry_sw"]].drop_duplicates().copy()
        base["stock_id"] = base["stock_id"].astype(str).str.strip().str.zfill(6)
        base["industry_sw"] = _clean_industry(base["industry_sw"])
        base["sector"] = base["industry_sw"].map(_sector_for)
        base = base.reset_index(drop=True)

        total = np.zeros(len(base), dtype=float)
        industries = base["industry_sw"].to_numpy(dtype=str)
        sectors = base["sector"].to_numpy(dtype=str)

        for m in self.models:
            if m not in today_ranks:
                raise KeyError(f"Missing today_ranks for model: {m}")
            r = today_ranks[m][["time", "stock_id", "rank_pct"]].copy()
            r["stock_id"] = r["stock_id"].astype(str).str.strip().str.zfill(6)
            merged = base[["time", "stock_id"]].merge(r, on=["time", "stock_id"], how="left")
            ranks = merged["rank_pct"].fillna(0.5).to_numpy(dtype=float)
            bins = self._rank_to_bin(ranks)
            total += self._model_contrib(m, industries, sectors, bins)

        return pd.Series(total, index=base["stock_id"], name="bayes_score")

    def score_detailed(
        self, today: pd.DataFrame, today_ranks: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Like score() but returns per-model LLR breakdown for diagnostics.

        Output columns: time, stock_id, industry_sw, sector, bayes_score, H,
                        {m}_rank, {m}_llr_global, {m}_llr_sector, {m}_llr_industry
        """
        base = today[["time", "stock_id", "industry_sw"]].drop_duplicates().copy()
        base["stock_id"] = base["stock_id"].astype(str).str.strip().str.zfill(6)
        base["industry_sw"] = _clean_industry(base["industry_sw"])
        base["sector"] = base["industry_sw"].map(_sector_for)
        base = base.reset_index(drop=True)
        n = len(base)
        industries = base["industry_sw"].to_numpy(dtype=str)
        sectors = base["sector"].to_numpy(dtype=str)

        cols = {}
        total = np.zeros(n, dtype=float)
        for m in self.models:
            r = today_ranks[m][["time", "stock_id", "rank_pct"]].copy()
            r["stock_id"] = r["stock_id"].astype(str).str.strip().str.zfill(6)
            merged = base[["time", "stock_id"]].merge(r, on=["time", "stock_id"], how="left")
            ranks = merged["rank_pct"].fillna(0.5).to_numpy(dtype=float)
            bins = self._rank_to_bin(ranks)
            cols[f"{m}_rank"] = ranks

            # per-level LLR
            gt, gg = self._t["global"].get(m, (None, []))
            st, sg = self._t["sector"].get(m, (None, []))
            it, ig = self._t["industry"].get(m, (None, []))
            gl = self._lookup_many("global", m, gt, gg, np.full(n, self.GLOBAL_KEY), bins, np.zeros(n), False)
            sl = self._lookup_many("sector", m, st, sg, sectors, bins, gl, True)
            il = self._lookup_many("industry", m, it, ig, industries, bins, sl, True)
            cols[f"{m}_llr_global"]   = gl
            cols[f"{m}_llr_sector"]   = sl
            cols[f"{m}_llr_industry"] = il

            wg, ws, wi = self._weights_many(m, industries, self.cfg.model_weights)
            contrib = wg * gl + ws * sl + wi * il
            cols[f"{m}_llr_blended"] = contrib
            total += contrib

        out = pd.DataFrame(cols, index=base.index)
        out["time"] = base["time"].values
        out["stock_id"] = base["stock_id"].values
        out["industry_sw"] = industries
        out["sector"] = sectors
        out["bayes_score"] = total
        return out[["time", "stock_id", "industry_sw", "sector", "bayes_score"] +
                    [f"{m}_rank" for m in self.models] +
                    [f"{m}_llr_global" for m in self.models] +
                    [f"{m}_llr_sector" for m in self.models] +
                    [f"{m}_llr_industry" for m in self.models] +
                    [f"{m}_llr_blended" for m in self.models]]

    def diagnostic_report(self) -> pd.DataFrame:
        """Per-group per-model per-bin P(H|bin) from current calibration."""
        rows = []
        for level_name in ["global", "sector", "industry"]:
            for m in self.models:
                table, groups = self._t[level_name].get(m, (None, None))
                if table is None:
                    continue
                for gi, grp in enumerate(groups):
                    for bi in range(self.n_bins):
                        p_h = np.clip(np.exp(table[bi, gi]), 0, None)  # exp(LLR) = odds
                        p_h = p_h / (1.0 + p_h)  # odds → probability
                        rows.append({
                            "level": level_name, "model": m, "group": grp,
                            "bin": bi, "bin_lo": self.rank_bins[bi],
                            "bin_hi": self.rank_bins[bi + 1],
                            "llr": table[bi, gi],
                            "P_H_given_bin": round(float(p_h), 6),
                        })
        return pd.DataFrame(rows)

    # ── per-model contribution ─────────────────────────────────

    def _model_contrib(
        self,
        model: str,
        inds: np.ndarray,
        secs: np.ndarray,
        bins: np.ndarray,
    ) -> np.ndarray:
        """Vectorized global/sector/industry LLR blending for one model."""
        n = len(inds)

        gt, gg = self._t["global"].get(model, (None, []))
        st, sg = self._t["sector"].get(model, (None, []))
        it, ig = self._t["industry"].get(model, (None, []))

        # global: fallback to zero if absent
        gl = self._lookup_many(
            level="global", model=model, table=gt, groups=gg,
            keys=np.full(n, self.GLOBAL_KEY, dtype=object), bins=bins,
            fallback=np.zeros(n, dtype=float), require_quality=False,
        )

        # sector: fallback to global
        sl = self._lookup_many(
            level="sector", model=model, table=st, groups=sg,
            keys=secs.astype(str), bins=bins,
            fallback=gl, require_quality=True,
        )

        # industry: fallback to sector
        il = self._lookup_many(
            level="industry", model=model, table=it, groups=ig,
            keys=inds.astype(str), bins=bins,
            fallback=sl, require_quality=True,
        )

        wg, ws, wi = self._weights_many(model, inds.astype(str), self.cfg.model_weights)
        return wg * gl + ws * sl + wi * il

    def _lookup_many(
        self,
        level: str,
        model: str,
        table: np.ndarray | None,
        groups: list[str],
        keys: np.ndarray,
        bins: np.ndarray,
        fallback: np.ndarray,
        require_quality: bool,
    ) -> np.ndarray:
        """Lookup LLR by group and bin with vectorized per-group assignment."""
        out = fallback.copy()
        if table is None or not groups:
            return out

        group_to_idx = {g: i for i, g in enumerate(groups)}
        keys = keys.astype(str)
        bins = bins.astype(int)

        for key in np.unique(keys):
            gi = group_to_idx.get(str(key))
            if gi is None:
                continue
            if require_quality and not self._group_has_enough_samples(level, model, str(key)):
                continue
            mask = keys == key
            out[mask] = table[bins[mask], gi]
        return out

    @staticmethod
    @staticmethod
    def _weights_many(model: str, inds: np.ndarray,
                       weight_map: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wmap = weight_map or {}
        model_w = wmap.get(model, {})
        def _w(ind):
            i = str(ind)
            if i in STRONG:      return model_w.get("strong", W_STRONG)
            if i in WEAK:        return model_w.get("weak", W_WEAK)
            if i in SMALL:       return model_w.get("small", W_SMALL)
            if i in GLOBAL_ONLY: return model_w.get("global_only", W_GLOBAL_ONLY)
            return model_w.get("small", W_SMALL)
        w = np.asarray([_w(ind) for ind in inds.astype(str)], dtype=float)
        return w[:, 0], w[:, 1], w[:, 2]

    def _group_has_enough_samples(self, level: str, model: str, group: str) -> bool:
        if level == "global":
            return True
        try:
            m = self._meta[level][model][str(group)]
        except KeyError:
            return False
        return (
            m["total"] >= self.cfg.min_group_total
            and m["head"] >= self.cfg.min_group_head
        )

    # ── estimation ─────────────────────────────────────────────

    def _estimate(
        self,
        df: pd.DataFrame,
        level: str,
        group_col: str | None,
    ) -> tuple[np.ndarray | None, list[str], dict[str, dict[str, int]]]:
        cols = ["rank_pct", "H"] + ([group_col] if group_col else [])
        d = df[cols].copy().dropna(subset=["rank_pct", "H"])
        if d.empty:
            return None, [], {}

        d["H"] = d["H"].astype(int).clip(0, 1)
        d["_bin"] = self._rank_to_bin(d["rank_pct"].to_numpy(dtype=float))
        d["_group"] = self.GLOBAL_KEY if group_col is None else d[group_col].astype(str)

        groups = sorted(d["_group"].unique())
        n_g = len(groups)
        nb = self.n_bins
        table = np.zeros((nb, n_g), dtype=float)
        meta: dict[str, dict[str, int]] = {}

        for gi, grp in enumerate(groups):
            g = d.loc[d["_group"] == grp]
            h = g["H"].to_numpy(dtype=int)
            b = g["_bin"].to_numpy(dtype=int)

            head_n = int((h == 1).sum())
            not_n = int((h == 0).sum())
            total = head_n + not_n
            if total == 0:
                continue

            head_b = np.bincount(b[h == 1], minlength=nb).astype(float)
            not_b = np.bincount(b[h == 0], minlength=nb).astype(float)

            alpha = self._alpha_for(level, str(grp))
            p_h = (head_b + alpha) / (head_n + alpha * nb)
            p_n = (not_b + alpha) / (not_n + alpha * nb)

            llr = np.log(p_h / p_n)
            table[:, gi] = np.clip(llr, -self.cfg.llr_clip, self.cfg.llr_clip)
            meta[str(grp)] = {"total": total, "head": head_n, "not_head": not_n}

        return table, [str(g) for g in groups], meta

    def _alpha_for(self, level: str, group: str) -> float:
        if level == "industry" and str(group) in STRONG:
            return float(self.cfg.laplace_alpha_strong)
        return float(self.cfg.laplace_alpha_other)

    # ── helpers ────────────────────────────────────────────────

    def _rank_to_bin(self, v: np.ndarray) -> np.ndarray:
        v = np.clip(np.asarray(v, dtype=float), self.rank_bins[0], self.rank_bins[-1])
        idx = np.digitize(v, self.rank_bins[1:-1], right=True)
        return np.clip(idx, 0, self.n_bins - 1).astype(int)


# ------------------------------------------------------------------
# Shared helpers (outside class — also callable from outer script)
# ------------------------------------------------------------------

def _clean_industry(s: pd.Series) -> pd.Series:
    def _fmt(x):
        try:
            return str(int(float(x)))
        except (ValueError, TypeError):
            return str(x).strip()

    return s.fillna(-1).map(_fmt)


def _sector_for(ind: str) -> str:
    return SECTOR_MAP.get(str(ind), "UNKNOWN")


def _weights_for(ind: str) -> Tuple[float, float, float]:
    i = str(ind)
    if i in STRONG:
        return W_STRONG
    if i in WEAK:
        return W_WEAK
    if i in SMALL:
        return W_SMALL
    if i in GLOBAL_ONLY:
        return W_GLOBAL_ONLY
    return W_SMALL


# ------------------------------------------------------------------
# Top-level builder used by the runner
# ------------------------------------------------------------------

def build_bayes_scores(
    merged_ranked: pd.DataFrame,
    models: list[str],
    config: BayesBlenderConfig | None = None,
    eval_dates: set[pd.Timestamp] | set[np.datetime64] | None = None,
    eval_burnin: int | None = None,
) -> pd.DataFrame:
    """
    Run rolling Bayes calibration across all dates in merged_ranked.

    merged_ranked must have: time, stock_id, industry_sw, {m}_r columns, H.
    If eval_dates is provided, rolling calibration can use all historical dates,
    but scores are emitted only for eval_dates. This enables valid warm-up for
    final test evaluation without reporting valid results.

    eval_burnin optionally enforces a common evaluation start across different
    rolling windows. Example: compare rolling_window=126/168/252 fairly by
    setting eval_burnin=252; all versions then emit scores only after date #252.
    """
    cfg = config or BayesBlenderConfig(models=tuple(models))
    cal = IndustryBayesCalibrator(cfg)
    all_dates = sorted(pd.to_datetime(merged_ranked["time"].unique()))
    eval_date_set = None if eval_dates is None else {pd.Timestamp(d) for d in eval_dates}
    effective_burnin = cfg.rolling_window if eval_burnin is None else max(cfg.rolling_window, int(eval_burnin))
    scores_list: list[pd.DataFrame] = []

    for i, t in enumerate(all_dates):
        if i < cfg.rolling_window:
            continue

        w_start = all_dates[i - cfg.rolling_window]
        w_end = all_dates[i - 1]
        window = merged_ranked[
            (merged_ranked["time"] >= w_start) &
            (merged_ranked["time"] <= w_end)
        ]

        w_ranks = {}
        for m in models:
            rcol = f"{m}_r"
            w_ranks[m] = window[["time", "stock_id", "industry_sw", rcol, "H"]].rename(
                columns={rcol: "rank_pct"}
            )

        cal.fit_window(w_ranks)

        if i < effective_burnin:
            continue

        if eval_date_set is not None and pd.Timestamp(t) not in eval_date_set:
            continue

        today = merged_ranked.loc[
            merged_ranked["time"] == t, ["time", "stock_id", "industry_sw"]
        ]
        t_ranks = {}
        for m in models:
            rcol = f"{m}_r"
            t_ranks[m] = merged_ranked.loc[
                merged_ranked["time"] == t, ["time", "stock_id", rcol]
            ].rename(columns={rcol: "rank_pct"})

        scores = cal.score(today, t_ranks).reset_index()
        scores.columns = ["stock_id", "bayes_score"]
        scores["time"] = t
        scores_list.append(scores)

    if not scores_list:
        return pd.DataFrame(columns=["stock_id", "bayes_score", "time"])
    return pd.concat(scores_list, ignore_index=True)


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("BayesBlender smoke test ...")
    cfg = BayesBlenderConfig(models=("lightgbm",), rolling_window=20)
    cal = IndustryBayesCalibrator(cfg)

    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=100, freq="B")
    fake = pd.DataFrame({
        "time": np.random.choice(dates, 5000),
        "stock_id": [f"{i:06d}" for i in np.random.randint(1, 500, 5000)],
        "industry_sw": np.random.choice(["27", "37", "63", "11", "77"], 5000),
        "rank_pct": np.clip(np.random.beta(2, 5, 5000), 0, 1),
        "H": np.random.binomial(1, 0.05, 5000),
    })
    cal.fit_window({"lightgbm": fake})
    today = fake.head(10)[["time", "stock_id", "industry_sw"]]
    today_r = {"lightgbm": fake.head(10)[["time", "stock_id", "rank_pct"]]}
    scores = cal.score(today, today_r)
    print(f"  OK — {len(scores)} scores, [{scores.min():.4f}, {scores.max():.4f}]")
    print(
        f"  rolling={cfg.rolling_window}, "
        f"alpha_strong={cfg.laplace_alpha_strong}, "
        f"alpha_other={cfg.laplace_alpha_other}, clip={cfg.llr_clip}"
    )
