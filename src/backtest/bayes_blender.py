"""
Industry-Conditional Rolling Bayes Model Blender.

Replaces Rank-Ridge / equal-weight fusion: instead of fitting a single weight
per model, this calibrates per-industry per-rank-bin likelihood ratios that
answer "when model M puts a stock in rank bin B within industry G, how likely
is that stock to be a future head-winner?"

Design:
    Global models (no retrain) -> daily rank_pct per model
    -> rolling-window LLR calibration at global / sector / industry levels
    -> Naive Bayes score with auto alpha, auto clip, and auto hierarchy weights.

Key choices:
    1. H = future 1d return in top 5% (handled by the runner)
    2. Rolling calibration avoids fixed-valid-regime overfitting
    3. No industry prior by default (no direct sector timing)
    4. tau removed — applied post-sum, equal for all stocks -> no-op for ranking

V2.1 implementation notes:
    - SW2021 sector map fixed.
    - GLOBAL_ONLY fixed to {"51", "-1"}; 77 is beauty care, not global.
    - Auto weights keep across-model per-level normalisation.
    - Added level-budget gate: first decide which hierarchy level is useful,
      then decide which model is useful within that level.
    - global_only names only use global LLR and respect auto-learned global
      model weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Sector & industry classification — SW2021 first-level industry code
# ------------------------------------------------------------------

SECTOR_MAP: dict[str, str] = {
    # TMT: 电子＋计算机＋传媒＋通信
    "27": "TMT", "71": "TMT", "72": "TMT", "73": "TMT",
    # MFG: 汽车＋电力设备＋机械设备＋国防军工
    "28": "MFG", "63": "MFG", "64": "MFG", "65": "MFG",
    # CYCLICAL: 化工＋钢铁＋有色＋煤炭＋石油石化
    "22": "CYCLICAL", "23": "CYCLICAL", "24": "CYCLICAL",
    "74": "CYCLICAL", "75": "CYCLICAL",
    # PROPERTY: 房地产＋建材＋建筑＋银行＋非银
    "43": "PROPERTY", "61": "PROPERTY", "62": "PROPERTY",
    "48": "PROPERTY", "49": "PROPERTY",
    # CONSUMER: 家电＋食品＋纺织＋轻工＋商贸＋社服＋美容＋医药＋农林牧渔
    "33": "CONSUMER", "34": "CONSUMER", "35": "CONSUMER",
    "36": "CONSUMER", "45": "CONSUMER", "46": "CONSUMER",
    "77": "CONSUMER", "37": "CONSUMER", "11": "CONSUMER",
    # STABLE: 公用事业＋交通运输＋环保
    "41": "STABLE", "42": "STABLE", "76": "STABLE",
    # GLOBAL: 综合＋未知
    "51": "GLOBAL", "-1": "GLOBAL",
}

# Manual fallback tiers. In auto mode these are mostly for readable fallback,
# but keep the SW2021 code base consistent.
STRONG = frozenset({
    "22",  # 基础化工
    "24",  # 有色金属
    "27",  # 电子
    "28",  # 汽车
    "37",  # 医药生物
    "63",  # 电力设备
    "64",  # 机械设备
    "71",  # 计算机
})
WEAK = frozenset({
    "23",  # 钢铁
    "34",  # 食品饮料
    "41",  # 公用事业
    "42",  # 交通运输
    "43",  # 房地产
    "49",  # 非银金融
    "61",  # 建筑材料
    "62",  # 建筑装饰
    "65",  # 国防军工
    "72",  # 传媒
    "73",  # 通信
})
SMALL = frozenset({
    "11",  # 农林牧渔
    "33",  # 家用电器
    "35",  # 纺织服饰
    "36",  # 轻工制造
    "45",  # 商贸零售
    "46",  # 社会服务
    "48",  # 银行
    "74",  # 煤炭
    "75",  # 石油石化
    "76",  # 环保
    "77",  # 美容护理
})
GLOBAL_ONLY = frozenset({"51", "-1"})

W_STRONG = (0.30, 0.20, 0.50)       # global, sector, industry
W_WEAK = (0.45, 0.30, 0.25)
W_SMALL = (0.55, 0.45, 0.00)
W_GLOBAL_ONLY = (1.00, 0.00, 0.00)

ALPHA_STRONG = 10.0
ALPHA_OTHER = 20.0


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

@dataclass
class BayesBlenderConfig:
    models: Tuple[str, ...] = ("lightgbm", "xgboost", "dlinear", "gated_dwtcn")

    rolling_window: int = 63

    rank_bins: Tuple[float, ...] = (
        0.00, 0.94, 0.96, 0.985, 1.00,
    )
    # 4 bins:
    #   bin0 = background / negative-evidence region
    #   bin1 = buffer
    #   bin2 = strong
    #   bin3 = elite
    # The lowest bin is not ignored; it may provide negative evidence.

    laplace_alpha_strong: float = 10.0    # fallback when auto_alpha=False
    laplace_alpha_other: float = 20.0     # fallback when auto_alpha=False
    auto_alpha: bool = True               # alpha = clamp(3, 1000/sqrt(total), 30)

    # Manual clip fallback when auto_clip=False.
    llr_clip: float = 2.0

    # Auto clip and auto hierarchy weights.
    # clip = max(0.5, percentile(|raw LLR|, clip_percentile)) each window.
    auto_clip: bool = True
    auto_weights: bool = True
    clip_percentile: float = 85.0  # optimal in smooth=3 scan
    use_sector: bool = True

    # If True, auto weights are:
    #     final_weight[m, level] = B[level] * q(model=m | level)
    # rather than only q(model | level). This prevents weak hierarchy levels from
    # being forced to speak.
    level_budget_gate: bool = False
    level_budget: tuple[float, float, float] = (0.333, 0.333, 0.334)
    # Manual level budget (global, sector, industry). Used when gate=False.
    # Sum must be ~1.0. Only the relative proportions matter.

    min_group_total: int = 300
    min_group_head: int = 20

    # Manual per-model hierarchy weights used when auto_weights=False.
    model_weights: dict = field(default_factory=lambda: {
        "xgboost": {
            "strong":      (0.25, 0.20, 0.55),
            "weak":        (0.40, 0.30, 0.30),
            "small":       (0.55, 0.45, 0.00),
            "global_only": (1.00, 0.00, 0.00),
        },
        "dlinear": {
            "strong":      (0.25, 0.20, 0.55),
            "weak":        (0.40, 0.30, 0.30),
            "small":       (0.55, 0.45, 0.00),
            "global_only": (1.00, 0.00, 0.00),
        },
        "lightgbm": {
            "strong":      (0.05, 0.25, 0.70),
            "weak":        (0.15, 0.40, 0.45),
            "small":       (0.20, 0.80, 0.00),
            "global_only": (1.00, 0.00, 0.00),
        },
        "gated_dwtcn": {
            "strong":      (0.05, 0.25, 0.70),
            "weak":        (0.15, 0.40, 0.45),
            "small":       (0.20, 0.80, 0.00),
            "global_only": (1.00, 0.00, 0.00),
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
        self._auto_clip: float = self.cfg.llr_clip
        self._auto_weights: dict = {}
        self._level_budgets: dict[str, float] = {}
        self.params_history: list[dict] = []
        self._levels = ["global", "industry"] if not self.cfg.use_sector else ["global", "sector", "industry"]
        self._reset()

    def _reset(self) -> None:
        self._t: dict[str, dict[str, tuple[np.ndarray | None, list[str]]]] = {
            "global": {}, "sector": {}, "industry": {},
        }
        self._meta: dict[str, dict[str, dict[str, dict[str, int]]]] = {
            "global": {}, "sector": {}, "industry": {},
        }
        self._rt: dict[str, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]] = {
            "global": {}, "sector": {}, "industry": {},
        }
        from collections import deque
        self._daily_counts: deque[list] = deque()
        self._current_window_size: int = 0
        self._auto_clip = self.cfg.llr_clip
        self._auto_weights = {}
        self._level_budgets = {}
        self.params_history = []

    # ── public API ────────────────────────────────────────────

    def fit_window(self, window_ranks: dict[str, pd.DataFrame]) -> None:
        """Fit one full rolling window from scratch."""
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
            if self.cfg.use_sector:
                df["sector"] = df["industry_sw"].map(_sector_for)

            for level, gcol in self._level_groups():
                table, groups, meta = self._estimate(df, level=level, group_col=gcol)
                self._t[level][m] = (table, groups)
                self._meta[level][m] = meta

        self._finalize_tables()

    # ── incremental update ────────────────────────────────────

    def update_incremental(
        self,
        new_day_ranks: dict[str, pd.DataFrame],
        drop_day_ranks: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """+new_day counts, -drop_day counts, recompute all LLR tables."""
        if not self._rt["global"]:
            self._init_from_window(new_day_ranks)
            return
        for m in self.models:
            new_df = self._day_df(new_day_ranks, m)
            drop_df = self._day_df(drop_day_ranks, m) if drop_day_ranks else None
            for level, gcol in self._level_groups():
                self._update_level(level, m, new_df, drop_df, gcol)
        self._recompute_all_tables()

    def _init_from_window(self, ranks: dict[str, pd.DataFrame]) -> None:
        for m in self.models:
            df = self._day_df(ranks, m)
            if df is None:
                continue
            for level, gcol in self._level_groups():
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
        if ranks_dict is None or model not in ranks_dict:
            return None
        df = ranks_dict[model].copy()
        if {"rank_pct", "H"} - set(df.columns):
            return None
        df["_bin"] = self._rank_to_bin(df["rank_pct"].to_numpy(dtype=float))
        df["H"] = df["H"].astype(int).clip(0, 1)
        df["industry_sw"] = _clean_industry(df.get("industry_sw", pd.Series(["-1"] * len(df))))
        if self.cfg.use_sector:
            df["sector"] = df["industry_sw"].map(_sector_for)
        return df

    def _update_level(self, level, model, new_df, drop_df, group_col):
        rt_m = self._rt[level].get(model, {})
        for df, sign in [(new_df, +1), (drop_df, -1)]:
            if df is None:
                continue
            h, b = df["H"].to_numpy(dtype=int), df["_bin"].to_numpy(dtype=int)
            groups = np.full(len(df), self.GLOBAL_KEY) if group_col is None else df[group_col].astype(str).to_numpy()
            for grp in np.unique(groups):
                msk = (groups == grp)
                head = np.bincount(b[msk & (h == 1)], minlength=self.n_bins).astype(float) * sign
                not_h = np.bincount(b[msk & (h == 0)], minlength=self.n_bins).astype(float) * sign
                k = str(grp)
                if k not in rt_m:
                    rt_m[k] = (np.zeros(self.n_bins), np.zeros(self.n_bins))
                rt_m[k] = (
                    np.maximum(0, rt_m[k][0] + head),
                    np.maximum(0, rt_m[k][1] + not_h),
                )
        self._rt[level][model] = rt_m

    def _recompute_all_tables(self) -> None:
        for level in self._levels:
            for m in self.models:
                rm = self._rt[level].get(m, {})
                if not rm:
                    continue
                grps = sorted(rm.keys())
                n_g = len(grps)
                tbl = np.zeros((self.n_bins, n_g), dtype=float)
                mt: dict[str, dict] = {}
                for gi, grp in enumerate(grps):
                    hd, nh = rm[grp]
                    hn, nn = int(hd.sum()), int(nh.sum())
                    total = hn + nn
                    alpha = self._alpha_for(level, grp, total)
                    ph = np.maximum(1e-12, (hd + alpha) / (hn + alpha * self.n_bins))
                    pn = np.maximum(1e-12, (nh + alpha) / (nn + alpha * self.n_bins))
                    llr = np.log(ph / pn)
                    tbl[:, gi] = llr
                    mt[grp] = {"total": total, "head": hn, "not_head": nn}
                self._t[level][m] = (tbl, grps)
                self._meta[level][m] = mt
        self._finalize_tables()

    def _level_groups(self):
        pairs = [("global", None)]
        if self.cfg.use_sector:
            pairs.append(("sector", "sector"))
        pairs.append(("industry", "industry_sw"))
        return pairs

    # ── auto-parameter calibration ──────────────────────────────

    def _finalize_tables(self) -> None:
        """Compute auto_clip + auto_weights from raw LLR tables, then clip in-place."""
        all_vals = []
        for level in self._levels:
            for m in self.models:
                tbl, _ = self._t[level].get(m, (None, None))
                if tbl is not None:
                    all_vals.append(tbl.ravel())

        if all_vals:
            combined = np.concatenate(all_vals)
            finite = combined[np.isfinite(combined)]
            if self.cfg.auto_clip and len(finite) > 0:
                self._auto_clip = max(0.5, float(np.percentile(np.abs(finite), self.cfg.clip_percentile)))
            else:
                self._auto_clip = float(self.cfg.llr_clip)
        else:
            self._auto_clip = float(self.cfg.llr_clip)

        # Clip all tables after using raw LLR to determine the cap.
        for level in self._levels:
            for m in self.models:
                if m in self._t[level]:
                    tbl, grps = self._t[level][m]
                    if tbl is not None:
                        self._t[level][m] = (np.clip(tbl, -self._auto_clip, self._auto_clip), grps)

        if self.cfg.auto_weights:
            self._auto_weights = self._compute_auto_weights()
        else:
            self._auto_weights = {}
            self._level_budgets = {level: 1.0 / len(self._levels) for level in self._levels}

        self.params_history.append({
            "auto_clip": float(self._auto_clip),
            "level_budgets": dict(self._level_budgets),
            "auto_weights": {
                m: {
                    "global": float(self._auto_weights.get(m, {}).get("strong", (0, 0, 0))[0]),
                    "sector": float(self._auto_weights.get(m, {}).get("strong", (0, 0, 0))[1]) if self.cfg.use_sector else 0.0,
                    "industry": float(self._auto_weights.get(m, {}).get("strong", (0, 0, 0))[2]),
                }
                for m in self.models
            } if self.cfg.auto_weights else {},
        })

    def _top_bin_indices_and_weights(self) -> tuple[list[int], np.ndarray]:
        """Top 2 bins only — strong + elite. Buffer bin excluded (noisy LLR)."""
        top_bins = [self.n_bins - 2, self.n_bins - 1]
        return top_bins, np.array([1.0, 1.0], dtype=float)

    def _compute_auto_weights(self) -> dict:
        """Auto model-level hierarchy weights with level budget gate.

        Step 1: compute positive count-weighted top-bin LLR strength for each
                (model, level).
        Step 2: compute B(level), the whole hierarchy level budget.
        Step 3: compute q(model | level), across-model per-level allocation.
        Step 4: final weight(model, level) = B(level) * q(model | level).

        This preserves vertical normalisation, so weak models do not have to
        speak, while also preventing weak hierarchy levels from being forced to
        speak.
        """
        top_bins, top_bin_weights = self._top_bin_indices_and_weights()

        raw: dict[str, dict[str, float]] = {m: {l: 0.0 for l in self._levels} for m in self.models}
        for m in self.models:
            for level in self._levels:
                tbl, grps = self._t[level].get(m, (None, None))
                if tbl is None or not grps:
                    continue
                meta = self._meta[level].get(m, {})
                llr_sum = 0.0
                total_w = 0.0
                for gi, grp in enumerate(grps):
                    count = meta.get(str(grp), {}).get("total", 0)
                    if count <= 0:
                        continue
                    for bi, bw in zip(top_bins, top_bin_weights):
                        llr_val = tbl[bi, gi]
                        if np.isfinite(llr_val):
                            llr_sum += count * float(bw) * max(float(llr_val), 0.0)
                            total_w += count * float(bw)
                raw[m][level] = llr_sum / total_w if total_w > 0 else 0.0

        level_strength = {level: sum(max(0.0, raw[m][level]) for m in self.models) for level in self._levels}
        total_level_strength = sum(level_strength.values())

        if self.cfg.level_budget_gate and total_level_strength > 0:
            level_budget = {level: level_strength[level] / total_level_strength for level in self._levels}
        else:
            bg, bs, bi = self.cfg.level_budget
            level_budget = {"global": bg, "sector": bs, "industry": bi}
        # Normalize
        s = sum(level_budget.values())
        if s > 0:
            level_budget = {k: v / s for k, v in level_budget.items()}
        self._level_budgets = level_budget

        q_model_given_level: dict[str, dict[str, float]] = {m: {l: 0.0 for l in self._levels} for m in self.models}
        for level in self._levels:
            denom = level_strength[level]
            if denom <= 0:
                continue
            for m in self.models:
                q_model_given_level[m][level] = max(0.0, raw[m][level]) / denom

        weights: dict[str, dict[str, tuple[float, float, float]]] = {}
        for m in self.models:
            wg = level_budget.get("global", 0.0) * q_model_given_level[m].get("global", 0.0)
            ws = level_budget.get("sector", 0.0) * q_model_given_level[m].get("sector", 0.0)
            wi = level_budget.get("industry", 0.0) * q_model_given_level[m].get("industry", 0.0)
            if not self.cfg.use_sector:
                ws = 0.0
            weights[m] = {
                "strong":      (float(wg), float(ws), float(wi)),
                "weak":        (float(wg), float(ws), float(wi)),
                "small":       (float(wg), float(ws), float(wi)),
                "global_only": (float(wg), 0.0, 0.0),
            }
        return weights

    # ── scoring ─────────────────────────────────────────────────

    def score(self, today: pd.DataFrame, today_ranks: dict[str, pd.DataFrame]) -> pd.Series:
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

    def score_detailed(self, today: pd.DataFrame, today_ranks: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Like score(), but returns per-model LLR breakdown for diagnostics."""
        base = today[["time", "stock_id", "industry_sw"]].drop_duplicates().copy()
        base["stock_id"] = base["stock_id"].astype(str).str.strip().str.zfill(6)
        base["industry_sw"] = _clean_industry(base["industry_sw"])
        if self.cfg.use_sector:
            base["sector"] = base["industry_sw"].map(_sector_for)
        base = base.reset_index(drop=True)
        n = len(base)
        industries = base["industry_sw"].to_numpy(dtype=str)
        sectors = base["sector"].to_numpy(dtype=str) if self.cfg.use_sector else np.full(n, "")
        is_go = np.isin(industries.astype(str), list(GLOBAL_ONLY))

        cols = {}
        total = np.zeros(n, dtype=float)
        for m in self.models:
            r = today_ranks[m][["time", "stock_id", "rank_pct"]].copy()
            r["stock_id"] = r["stock_id"].astype(str).str.strip().str.zfill(6)
            merged = base[["time", "stock_id"]].merge(r, on=["time", "stock_id"], how="left")
            ranks = merged["rank_pct"].fillna(0.5).to_numpy(dtype=float)
            bins = self._rank_to_bin(ranks)
            cols[f"{m}_rank"] = ranks

            gt, gg = self._t["global"].get(m, (None, []))
            it, ig = self._t["industry"].get(m, (None, []))
            gl = self._lookup_many("global", m, gt, gg, np.full(n, self.GLOBAL_KEY), bins, np.zeros(n), False)
            if self.cfg.use_sector:
                st, sg = self._t["sector"].get(m, (None, []))
                sl = self._lookup_many("sector", m, st, sg, sectors, bins, gl, True)
                il = self._lookup_many("industry", m, it, ig, industries, bins, sl, True)
            else:
                sl = np.zeros(n, dtype=float)
                il = self._lookup_many("industry", m, it, ig, industries, bins, gl, True)

            # global-only names cannot inherit sector/industry through fallback.
            sl[is_go] = 0.0
            il[is_go] = 0.0

            cols[f"{m}_llr_global"] = gl
            cols[f"{m}_llr_sector"] = sl
            cols[f"{m}_llr_industry"] = il

            wmap = self._auto_weights if (self.cfg.auto_weights and self._auto_weights) else self.cfg.model_weights
            wg, ws, wi = self._weights_many(m, industries, wmap)
            if not self.cfg.use_sector:
                wg = wg + ws
                ws = np.zeros_like(ws)
            contrib = wg * gl + ws * sl + wi * il
            cols[f"{m}_llr_blended"] = contrib
            total += contrib

        out = pd.DataFrame(cols, index=base.index)
        out["time"] = base["time"].values
        out["stock_id"] = base["stock_id"].values
        out["industry_sw"] = industries
        if self.cfg.use_sector:
            out["sector"] = sectors
        out["bayes_score"] = total
        base_cols = ["time", "stock_id", "industry_sw"]
        if self.cfg.use_sector:
            base_cols.append("sector")
        base_cols.append("bayes_score")
        return out[base_cols +
                   [f"{m}_rank" for m in self.models] +
                   [f"{m}_llr_global" for m in self.models] +
                   [f"{m}_llr_sector" for m in self.models] +
                   [f"{m}_llr_industry" for m in self.models] +
                   [f"{m}_llr_blended" for m in self.models]]

    def diagnostic_report(self) -> pd.DataFrame:
        """Per-group per-model per-bin diagnostics from current calibration.

        exp(LLR) is a likelihood ratio, not posterior odds. P_H_given_bin below
        is computed as prior_odds * exp(LLR) converted back to probability.
        """
        rows = []
        for level_name in ["global", "sector", "industry"]:
            if level_name not in self._levels:
                continue
            for m in self.models:
                table, groups = self._t[level_name].get(m, (None, None))
                if table is None:
                    continue
                meta = self._meta[level_name].get(m, {})
                for gi, grp in enumerate(groups):
                    gmeta = meta.get(str(grp), {})
                    head = int(gmeta.get("head", 0))
                    total = int(gmeta.get("total", 0))
                    base_rate = float(head / total) if total > 0 else np.nan
                    if np.isfinite(base_rate):
                        base_rate = min(max(base_rate, 1e-12), 1.0 - 1e-12)
                        prior_odds = base_rate / (1.0 - base_rate)
                    else:
                        prior_odds = np.nan
                    for bi in range(self.n_bins):
                        llr = float(table[bi, gi])
                        likelihood_ratio = float(np.exp(llr)) if np.isfinite(llr) else np.nan
                        if np.isfinite(prior_odds) and np.isfinite(likelihood_ratio):
                            posterior_odds = prior_odds * likelihood_ratio
                            p_h = posterior_odds / (1.0 + posterior_odds)
                        else:
                            p_h = np.nan
                        rows.append({
                            "level": level_name,
                            "model": m,
                            "group": grp,
                            "bin": bi,
                            "bin_lo": self.rank_bins[bi],
                            "bin_hi": self.rank_bins[bi + 1],
                            "llr": llr,
                            "likelihood_ratio": likelihood_ratio,
                            "base_head_rate": base_rate,
                            "P_H_given_bin": round(float(p_h), 6) if np.isfinite(p_h) else np.nan,
                        })
        return pd.DataFrame(rows)

    # ── per-model contribution ─────────────────────────────────

    def _model_contrib(self, model: str, inds: np.ndarray, secs: np.ndarray, bins: np.ndarray) -> np.ndarray:
        n = len(inds)

        gt, gg = self._t["global"].get(model, (None, []))
        it, ig = self._t["industry"].get(model, (None, []))

        gl = self._lookup_many(
            level="global", model=model, table=gt, groups=gg,
            keys=np.full(n, self.GLOBAL_KEY, dtype=object), bins=bins,
            fallback=np.zeros(n, dtype=float), require_quality=False,
        )

        if self.cfg.use_sector:
            st, sg = self._t["sector"].get(model, (None, []))
            sl = self._lookup_many(
                level="sector", model=model, table=st, groups=sg,
                keys=secs.astype(str), bins=bins,
                fallback=gl, require_quality=True,
            )
            il = self._lookup_many(
                level="industry", model=model, table=it, groups=ig,
                keys=inds.astype(str), bins=bins,
                fallback=sl, require_quality=True,
            )
        else:
            sl = np.zeros(n, dtype=float)
            il = self._lookup_many(
                level="industry", model=model, table=it, groups=ig,
                keys=inds.astype(str), bins=bins,
                fallback=gl, require_quality=True,
            )

        is_go = np.isin(inds.astype(str), list(GLOBAL_ONLY))
        sl[is_go] = 0.0
        il[is_go] = 0.0

        wmap = self._auto_weights if (self.cfg.auto_weights and self._auto_weights) else self.cfg.model_weights
        wg, ws, wi = self._weights_many(model, inds.astype(str), wmap)
        if not self.cfg.use_sector:
            wg = wg + ws
            ws = np.zeros_like(ws)
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
    def _weights_many(model: str, inds: np.ndarray, weight_map: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wmap = weight_map or {}
        model_w = wmap.get(model, {})

        def _w(ind):
            i = str(ind)
            if i in STRONG:
                return model_w.get("strong", W_STRONG)
            if i in WEAK:
                return model_w.get("weak", W_WEAK)
            if i in SMALL:
                return model_w.get("small", W_SMALL)
            if i in GLOBAL_ONLY:
                return model_w.get("global_only", W_GLOBAL_ONLY)
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
        return m["total"] >= self.cfg.min_group_total and m["head"] >= self.cfg.min_group_head

    # ── estimation ─────────────────────────────────────────────

    def _estimate(self, df: pd.DataFrame, level: str, group_col: str | None) -> tuple[np.ndarray | None, list[str], dict[str, dict[str, int]]]:
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

            alpha = self._alpha_for(level, str(grp), total)
            p_h = (head_b + alpha) / (head_n + alpha * nb)
            p_n = (not_b + alpha) / (not_n + alpha * nb)

            llr = np.log(p_h / p_n)
            table[:, gi] = llr
            meta[str(grp)] = {"total": total, "head": head_n, "not_head": not_n}

        return table, [str(g) for g in groups], meta

    def _alpha_for(self, level: str, group: str, total: int = 0) -> float:
        if not self.cfg.auto_alpha:
            if level == "industry" and str(group) in STRONG:
                return float(self.cfg.laplace_alpha_strong)
            return float(self.cfg.laplace_alpha_other)
        if total <= 0:
            return 20.0
        if level == "global":
            return 20.0
        if level == "sector":
            alpha = 1000.0 / np.sqrt(total)
            return float(np.clip(alpha, 10.0, 20.0))
        if level == "industry":
            alpha = 1000.0 / np.sqrt(total)
            if str(group) in STRONG:
                return float(np.clip(alpha, 8.0, 15.0))
            return float(np.clip(alpha, 15.0, 25.0))
        return 20.0

    # ── helpers ────────────────────────────────────────────────

    def _rank_to_bin(self, v: np.ndarray) -> np.ndarray:
        v = np.clip(np.asarray(v, dtype=float), self.rank_bins[0], self.rank_bins[-1])
        idx = np.digitize(v, self.rank_bins[1:-1], right=True)
        return np.clip(idx, 0, self.n_bins - 1).astype(int)


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _clean_industry(s: pd.Series) -> pd.Series:
    def _fmt(x):
        try:
            # industry_sw may come as 270000, 27, 27.0, or strings.
            sx = str(x).strip()
            if sx.endswith(".0"):
                sx = sx[:-2]
            if sx.isdigit() and len(sx) >= 6:
                return sx[:2]
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
    rolling windows.
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
        window = merged_ranked[(merged_ranked["time"] >= w_start) & (merged_ranked["time"] <= w_end)]

        w_ranks = {}
        for m in models:
            rcol = f"{m}_r"
            w_ranks[m] = window[["time", "stock_id", "industry_sw", rcol, "H"]].rename(columns={rcol: "rank_pct"})

        cal.fit_window(w_ranks)

        if i < effective_burnin:
            continue
        if eval_date_set is not None and pd.Timestamp(t) not in eval_date_set:
            continue

        today = merged_ranked.loc[merged_ranked["time"] == t, ["time", "stock_id", "industry_sw"]]
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
        f"  rolling={cfg.rolling_window}, auto_alpha={cfg.auto_alpha}, "
        f"auto_clip={cfg.auto_clip}, clip_pct={cfg.clip_percentile}, "
        f"clip_now={cal._auto_clip:.4f}"
    )
