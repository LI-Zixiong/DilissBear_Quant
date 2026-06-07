"""
Factor audit: ICIR ranking + pairwise correlation + automatic selection.

Usage:
    python -m scripts.factor.audit_factors
    python -m scripts.factor.audit_factors --target 1d_next_raw --icir-min 0.10 --corr-max 0.80
    python -m scripts.factor.audit_factors --select-start 2018-01-01 --select-end 2023-12-31

Outputs (all in --output-dir):
    icir_ranking.csv        — all factors sorted by abs(ICIR), with sign/missing_rate/std
    correlation_matrix.csv   — pairwise Spearman correlation
    selection_report.csv    — kept/dropped with reasons
    selected_factors.txt    — final factor list (copy-paste into ExperimentConfig)
    summary.txt             — human-readable summary
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


# ── Defaults ──────────────────────────────────────────────
DEF_FACTOR_PANEL = "dataset/processed/factor_panel_54_ind.parquet"
DEF_TARGET = "5d_next_raw"
DEF_OUTPUT_DIR = "reports/factor_audit"
DEF_ICIR_MIN = 0.15
DEF_CORR_MAX = 0.85
DEF_DROP_MISSING = 0.50
DEF_MIN_UNIQUE = 20
# ──────────────────────────────────────────────────────────


def compute_factor_icir(fp, factor_names, target):
    """Per-factor daily rank IC -> ICIR. Adds missing_rate and std."""
    rows = []
    for f in factor_names:
        missing_rate = fp[f].isna().mean()
        std_val = fp[f].std()
        n_unique = fp[f].nunique(dropna=True)

        sub = fp[["time", "stock_id", f, target]].dropna(subset=[f, target])
        if len(sub) < 1000 or n_unique < DEF_MIN_UNIQUE or std_val < 1e-8:
            rows.append({
                "factor": f, "ic_mean": np.nan, "ic_std": np.nan,
                "icir": np.nan, "pos_rate": np.nan, "n_dates": 0,
                "sign": 0, "missing_rate": round(missing_rate, 4),
                "std": round(std_val, 6), "n_unique": n_unique,
            })
            continue
        daily_ic = []
        for t, g in sub.groupby("time"):
            if len(g) >= 30 and g[f].nunique() >= 2:
                ic = g[f].corr(g[target], method="spearman")
                daily_ic.append(ic)
        ic = pd.Series(daily_ic).dropna()
        if len(ic) >= 10:
            icir_val = ic.mean() / ic.std() if ic.std() > 0 else 0
            rows.append({
                "factor": f,
                "ic_mean": round(ic.mean(), 6),
                "ic_std": round(ic.std(), 6),
                "icir": round(icir_val, 4),
                "pos_rate": round((ic > 0).mean(), 4),
                "n_dates": len(ic),
                "sign": 1 if icir_val >= 0 else -1,
                "missing_rate": round(missing_rate, 4),
                "std": round(std_val, 6),
                "n_unique": n_unique,
            })
        else:
            rows.append({
                "factor": f, "ic_mean": np.nan, "ic_std": np.nan,
                "icir": np.nan, "pos_rate": np.nan, "n_dates": 0,
                "sign": 0, "missing_rate": round(missing_rate, 4),
                "std": round(std_val, 6), "n_unique": n_unique,
            })
    return pd.DataFrame(rows).sort_values(
        "icir", key=abs, ascending=False, na_position="last"
    )


def compute_corr_matrix(fp, factor_names, max_rows=100000, min_periods=1000):
    """Pairwise Spearman correlation with min_periods (no full dropna)."""
    data = fp[factor_names]
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=42)
    return data.corr(method="spearman", min_periods=min_periods)


def select_factors(icir_df, corr_df, icir_min, corr_max, drop_missing):
    """Apply quality + ICIR + corr pruning. Returns (kept, dropped, reasons)."""
    reasons = {}
    status = {}

    # Step 0: missing rate / constant check
    for _, row in icir_df.iterrows():
        f = row["factor"]
        if pd.isna(row["icir"]):
            reasons[f] = "cannot compute ICIR (insufficient data)"
            status[f] = "DROPPED"
        elif row["missing_rate"] > drop_missing:
            reasons[f] = f"missing_rate={row['missing_rate']:.1%} > {drop_missing:.0%}"
            status[f] = "DROPPED"
        elif row.get("std", 1) < 1e-8:
            reasons[f] = "near-constant factor"
            status[f] = "DROPPED"
        elif abs(row["icir"]) < icir_min:
            reasons[f] = f"abs(ICIR)={abs(row['icir']):.4f} < {icir_min}"
            status[f] = "DROPPED"
        else:
            status[f] = "CANDIDATE"

    # Step 1: Corr pruning (greedy, higher ICIR survives)
    candidates = sorted(
        [f for f, s in status.items() if s == "CANDIDATE"],
        key=lambda f: abs(icir_df.set_index("factor").loc[f, "icir"]),
        reverse=True,
    )
    for i, f1 in enumerate(candidates):
        if status[f1] == "DROPPED":
            continue
        for f2 in candidates[i + 1:]:
            if status[f2] == "DROPPED":
                continue
            if f1 not in corr_df.index or f2 not in corr_df.columns:
                continue
            c = corr_df.loc[f1, f2]
            if abs(c) > corr_max:
                ic1 = abs(icir_df.set_index("factor").loc[f1, "icir"])
                ic2 = abs(icir_df.set_index("factor").loc[f2, "icir"])
                loser = f2 if ic1 >= ic2 else f1
                winner = f1 if ic1 >= ic2 else f2
                reasons[loser] = (
                    f"corr={c:.3f} with {winner} "
                    f"(abs_icir={min(ic1,ic2):.4f} < {max(ic1,ic2):.4f})"
                )
                status[loser] = "DROPPED"

    for f, s in status.items():
        if s == "CANDIDATE":
            status[f] = "KEPT"
            reasons[f] = "passes all filters"

    kept = [f for f, s in status.items() if s == "KEPT"]
    dropped = [f for f, s in status.items() if s == "DROPPED"]
    return kept, dropped, reasons


def main():
    parser = argparse.ArgumentParser(description="Factor audit and selection")
    parser.add_argument("--factor-panel", default=DEF_FACTOR_PANEL)
    parser.add_argument("--target", default=DEF_TARGET)
    parser.add_argument("--output-dir", default=DEF_OUTPUT_DIR)
    parser.add_argument("--icir-min", type=float, default=DEF_ICIR_MIN)
    parser.add_argument("--corr-max", type=float, default=DEF_CORR_MAX)
    parser.add_argument("--drop-missing", type=float, default=DEF_DROP_MISSING)
    parser.add_argument("--select-start", default=None)
    parser.add_argument("--select-end", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading factor panel: {args.factor_panel}")
    fp = pd.read_parquet(args.factor_panel)
    fp["time"] = pd.to_datetime(fp["time"])
    fp["stock_id"] = fp["stock_id"].astype(str).str.strip().str.zfill(6)

    # Date filtering
    if args.select_start:
        fp = fp[fp["time"] >= pd.Timestamp(args.select_start)]
        print(f"  Date filter: >= {args.select_start}")
    if args.select_end:
        fp = fp[fp["time"] <= pd.Timestamp(args.select_end)]
        print(f"  Date filter: <= {args.select_end}")
    if args.select_end is None:
        print("  WARNING: --select-end is None. Factor selection may include future/test data.")

    factor_names = sorted(c for c in fp.columns if c.startswith("F0"))
    print(f"  Found {len(factor_names)} F-format factors")
    print(f"  Target: {args.target}")
    print(f"  ICIR min: {args.icir_min}  Corr max: {args.corr_max}")

    # 1. ICIR
    print("\n--- ICIR ---")
    icir = compute_factor_icir(fp, factor_names, args.target)
    icir_path = output_dir / "icir_ranking.csv"
    icir.to_csv(icir_path, index=False)
    print(f"  Saved: {icir_path}")

    # 2. Correlation
    print("\n--- Correlation ---")
    corr = compute_corr_matrix(fp, factor_names)
    corr_path = output_dir / "correlation_matrix.csv"
    corr.to_csv(corr_path)
    print(f"  Saved: {corr_path}")

    # 3. Selection
    print("\n--- Selection ---")
    kept, dropped, reasons = select_factors(
        icir, corr, args.icir_min, args.corr_max, args.drop_missing
    )

    report_rows = []
    for f in factor_names:
        report_rows.append({
            "factor": f,
            "status": "KEPT" if f in kept else "DROPPED",
            "icir": icir.set_index("factor").loc[f, "icir"] if f in icir["factor"].values else np.nan,
            "sign": icir.set_index("factor").loc[f, "sign"] if f in icir["factor"].values else 0,
            "reason": reasons.get(f, "unknown"),
        })
    report = pd.DataFrame(report_rows)
    report_path = output_dir / "selection_report.csv"
    report.to_csv(report_path, index=False)
    print(f"  Saved: {report_path}")

    # 4. Final factor list
    selected_path = output_dir / "selected_factors.txt"
    selected_path.write_text(",\n".join(f'"{f}"' for f in kept) + "\n")
    print(f"  Saved: {selected_path}")

    # 5. Summary
    summary_lines = [
        f"Factor Audit Summary",
        f"====================",
        f"Panel: {args.factor_panel}",
        f"Target: {args.target}",
        f"Date range: {args.select_start or 'full'} → {args.select_end or 'full'}",
        f"Total factors: {len(factor_names)}",
        f"ICIR threshold: abs(ICIR) >= {args.icir_min}",
        f"Corr threshold: |corr| <= {args.corr_max}",
        f"Missing rate max: {args.drop_missing}",
        f"",
        f"Kept: {len(kept)}",
        f"Dropped: {len(dropped)}",
        f"",
        f"NOTE: Negative ICIR factors have sign=-1 in selection_report.csv.",
        f"      If used by linear/rank models, sign-flip before training.",
        f"      LGBM does not need sign-flip (trees handle direction).",
        f"",
        f"--- Kept ---",
    ]
    for f in kept:
        ic = icir.set_index("factor").loc[f, "icir"]
        s = icir.set_index("factor").loc[f, "sign"]
        summary_lines.append(f"  {f:>16s}  ICIR={ic:+.4f}  sign={int(s):+d}")
    summary_lines.append(f"\n--- Dropped ---")
    for f in dropped:
        summary_lines.append(f"  {f:>16s}  {reasons.get(f, 'unknown')}")
    summary_lines.append(f"\n--- Final factor list (copy into ExperimentConfig) ---")
    summary_lines.append(f"# {len(kept)} factors, target={args.target}")
    summary_lines.append("feature_cols=(")
    for f in kept:
        summary_lines.append(f'    "{f}",')
    summary_lines.append(")")
    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"  Saved: {summary_path}")

    print(f"\nDone. {len(kept)} factors kept, {len(dropped)} dropped.")


if __name__ == "__main__":
    main()
