"""
Four-model factor importance diagnostic.

Tree models (LightGBM/XGBoost):
    built-in gain-based feature importance.

Torch models (DLinear/GatedDW-TCN):
    permutation importance. For each selected factor f:
        X_perm = X.copy()
        X_perm[:, :, f] = X[perm, :, f]
        y_perm = model(X_perm)
        backtest(y_perm)
        importance = baseline_sharpe - permuted_sharpe

Designed for Big Bear / JulongQuant experiment_005.

Usage examples
--------------
# Tree gain + DLinear/GatedDW permutation on top 30 IC candidates:
python -m scripts.evaluation.diagnose_factor_importance \
  --experiment-dir dataset/output/experiment_005 \
  --data-path dataset/processed/factor_panel_1500_54_ind.parquet \
  --icir-csv dataset/output/experiment_005/icir_ranking.csv \
  --torch-top-n 30 \
  --ic-threshold 0.03

# Only DLinear, permutation:
python -m scripts.evaluation.diagnose_factor_importance \
  --models dlinear \
  --torch-top-n 30

# Full torch permutation on all factors:
python -m scripts.evaluation.diagnose_factor_importance \
  --models dlinear gated_dwtcn \
  --torch-factor-mode all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.experiment.config import ExperimentConfig
from src.data.dataset_builder import PanelDatasetBuilder
from src.experiment.data import prepare_experiment_data
from src.backtest.ensemble_utils import normalize_keys


MODELS = ("lightgbm", "xgboost", "dlinear", "gated_dwtcn")
TREE_MODELS = {"lightgbm", "xgboost"}
TORCH_MODELS = {"dlinear", "gated_dwtcn"}

DEFAULT_META_COLS = ["industry_sw", "list_date", "ret_daily"]


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------


def factor_columns_from_panel(path: str | Path) -> list[str]:
    """Read only parquet metadata/columns when possible and return sorted F columns."""
    # pandas has to open the file; for typical parquet this is still cheap enough.
    cols = list(pd.read_parquet(path, columns=None).columns)
    return sorted([c for c in cols if c.startswith("F")])


def read_feature_cols(data_path: str | Path, feature_cols_json: str | None = None) -> list[str]:
    """Resolve actual factor columns. Prefer explicit json if supplied."""
    if feature_cols_json:
        obj = json.loads(Path(feature_cols_json).read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            for key in ("feature_cols", "features", "factor_names"):
                if key in obj:
                    return list(obj[key])
        if isinstance(obj, list):
            return list(obj)
        raise ValueError(f"Unsupported feature cols json format: {feature_cols_json}")

    panel = pd.read_parquet(data_path)
    cols = sorted([c for c in panel.columns if c.startswith("F")])
    del panel
    if not cols:
        raise ValueError(f"No F* factor columns found in {data_path}")
    return cols


def safe_mkdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p



def evaluate_prediction_frame(pred_df: pd.DataFrame, data: Any, args: argparse.Namespace) -> tuple[float, Any]:
    """Align predictions and run backtest; return Sharpe and full summary."""
    from src.experiment.returns import align_predictions_to_returns
    from src.backtest.engine import TransactionCostConfig, run_backtest
    from src.backtest.portfolio import PortfolioConfig
    pred_df = pred_df.copy()
    pred_df["time"] = pd.to_datetime(pred_df["time"])
    pred_df["stock_id"] = pred_df["stock_id"].astype(str).str.zfill(6)
    aligned = align_predictions_to_returns(
        pred_df=pred_df, returns_df=data.returns_df,
        date_col="time", stock_col="stock_id", pred_col="y_pred", return_col="return_1d",
    )
    pfolio = PortfolioConfig(strategy="top_n", top_n=args.top_n, pred_col="y_pred", stock_col="stock_id")
    cost_config = TransactionCostConfig()
    summary = run_backtest(
        aligned, data.returns_df, pfolio,
        return_col="return_1d", date_col="time", stock_col="stock_id",
        cost_config=cost_config,
    )["summary"]
    sr = float(summary.get("sharpe_ratio", np.nan))
    return sr, summary


def load_predictions(exp_dir: str | Path, model: str) -> pd.DataFrame:
    """Load saved model prediction parquet if present."""
    exp = Path(exp_dir)
    candidates = [
        exp / f"predictions_{model}.parquet",
        exp / f"predictions_test_{model}.parquet",
        exp / "predictions" / f"{model}.parquet",
        exp / "predictions" / f"predictions_{model}.parquet",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(p)
            return normalize_keys(df)
    raise FileNotFoundError(f"Cannot find predictions for {model}. Tried: {candidates}")


# ---------------------------------------------------------------------
# Factor name resolution
# ---------------------------------------------------------------------


def _is_generic_name(name: str) -> bool:
    """Return True if name looks like an auto-generated internal feature name."""
    return bool(
        name.startswith("Column_")
        or (name.startswith("f") and name[1:].isdigit())
        or (name.startswith("Feature_") and name[8:].isdigit())
    )


def _resolve_factor_names(
    raw_names: list[str],
    feature_cols: list[str],
    n_features: int,
) -> list[str]:
    """Map internal feature names (Column_X, f0, ...) to real factor names by index.

    If raw_names already use real factor names (e.g. F001SIZE), return them as-is.
    """
    if not raw_names:
        return feature_cols[:n_features]

    if _is_generic_name(raw_names[0]):
        if n_features > len(feature_cols):
            raise ValueError(
                f"Model has {n_features} features but only {len(feature_cols)} "
                f"factor names available. Cannot map by index."
            )
        return feature_cols[:n_features]

    return raw_names


# ---------------------------------------------------------------------
# Tree model gain importance
# ---------------------------------------------------------------------


def tree_importance(exp_dir: str | Path, model: str, feature_cols: Sequence[str]) -> pd.DataFrame:
    """Gain-based importance for LightGBM/XGBoost."""
    model_dir = Path(exp_dir) / "models" / model

    if model == "lightgbm":
        import lightgbm as lgb

        candidates = [
            model_dir / "LightGBMReturnRegressor.txt",
            model_dir / "model.txt",
            model_dir / "lightgbm.txt",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"LightGBM model file not found in {model_dir}")

        m = lgb.Booster(model_file=str(path))
        gain = m.feature_importance(importance_type="gain").astype(float)
        n_gain = len(gain)
        names = _resolve_factor_names(
            raw_names=list(m.feature_name()),
            feature_cols=list(feature_cols),
            n_features=n_gain,
        )

    elif model == "xgboost":
        import xgboost as xgb

        candidates = [
            model_dir / "XGBoostReturnRegressor.json",
            model_dir / "model.json",
            model_dir / "xgboost.json",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"XGBoost model file not found in {model_dir}")

        m = xgb.Booster()
        m.load_model(str(path))
        raw = m.get_score(importance_type="total_gain")

        n_features = len(feature_cols)
        gain = np.array([raw.get(f"f{i}", 0.0) for i in range(n_features)], dtype=float)
        names = list(feature_cols)

    else:
        raise ValueError(f"Unsupported tree model: {model}")

    total = float(np.nansum(gain))
    out = pd.DataFrame(
        {
            "model": model,
            "factor": names[: len(gain)],
            "gain": gain,
            "gain_pct": gain / total * 100.0 if total > 0 else 0.0,
        }
    )
    out = out.sort_values("gain", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out[["model", "rank", "factor", "gain", "gain_pct"]]


# ---------------------------------------------------------------------
# Torch model loading / inference
# ---------------------------------------------------------------------



def load_torch_model(exp_dir: str | Path, model: str, n_features: int, seq_len: int, device: str):
    """Load DLinear/GatedDW checkpoint and return eval model."""
    import torch

    model_dir = Path(exp_dir) / "models" / model
    ckpt_map = {
        "dlinear": model_dir / "DLinearPanelRegressor_best_composite.pt",
        "gated_dwtcn": model_dir / "GatedDWTcnRegressor_best_composite.pt",
    }
    ckpt_path = ckpt_map[model]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    # Filter to model weights only (exclude optimizer/epoch/feature_names)
    state = {k: v for k, v in ckpt.items()
             if not k.startswith("optimizer") and k not in ("epoch", "seq_len", "feature_names")}

    if model == "dlinear":
        from src.models.dlinear import DLinearPanelRegressor
        net = DLinearPanelRegressor(n_features=n_features, seq_len=seq_len)
    else:
        from src.models.gated_dwtcn import GatedDWTcnRegressor
        # Infer gate_rank from checkpoint (default=4, model was trained with rank=8)
        gate_rank = None
        for k, v in state.items():
            if "gate.A" in k and hasattr(v, "shape") and len(v.shape) == 2:
                gate_rank = v.shape[0]
                break
        net = GatedDWTcnRegressor(n_features=n_features, seq_len=seq_len, gate_rank=gate_rank or 8)
    net.load_state_dict(state, strict=False)
    net.to(device)
    net.eval()
    return net


def predict_torch(model_obj: Any, X: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    """Batched torch inference."""
    import torch

    preds: list[np.ndarray] = []
    n = X.shape[0]

    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
            out = model_obj(xb)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = out.detach().cpu().numpy().reshape(-1)
            preds.append(out)

    return np.concatenate(preds)


def get_built_meta(built: Any, data_test_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Extract predict-set meta dataframe robustly."""
    for attr in ("meta", "meta_df", "index_df", "id_df", "df"):
        if hasattr(built, attr):
            obj = getattr(built, attr)
            if isinstance(obj, pd.DataFrame) and {args.date_col, args.stock_col}.issubset(obj.columns):
                return normalize_keys(obj.copy())

    if isinstance(built, dict):
        for key in ("meta", "meta_df", "index_df", "id_df", "df"):
            obj = built.get(key)
            if isinstance(obj, pd.DataFrame) and {args.date_col, args.stock_col}.issubset(obj.columns):
                return normalize_keys(obj.copy())

    # Fallback: try to reconstruct from builder output length using last rows of test_df.
    raise ValueError(
        "Cannot find meta dataframe in PanelDatasetBuilder output. "
        "Please expose built.meta_df with time/stock_id columns."
    )


def build_torch_predict_set(data: Any, feature_cols: Sequence[str], args: argparse.Namespace):
    """Build torch predict set exactly like training pipeline."""
    builder = PanelDatasetBuilder(
        feature_cols=list(feature_cols),
        target_col=args.target_col,
        date_col=args.date_col,
        stock_col=args.stock_col,
        seq_len=args.seq_len,
        meta_cols=DEFAULT_META_COLS,
    )
    built = builder.build_sequence_dataset(data.test_df)
    X = built.X if hasattr(built, "X") else built["X"]
    meta = get_built_meta(built, data.test_df, args)
    return X.astype(np.float32, copy=False), meta


# ---------------------------------------------------------------------
# Torch permutation importance
# ---------------------------------------------------------------------


def choose_torch_factors(
    feature_cols: Sequence[str],
    args: argparse.Namespace,
) -> list[str]:
    """Choose factors for permutation."""
    feature_cols = list(feature_cols)

    if args.torch_factor_mode == "all":
        return feature_cols

    if args.factor_list:
        wanted = [x.strip() for x in args.factor_list.split(",") if x.strip()]
        missing = sorted(set(wanted) - set(feature_cols))
        if missing:
            raise ValueError(f"Requested factors not in feature_cols: {missing}")
        return wanted

    if args.icir_csv:
        icir_path = Path(args.icir_csv)
        if not icir_path.exists():
            raise FileNotFoundError(f"ICIR csv not found: {icir_path}")

        df = pd.read_csv(icir_path)

        # factor column
        factor_col = None
        for c in ("factor", "name", "factor_name", "feature"):
            if c in df.columns:
                factor_col = c
                break
        if factor_col is None:
            # fallback: first column with F-like values
            for c in df.columns:
                if df[c].astype(str).str.startswith("F").mean() > 0.5:
                    factor_col = c
                    break
        if factor_col is None:
            raise ValueError(f"Cannot infer factor column from {icir_path}: {df.columns.tolist()}")

        # score column
        score_col = None
        for c in ("rank_ic", "mean_ic", "IC", "ic", "ICIR", "icir", "abs_rank_ic"):
            if c in df.columns:
                score_col = c
                break
        if score_col is None:
            numeric_cols = [c for c in df.columns if c != factor_col and pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                raise ValueError(f"Cannot infer IC/ICIR score column from {icir_path}")
            score_col = numeric_cols[0]

        df = df[[factor_col, score_col]].rename(columns={factor_col: "factor", score_col: "score"})
        df["factor"] = df["factor"].astype(str)
        df = df[df["factor"].isin(feature_cols)].copy()
        df["abs_score"] = df["score"].abs()
        df = df[df["abs_score"] >= args.ic_threshold]
        df = df.sort_values("abs_score", ascending=False)

        if args.torch_top_n:
            df = df.head(args.torch_top_n)

        selected = df["factor"].tolist()
        if not selected:
            print("  [WARN] ICIR filter selected no factors; falling back to first torch_top_n feature columns.")
            return feature_cols[: args.torch_top_n or len(feature_cols)]
        return selected

    # Default: top N by feature order if no ICIR.
    return feature_cols[: args.torch_top_n or len(feature_cols)]


def torch_permutation_importance(
    exp_dir: str | Path,
    model: str,
    data: Any,
    feature_cols: Sequence[str],
    args: argparse.Namespace,
    out_file: str | Path | None = None,
) -> pd.DataFrame:
    """Permutation importance for DLinear/GatedDW. Importance = baseline SR - permuted SR."""
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Resume support ──
    old_rows: list[dict[str, Any]] = []
    completed_factors: set[str] = set()
    old_baseline_sr: float | None = None
    if args.resume and out_file and Path(out_file).exists():
        old = pd.read_csv(out_file)
        old_rows = old.to_dict("records")
        completed_factors = set(old["factor"].tolist())
        if "baseline_sharpe" in old.columns and len(old) > 0:
            old_baseline_sr = float(old["baseline_sharpe"].iloc[0])
        print(f"  Resuming: {len(completed_factors)} factors already done, skipping")

    print(f"  Building predict set: seq_len={args.seq_len}, features={len(feature_cols)}")
    X, meta = build_torch_predict_set(data, feature_cols, args)
    print(f"  X shape: {X.shape}; meta rows: {len(meta):,}")

    net = load_torch_model(exp_dir, model, len(feature_cols), args.seq_len, device=device)

    # Baseline — recompute or reuse
    if old_baseline_sr is not None:
        baseline_sr = old_baseline_sr
        baseline_summary = None
        print(f"  Reusing baseline Sharpe: {baseline_sr:.6f}")
    else:
        print("  Baseline inference...")
        t0 = time.perf_counter()
        baseline_pred = predict_torch(net, X, args.batch_size, device)
        print(f"  Baseline pred: mean={np.nanmean(baseline_pred):.6f} std={np.nanstd(baseline_pred):.6f} nan={np.isnan(baseline_pred).sum()} n={len(baseline_pred)}")
        pred_df = meta[[args.date_col, args.stock_col]].copy()
        pred_df["y_pred"] = baseline_pred
        baseline_sr, baseline_summary = evaluate_prediction_frame(pred_df, data, args)
        print(f"  Baseline Sharpe: {baseline_sr:.6f} ({time.perf_counter() - t0:.1f}s)")

    selected = choose_torch_factors(feature_cols, args)
    # Filter out already-completed factors
    if completed_factors:
        selected = [f for f in selected if f not in completed_factors]
        print(f"  Remaining: {len(selected)} factors")
    if not selected:
        print("  All factors done. Returning existing results.")
        out = pd.DataFrame(old_rows)
        out = out.sort_values("importance", ascending=False).reset_index(drop=True)
        out["rank"] = np.arange(1, len(out) + 1)
        return out[[
            "model", "rank", "factor", "feature_index", "importance",
            "sharpe_drop", "baseline_sharpe", "perm_sharpe_mean",
            "perm_sharpe_std", "n_repeats",
        ]]

    print(f"  Permuting {len(selected)} factors: {selected[:10]}{'...' if len(selected) > 10 else ''}")

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    n = X.shape[0]
    factor_to_idx = {f: i for i, f in enumerate(feature_cols)}

    for k, factor in enumerate(selected, 1):
        fi = factor_to_idx[factor]
        perm_srs = []
        t_factor = time.perf_counter()

        for rep in range(args.permutation_repeats):
            perm = rng.permutation(n)

            Xp = X.copy()
            # Per-timestep cross-sectional shuffle: break factor->return link
            # at each bar without breaking within-sample temporal structure.
            for t in range(X.shape[1]):
                perm_t = rng.permutation(n)
                Xp[:, t, fi] = X[perm_t, t, fi]

            y_perm = predict_torch(net, Xp, args.batch_size, device)
            p = meta[[args.date_col, args.stock_col]].copy()
            p["y_pred"] = y_perm
            sr, _summary = evaluate_prediction_frame(p, data, args)
            perm_srs.append(sr)

            del Xp, y_perm, p

        perm_sr_mean = float(np.nanmean(perm_srs))
        perm_sr_std = float(np.nanstd(perm_srs))
        sr_drop = float(baseline_sr - perm_sr_mean)

        rows.append(
            {
                "model": model,
                "factor": factor,
                "feature_index": fi,
                "baseline_sharpe": baseline_sr,
                "perm_sharpe_mean": perm_sr_mean,
                "perm_sharpe_std": perm_sr_std,
                "sharpe_drop": sr_drop,
                "importance": sr_drop,
                "n_repeats": args.permutation_repeats,
            }
        )

        print(
            f"  [{k:03d}/{len(selected):03d}] {factor:>18s} "
            f"perm_SR={perm_sr_mean:+.6f} drop={sr_drop:+.6f} "
            f"({time.perf_counter() - t_factor:.1f}s)",
            flush=True,
        )

        # Incremental save — safe to interrupt at any time
        if out_file:
            temp = pd.DataFrame(old_rows + rows)
            temp = temp.sort_values("importance", ascending=False).reset_index(drop=True)
            temp["rank"] = np.arange(1, len(temp) + 1)
            temp.to_csv(out_file, index=False)

    out = pd.DataFrame(old_rows + rows)
    out = out.sort_values("importance", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out[
        [
            "model",
            "rank",
            "factor",
            "feature_index",
            "importance",
            "sharpe_drop",
            "baseline_sharpe",
            "perm_sharpe_mean",
            "perm_sharpe_std",
            "n_repeats",
        ]
    ]


# ---------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------


def make_experiment_config(args: argparse.Namespace, feature_cols: Sequence[str]) -> ExperimentConfig:
    """Create ExperimentConfig with actual 100 F columns override."""
    cfg = ExperimentConfig(
        data_path=args.data_path,
        feature_cols=tuple(feature_cols),
        target_col=args.target_col,
        seq_len=args.seq_len,
    )
    return cfg


def save_top_print(imp: pd.DataFrame, model: str, n: int = 20) -> None:
    print(f"\nTop-{n} {model} factor importance:")
    if "gain_pct" in imp.columns:
        cols = ["rank", "factor", "gain_pct"]
    else:
        cols = ["rank", "factor", "importance", "perm_sharpe_mean"]
    print(imp.head(n)[cols].to_string(index=False))


def build_combined(out_dir: Path, outputs: list[pd.DataFrame]) -> pd.DataFrame:
    factors = sorted(set().union(*[set(x["factor"]) for x in outputs if "factor" in x]))
    combined = pd.DataFrame({"factor": factors})

    for imp in outputs:
        model = str(imp["model"].iloc[0])
        if "gain_pct" in imp.columns:
            metric = imp[["factor", "gain_pct"]].rename(columns={"gain_pct": f"{model}_gain_pct"})
            rank = imp[["factor", "rank"]].rename(columns={"rank": f"{model}_rank"})
        else:
            metric = imp[["factor", "importance"]].rename(columns={"importance": f"{model}_perm_sr_drop"})
            rank = imp[["factor", "rank"]].rename(columns={"rank": f"{model}_rank"})

        combined = combined.merge(metric, on="factor", how="left")
        combined = combined.merge(rank, on="factor", how="left")

    combined.to_csv(out_dir / "factor_importance_combined.csv", index=False)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment-dir", type=str, default="dataset/output/experiment_005")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="dataset/processed/factor_panel_1500_54_ind.parquet")
    parser.add_argument("--feature-cols-json", type=str, default=None)

    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--target-col", type=str, default="5d_next_raw")
    parser.add_argument("--date-col", type=str, default="time")
    parser.add_argument("--stock-col", type=str, default="stock_id")
    parser.add_argument("--seq-len", type=int, default=20)

    # Torch permutation controls
    parser.add_argument("--torch-factor-mode", type=str, default="icir_top", choices=["icir_top", "all", "list"])
    parser.add_argument("--factor-list", type=str, default=None, help="Comma-separated factors for torch permutation.")
    parser.add_argument("--icir-csv", type=str, default=None)
    parser.add_argument("--ic-threshold", type=float, default=0.03)
    parser.add_argument("--torch-top-n", type=int, default=30)
    parser.add_argument("--permutation-repeats", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file, skipping completed factors.")

    # Backtest controls
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--buy-fee", type=float, default=0.0003)
    parser.add_argument("--sell-fee", type=float, default=0.0008)
    parser.add_argument("--backtest-return-col", type=str, default="return_1d")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.experiment_dir)
    out_dir = safe_mkdir(args.output_dir or exp_dir)

    print("=" * 80)
    print("Four-model factor importance diagnostic")
    print("=" * 80)
    print(f"experiment_dir: {exp_dir}")
    print(f"data_path     : {args.data_path}")
    print(f"models        : {args.models}")

    feature_cols = read_feature_cols(args.data_path, args.feature_cols_json)
    print(f"feature_cols  : {len(feature_cols)}")
    if len(feature_cols) != 100:
        print(f"[WARN] feature_cols count is {len(feature_cols)}, not 100. Proceeding with actual panel columns.")

    cfg = make_experiment_config(args, feature_cols)

    # prepare_experiment_data is needed for torch permutation and also baseline backtest.
    # Loading once avoids repeated expensive work.
    need_torch = any(m in TORCH_MODELS for m in args.models)
    data = None
    if need_torch:
        print("\nPreparing experiment data...")
        data = prepare_experiment_data(cfg)

    outputs: list[pd.DataFrame] = []

    for model in args.models:
        print("\n" + "=" * 80)
        print(f"Model: {model}")
        print("=" * 80)
        t0 = time.perf_counter()

        out_file = out_dir / f"factor_importance_{model}.csv"

        # Per-model feature list: try {exp_dir}/model_features_{model}.json first
        model_fc = feature_cols
        model_json = exp_dir / f"model_features_{model}.json"
        if model_json.exists():
            import json as _json
            model_fc = tuple(_json.loads(model_json.read_text(encoding="utf-8")))
            print(f"  Using per-model features: {len(model_fc)} (from {model_json})")

        if model in TREE_MODELS:
            imp = tree_importance(exp_dir, model, model_fc)
        else:
            assert data is not None
            imp = torch_permutation_importance(exp_dir, model, data, model_fc, args, out_file=out_file)

        imp.to_csv(out_file, index=False)
        print(f"Saved: {out_file}")
        save_top_print(imp, model, n=20)
        print(f"Model done in {time.perf_counter() - t0:.1f}s")
        outputs.append(imp)

    if outputs:
        combined = build_combined(out_dir, outputs)
        print(f"\nSaved combined: {out_dir / 'factor_importance_combined.csv'}")
        print(combined.head(20).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
