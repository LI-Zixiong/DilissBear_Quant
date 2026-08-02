"""Generate live predictions from frozen checkpoints for the daily pipeline.

Public API
----------
``generate_live_predictions(exp_dir, data_path, smooth_window, track_start)``
    Detect missing prediction dates from *track_start* onwards and fill them
    using the saved LightGBM/DLinear/GDCN checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from src.data.dataset_builder import PanelDatasetBuilder
from src.experiment.config import ExperimentConfig
from src.experiment.data import preprocess_experiment_data
from src.experiment.runner import (
    _apply_feature_signs,
    _tabular_predict_at_n,
)
from src.predict.generate_predictions import (
    PredictionConfig,
    generate_predictions,
    save_predictions,
)

# ── Per-model feature sets (frozen, matched to experiment_009 checkpoints) ──
MODEL_FEATURE_COLS: dict[str, tuple[str, ...]] = {
    "dlinear": (
        "F025GAP", "F030RSV20", "F028KLOW", "F013REV5", "F035LOWDEV20",
        "F058O2O_RET5", "F041RETVOLCORR20", "F075VOLUME_RATIO", "F031RSV60", "F015VOLREV",
        "F081STRONG_CLOSE", "F001SIZE", "F029KSFT", "F087LIST_AGE", "F027KUP",
        "F016MAXRET", "F002SIZENL", "F066EFFICIENCY20", "F056GAP_UP_FAIL", "F076SIGNED_AMT20",
        "F038TURNZ20", "F050ROA", "F007LTREV", "F068SKEW20", "F043SLOPE20",
        "F053RD_INTENSITY", "F045RESI20", "F012GROWTH", "F049ROE", "F010VALUE",
        "F051GPM", "F011EARNYLD", "F079TURN_ACCEL", "F088CF_SALES_Q", "F018AMIHUD",
        "F023ACCRUAL", "F072CORR_60D", "F006MOMENTUM", "F044RSQR20", "F060ON_INTRA_DIV5",
        "F026KLEN", "F064RET_ACCEL20", "F085SP_TTM", "F089CASH_PROFIT", "F073KURT_60D",
        "F061GAP_UP_HOLD", "F022GPTA", "F059GK_VOL20", "F098DIV_PAYOUT", "F097INT_BURDEN",
        "F082LOCKED_PCT", "F095NET_FIN", "F077AMP_VOL20", "F100INV_MINUS_REV", "F078TURN_SIZE",
        "F094CAPEX_INT", "F032RANGEZ20", "F074BETA_20D", "F021CFP", "F063RET5D_SKIP1",
        "F017IVOL", "F042AMTCORR20", "F003LIQUIDITY", "F008STREV", "F096DILUTION",
        "F080VWAP_DEV", "F084AMT_FREE20", "F019COSTDEV", "F071VOL_OF_VOL", "F054RECEIVABLE_RATIO",
        "F092EARN_STAB", "F037VOLSHOCK20", "F014MOM120_20", "F093FCF_YIELD", "F036VOLSHOCK5",
        "F005RESVOL", "F009LEVERAGE",
    ),
    "lightgbm": (
        "F001SIZE", "F002SIZENL", "F003LIQUIDITY", "F004BETA", "F005RESVOL",
        "F006MOMENTUM", "F007LTREV", "F008STREV", "F009LEVERAGE", "F010VALUE",
        "F011EARNYLD", "F012GROWTH", "F013REV5", "F014MOM120_20", "F015VOLREV",
        "F016MAXRET", "F017IVOL", "F018AMIHUD", "F019COSTDEV", "F021CFP",
        "F022GPTA", "F023ACCRUAL", "F024ASSETGR", "F025GAP", "F026KLEN",
        "F027KUP", "F028KLOW", "F029KSFT", "F030RSV20", "F031RSV60",
        "F032RANGEZ20", "F033GAPREV5", "F034HIGHDEV20", "F035LOWDEV20", "F036VOLSHOCK5",
        "F037VOLSHOCK20", "F038TURNZ20", "F039VSTD20", "F040PVCORR20", "F041RETVOLCORR20",
        "F042AMTCORR20", "F043SLOPE20", "F044RSQR20", "F045RESI20",
        "F049ROE", "F050ROA", "F051GPM", "F052CFOA", "F053RD_INTENSITY", "F054RECEIVABLE_RATIO",
        "F056GAP_UP_FAIL", "F057INTRA1", "F058O2O_RET5", "F059GK_VOL20", "F060ON_INTRA_DIV5",
        "F061GAP_UP_HOLD", "F062GAP_DN_RECOVER", "F063RET5D_SKIP1", "F064RET_ACCEL20",
        "F065MAXDD20", "F066EFFICIENCY20", "F067TAIL_LOSS20", "F068SKEW20", "F069DNVOL20",
        "F070UP_DN_VOL", "F071VOL_OF_VOL", "F072CORR_60D", "F073KURT_60D", "F074BETA_20D",
        "F075VOLUME_RATIO", "F076SIGNED_AMT20", "F077AMP_VOL20", "F078TURN_SIZE",
        "F079TURN_ACCEL", "F080VWAP_DEV", "F081STRONG_CLOSE", "F082LOCKED_PCT",
        "F083TURN_FREE", "F084AMT_FREE20", "F085SP_TTM", "F086DIV_TTM", "F087LIST_AGE",
        "F088CF_SALES_Q", "F089CASH_PROFIT", "F090CRR", "F091CF_VOL", "F092EARN_STAB",
        "F093FCF_YIELD", "F094CAPEX_INT", "F095NET_FIN", "F096DILUTION", "F097INT_BURDEN",
        "F098DIV_PAYOUT", "F099AR_MINUS_REV", "F100INV_MINUS_REV",
    ),
}

FULL_96F = (
    "F001SIZE", "F002SIZENL", "F003LIQUIDITY", "F004BETA", "F005RESVOL",
    "F006MOMENTUM", "F007LTREV", "F008STREV", "F009LEVERAGE", "F010VALUE",
    "F011EARNYLD", "F012GROWTH", "F013REV5", "F014MOM120_20", "F015VOLREV",
    "F016MAXRET", "F017IVOL", "F018AMIHUD", "F019COSTDEV", "F021CFP",
    "F022GPTA", "F023ACCRUAL", "F024ASSETGR", "F025GAP", "F026KLEN", "F027KUP",
    "F028KLOW", "F029KSFT", "F030RSV20", "F031RSV60", "F032RANGEZ20",
    "F033GAPREV5", "F034HIGHDEV20", "F035LOWDEV20", "F036VOLSHOCK5",
    "F037VOLSHOCK20", "F038TURNZ20", "F039VSTD20", "F040PVCORR20",
    "F041RETVOLCORR20", "F042AMTCORR20", "F043SLOPE20", "F044RSQR20",
    "F045RESI20", "F049ROE", "F050ROA", "F051GPM", "F052CFOA",
    "F053RD_INTENSITY", "F054RECEIVABLE_RATIO", "F055IND",
    "F056GAP_UP_FAIL", "F057INTRA1", "F058O2O_RET5", "F059GK_VOL20",
    "F060ON_INTRA_DIV5", "F061GAP_UP_HOLD", "F062GAP_DN_RECOVER",
    "F063RET5D_SKIP1", "F064RET_ACCEL20", "F065MAXDD20", "F066EFFICIENCY20",
    "F067TAIL_LOSS20", "F068SKEW20", "F069DNVOL20", "F070UP_DN_VOL",
    "F071VOL_OF_VOL", "F072CORR_60D", "F073KURT_60D", "F074BETA_20D",
    "F075VOLUME_RATIO", "F076SIGNED_AMT20", "F077AMP_VOL20", "F078TURN_SIZE",
    "F079TURN_ACCEL", "F080VWAP_DEV", "F081STRONG_CLOSE", "F082LOCKED_PCT",
    "F083TURN_FREE", "F084AMT_FREE20", "F085SP_TTM", "F086DIV_TTM",
    "F087LIST_AGE", "F088CF_SALES_Q", "F089CASH_PROFIT", "F090CRR",
    "F091CF_VOL", "F092EARN_STAB", "F093FCF_YIELD", "F094CAPEX_INT",
    "F095NET_FIN", "F096DILUTION", "F097INT_BURDEN", "F098DIV_PAYOUT",
    "F099AR_MINUS_REV", "F100INV_MINUS_REV",
)

MODEL_TYPES = {"lightgbm": "tabular", "dlinear": "torch", "gated_dwtcn": "torch"}
PRED_CFG = PredictionConfig(batch_size=8192, device="cpu")


def _saved_feature_list(exp_dir: Path, model_name: str) -> tuple[str, ...] | None:
    candidates = [
        exp_dir / f"model_features_{model_name}.json",
        exp_dir / "models" / model_name / f"model_features_{model_name}.json",
        exp_dir / "models" / model_name / "model_features.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        values: list | None = None
        if isinstance(payload, list):
            values = payload
        elif isinstance(payload, dict):
            values = payload.get("features") or payload.get("feature_cols") or payload.get(model_name)
        if values:
            print(f"  Feature manifest: {path} ({len(values)} columns)")
            return tuple(values)
    return None


def generate_live_predictions(
    exp_dir: str | Path,
    data_path: str | Path | None = None,
    smooth_window: int = 10,
    track_start: pd.Timestamp = pd.Timestamp("2026-07-20"),
    config: ExperimentConfig | None = None,
    _live_factor_df: pd.DataFrame | None = None,
) -> None:
    """Fill missing live predictions from *track_start* onwards.

    Parameters
    ----------
    exp_dir:
        Experiment output directory (contains checkpoints and saved manifests).
    data_path:
        Factor panel path.  Defaults to ``ExperimentConfig().data_path``.
        Ignored when *_live_factor_df* is provided.
    smooth_window:
        Number of warm-up prediction dates required before *track_start*.
    track_start:
        Public tracking inception date.
    config:
        Experiment config.  Created from defaults if not provided.
    _live_factor_df:
        Pre-loaded factor DataFrame from the live store.  When provided,
        skips the disk read and uses this directly (zero-copy fast path).
    """
    if config is None:
        config = ExperimentConfig()
    config.model_feature_cols = dict(MODEL_FEATURE_COLS)
    exp_dir = Path(exp_dir)

    if _live_factor_df is not None:
        full_raw = _live_factor_df.copy()
    else:
        fp = Path(data_path) if data_path is not None else Path(config.data_path)
        full_raw = pd.read_parquet(fp)
    full_raw["time"] = pd.to_datetime(full_raw["time"])
    panel_dates = sorted(pd.Timestamp(d) for d in full_raw["time"].dropna().unique())
    start_idx = next((i for i, d in enumerate(panel_dates) if d >= track_start), len(panel_dates))
    warmup_start = panel_dates[max(0, start_idx - (smooth_window - 1))] if panel_dates else track_start
    raw = full_raw[full_raw["time"] >= warmup_start].copy()

    all_dates = sorted(raw["time"].unique())
    print(f"Factor panel: {len(all_dates)} dates from {all_dates[0].date()} to {all_dates[-1].date()} "
          f"(tracking starts {track_start.date()})")

    missing_dates: dict[str, list[pd.Timestamp]] = {}
    for m in MODEL_TYPES:
        p = exp_dir / f"predictions_live_{m}.parquet"
        existing = set()
        if p.exists():
            existing = set(pd.read_parquet(p, columns=["time"])["time"])
        needed = set(all_dates) - existing
        missing_dates[m] = sorted(needed)
        print(f"{m:>14s}: {len(existing)} exist, {len(needed)} missing")

    if all(len(v) == 0 for v in missing_dates.values()):
        print("\nAll predictions up to date. Nothing to do.")
        return

    print("\nPreprocessing factor panel...")
    active_factors = [c for c in config.feature_cols if c in raw.columns]
    inference_df, _ = preprocess_experiment_data(raw, config, active_factors, drop_missing_target=False)
    torch_history: pd.DataFrame | None = None
    if any(MODEL_TYPES[m] == "torch" for m in missing_dates):
        torch_history, _ = preprocess_experiment_data(
            full_raw, config, active_factors, drop_missing_target=False,
        )
        torch_history = torch_history.sort_values(
            [config.stock_col, config.date_col],
        ).reset_index(drop=True)

    _ind_to_id: dict[int, int] = {}
    _n_industries = 0
    if "industry_sw" in full_raw.columns:
        all_ind = full_raw["industry_sw"].dropna().astype(int)
        unique_ind = sorted(all_ind.unique())
        _ind_to_id = {v: i + 1 for i, v in enumerate(unique_ind)}
        _n_industries = len(unique_ind) + 1
        print(f"Industry mapping: {_n_industries} categories (from {len(full_raw):,} rows)")

    # ── Predict each model ──
    for model_name, model_type in MODEL_TYPES.items():
        dates_needed = missing_dates[model_name]
        if not dates_needed:
            continue

        saved_cols = _saved_feature_list(exp_dir, model_name)
        if saved_cols is not None:
            cols = list(saved_cols)
        elif model_name == "gated_dwtcn":
            cols = [c for c in FULL_96F if c in raw.columns]
        else:
            from src.experiment.runner import _model_features
            cols = list(_model_features(model_name, config))

        missing_cols = [c for c in cols if c not in full_raw.columns and c != "industry_id"]
        if missing_cols:
            raise ValueError(f"{model_name} feature manifest columns missing from factor panel: {missing_cols[:10]}")
        print(f"\n{'='*60}")
        print(f"{model_name}: {len(dates_needed)} dates, {len(cols)} factors")
        print(f"{'='*60}")

        # Build model from checkpoint
        if model_type == "tabular":
            model_path = exp_dir / "models" / model_name / "LightGBMReturnRegressor.txt"
            scan_path = exp_dir / f"n_scan_{model_name}.csv"
            if not model_path.exists() or not scan_path.exists():
                raise FileNotFoundError(f"Required LightGBM model/scan missing: {model_path}, {scan_path}")
            from src.experiment.model_factory import build_experiment_model
            import src.utils.seed as seed_utils
            seed_utils.set_seed(config.seed)
            model = build_experiment_model(model_name, seed=config.seed, seq_len=1, n_features=len(cols), config=config)
            model.model = lgb.Booster(model_file=str(model_path))
            model._n_features = len(cols)
            scan = pd.read_csv(scan_path)
            best_n = int(scan.loc[scan["valid_total_return"].idxmax(), "n_tree"])
            print(f"  Loaded booster, best_n={best_n}")
        else:
            model_dir = exp_dir / "models" / model_name
            preferred = (
                ["DLinearPanelRegressor_best_composite.pt"] if model_name == "dlinear"
                else ["GatedDWTcnRegressor_best_composite.pt", "GatedDWTCNPanelRegressor_best_composite.pt"]
            )
            ckpt_path = next((model_dir / name for name in preferred if (model_dir / name).exists()), None)
            if ckpt_path is None:
                matches = sorted(model_dir.glob("*_best_composite.pt"))
                ckpt_path = matches[0] if len(matches) == 1 else None
            if ckpt_path is None:
                raise FileNotFoundError(f"No unique best_composite checkpoint found in {model_dir}")
            if model_name == "dlinear":
                from src.models.dlinear import DLinearConfig, build_dlinear_model
                model = build_dlinear_model(DLinearConfig(
                    n_features=len(cols), seq_len=20, n_industries=_n_industries, ind_rank=8,
                ))
            elif model_name == "gated_dwtcn":
                from src.models.gated_dwtcn import GatedDWTcnConfig, build_gated_dwtcn
                model = build_gated_dwtcn(GatedDWTcnConfig(
                    n_features=len(cols), seq_len=20, kernel_size=3, dilations=(1, 2, 4),
                    hidden_dim=32, gate_rank=20, dropout=0.0,
                ))
            model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu", weights_only=True))
            model.eval()
            print(f"  Loaded checkpoint: {ckpt_path.name}")

        # Predict each missing date
        all_preds = []
        for d in dates_needed:
            day_df = inference_df[inference_df["time"] == d].copy()
            _apply_feature_signs(day_df, model_name, config)

            day_cols = list(cols)
            if "industry_sw" in day_df.columns and _ind_to_id:
                day_df["industry_id"] = day_df["industry_sw"].fillna(-1).astype(int).map(_ind_to_id).fillna(0).astype("int64")
                if model_type == "tabular" and "industry_id" not in day_cols:
                    day_cols.append("industry_id")

            if config.one_hot_features and model_name in config.one_hot_features:
                for oh_col in config.one_hot_features[model_name]:
                    if oh_col in day_df.columns:
                        cats = sorted(day_df[oh_col].dropna().unique())
                        for c in cats:
                            day_df[f"{oh_col}_{int(c)}"] = (day_df[oh_col] == c).astype("uint8")
                        day_df.drop(columns=[oh_col], inplace=True)

            if model_type == "tabular":
                meta_cols = list(config.meta_cols)
                available_meta = [c for c in meta_cols if c in day_df.columns]
                builder = PanelDatasetBuilder(
                    feature_cols=day_cols, target_col=config.target_col,
                    date_col=config.date_col, stock_col=config.stock_col,
                    seq_len=1, meta_cols=available_meta,
                )
                data = builder.build_tabular_dataset(day_df, require_target=False)
                pred_df = _tabular_predict_at_n(model, data, best_n, config.date_col, config.stock_col)
            else:
                seq_len = 20
                hist_start = d - pd.Timedelta(days=60)
                if torch_history is None:
                    raise RuntimeError("torch_history was not built for torch model")
                hist_df = torch_history[
                    (torch_history["time"] >= hist_start)
                    & (torch_history["time"] <= d)
                ].copy()
                hist_df["_is_target"] = hist_df["time"] == d
                hist_df = hist_df.groupby(config.stock_col, sort=False).tail(seq_len).copy()
                hist_df["_live_endpoint"] = hist_df["_is_target"]

                _apply_feature_signs(hist_df, model_name, config)
                if _ind_to_id and "industry_sw" in hist_df.columns:
                    hist_df["industry_id"] = hist_df["industry_sw"].fillna(-1).astype(int).map(_ind_to_id).fillna(0).astype("int64")

                meta_cols = list(config.meta_cols)
                available_meta = [c for c in meta_cols if c in hist_df.columns]
                builder = PanelDatasetBuilder(
                    feature_cols=cols, target_col=config.target_col,
                    date_col=config.date_col, stock_col=config.stock_col,
                    seq_len=seq_len, meta_cols=available_meta,
                )
                live_data = builder.build_sequence_dataset(
                    hist_df, end_filter_col="_live_endpoint", end_filter_value=True, require_target=False,
                )
                pred_df = generate_predictions(
                    model, live_data, model_name, PRED_CFG,
                    required_meta_cols=(config.date_col, config.stock_col),
                )

            all_preds.append(pred_df)

        if all_preds:
            combined = pd.concat(all_preds, ignore_index=True)
            out_path = exp_dir / f"predictions_live_{model_name}.parquet"
            if out_path.exists():
                previous = pd.read_parquet(out_path)
                previous["time"] = pd.to_datetime(previous["time"])
                combined = pd.concat([previous, combined], ignore_index=True)
            combined = combined.sort_values(["time", "stock_id"]).drop_duplicates(
                ["time", "stock_id"], keep="last",
            ).reset_index(drop=True)
            tmp_path = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
            save_predictions(combined, output_path=tmp_path)
            tmp_path.replace(out_path)
            print(f"  Saved: {out_path} ({len(combined)} rows, {combined['time'].nunique()} dates)")

    print("\nDone. Run convert_data.py to refresh web data.")
