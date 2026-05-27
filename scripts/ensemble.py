"""
Final ensemble evaluation — all results on consistent 366-day inner-join口径.

Sections:
  1. Single-model baselines (valid + test)
  2. Equal-Weight (valid + test)
  3. Rank-Ridge fitted on valid → evaluated on valid + test
  4. Full grid on valid → top 5 on test
  5. LGBM alone on test (reference baseline, inner-join universe)

Output: temp/ensemble_results.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest
from src.backtest.portfolio import PortfolioConfig
from scripts.run_experiment import (
    ExperimentConfig, _load_experiment_raw_data, _split_clean_df_7_1_2,
    build_returns_frame_from_next_target, build_experiment_model, _get_model_params,
)
from src.data.preprocess import PreprocessConfig, preprocess_panel_data
from src.data.dataset_builder import PanelDatasetBuilder
from src.predict.generate_predictions import PredictionConfig, generate_predictions
from src.utils.seed import set_seed

OUT = Path("dataset/output/experiment_003")
MODELS = ["lightgbm", "xgboost", "dlinear", "itransformer", "tsmixer"]
COMBO_WEIGHTS = [
    (0.30, 0.30, 0.20, 0.15, 0.05),
    (0.30, 0.30, 0.20, 0.10, 0.10),
    (0.35, 0.35, 0.20, 0.05, 0.05),
    (0.30, 0.30, 0.20, 0.20, 0.00),
]


def load_test():
    """Load saved test predictions, inner join."""
    preds = {}
    for m in MODELS:
        df = pd.read_parquet(OUT / f"predictions_{m}.parquet")
        df["time"] = pd.to_datetime(df["time"]).dt.normalize()
        df["stock_id"] = df["stock_id"].astype(str).str.strip()
        preds[m] = df
    merged = preds["lightgbm"][["time", "stock_id", "y_true"]].reset_index(drop=True).copy()
    merged["lightgbm"] = preds["lightgbm"]["y_pred"].reset_index(drop=True)
    for m in ["xgboost", "dlinear", "itransformer", "tsmixer"]:
        merged = merged.merge(preds[m][["time", "stock_id", "y_pred"]].rename(
            columns={"y_pred": m}), on=["time", "stock_id"], how="inner")
    return merged


def build_returns():
    config = ExperimentConfig()
    raw = _load_experiment_raw_data(config.data_path)
    if config.date_col not in raw.columns and config.date_col in raw.index.names:
        raw = raw.reset_index()
    clean = preprocess_panel_data(raw, PreprocessConfig(
        date_col=config.date_col, stock_col=config.stock_col,
        feature_cols=list(config.feature_cols), target_col=config.target_col,
        meta_cols=list(config.meta_cols),
        replace_inf_with_nan=True, drop_rows_with_missing_keys=True,
        drop_rows_with_missing_target=True, duplicate_policy="raise", sort_values=True,
    )).df
    ret = clean[[config.date_col, config.stock_col, "ret_daily"]].rename(
        columns={"ret_daily": config.return_col}).copy()
    ret[config.date_col] = pd.to_datetime(ret[config.date_col]).dt.normalize()
    ret[config.stock_col] = ret[config.stock_col].astype(str).str.strip()
    ret[config.return_col] = ret[config.return_col].astype(float)
    ret["time"] = pd.to_datetime(ret["time"]).dt.normalize()
    ret["stock_id"] = ret["stock_id"].astype(str).str.strip()
    return ret


def generate_valid():
    config = ExperimentConfig()
    raw = _load_experiment_raw_data(config.data_path)
    if config.date_col not in raw.columns and config.date_col in raw.index.names:
        raw = raw.reset_index()
    clean = preprocess_panel_data(raw, PreprocessConfig(
        date_col=config.date_col, stock_col=config.stock_col,
        feature_cols=list(config.feature_cols), target_col=config.target_col,
        meta_cols=list(config.meta_cols),
        replace_inf_with_nan=True, drop_rows_with_missing_keys=True,
        drop_rows_with_missing_target=True, duplicate_policy="raise", sort_values=True,
    )).df
    _, valid_df, _, _, _ = _split_clean_df_7_1_2(clean, config.date_col, config.stock_col)

    preds = {}
    for m in MODELS:
        p = _get_model_params(m, config)
        model = build_experiment_model(m, seed=config.seed, seq_len=p["seq_len"],
                                       n_features=len(config.feature_cols))
        if m in ("lightgbm", "xgboost"):
            builder = PanelDatasetBuilder(
                feature_cols=list(config.feature_cols), target_col=config.target_col,
                date_col=config.date_col, stock_col=config.stock_col, seq_len=1,
                meta_cols=list(config.meta_cols))
            data = builder.build_tabular_dataset(valid_df)
            if m == "lightgbm":
                import lightgbm as lgb
                model.model = lgb.Booster(model_file=str(OUT / "models" / "LightGBMReturnRegressor.txt"))
                model._is_fitted = True
            else:
                model.model.load_model(str(OUT / "models" / "XGBoostReturnRegressor.json"))
                model._is_fitted = True
            pdf = generate_predictions(model=model, dataset=data, model_name=m,
                                       required_meta_cols=(config.date_col, config.stock_col))
        else:
            builder = PanelDatasetBuilder(
                feature_cols=list(config.feature_cols), target_col=config.target_col,
                date_col=config.date_col, stock_col=config.stock_col, seq_len=p["seq_len"],
                meta_cols=list(config.meta_cols))
            data = builder.build_sequence_dataset(valid_df)
            ckpt = OUT / "models" / f"{model.__class__.__name__}_best_icir.pt"
            if not ckpt.exists():
                ckpt = OUT / "models" / f"{model.__class__.__name__}.pt"
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            model.eval()
            pdf = generate_predictions(model=model, dataset=data, model_name=m,
                                       config=PredictionConfig(batch_size=8192, device="cpu"),
                                       required_meta_cols=(config.date_col, config.stock_col))
        pdf["time"] = pd.to_datetime(pdf["time"]).dt.normalize()
        pdf["stock_id"] = pdf["stock_id"].astype(str).str.strip()
        preds[m] = pdf

    # Inner join all 4 on valid set
    merged = preds["lightgbm"][["time", "stock_id", "y_true"]].reset_index(drop=True).copy()
    merged["lightgbm"] = preds["lightgbm"]["y_pred"].reset_index(drop=True)
    for m in ["xgboost", "dlinear", "itransformer", "tsmixer"]:
        merged = merged.merge(preds[m][["time", "stock_id", "y_pred"]].rename(
            columns={"y_pred": m}), on=["time", "stock_id"], how="inner")
    return merged


def bt(df, returns, yp_col="_yp"):
    d = df[["time", "stock_id", yp_col]].dropna(subset=[yp_col]).rename(columns={yp_col: "y_pred"})
    # Build (time, stock_id) set from returns — only keep predictions with valid returns
    ret_pairs = set(zip(returns["time"], returns["stock_id"]))
    pred_dates = pd.DatetimeIndex(d["time"].dropna().unique()).sort_values()
    ret_dates = pd.DatetimeIndex(returns["time"].dropna().unique()).sort_values()
    # Find next return date for each signal date
    next_date_map = {}
    for sd in pred_dates:
        future = ret_dates[ret_dates > sd]
        if len(future): next_date_map[sd] = future[0]
    # Keep only rows where the next return date has the same stock
    keep = pd.Series(False, index=d.index)
    for sd, nd in next_date_map.items():
        stocks_on_nd = returns.loc[returns["time"] == nd, "stock_id"].unique()
        mask = (d["time"] == sd) & (d["stock_id"].isin(stocks_on_nd))
        keep[mask] = True
    d = d[keep].copy()
    pfolio = PortfolioConfig(strategy="top_n", top_n=50, pred_col="y_pred", stock_col="stock_id")
    return run_backtest(pred_df=d, returns_df=returns, portfolio_config=pfolio,
                        return_col="return_1d", date_col="time", stock_col="stock_id")["summary"]


def main():
    set_seed(42)
    print("=== Loading test predictions ===")
    test = load_test()
    print(f"  {len(test):,} rows, {test['time'].nunique()} dates")

    print("\n=== Generating valid predictions ===")
    valid = generate_valid()
    print(f"  {len(valid):,} rows, {valid['time'].nunique()} dates")

    returns = build_returns()

    rows = []

    # 1. Single-model baselines
    for m in MODELS:
        rv = bt(valid.assign(_yp=valid[m]), returns)
        rt = bt(test.assign(_yp=test[m]), returns)
        rows.append({"section": "single", "model": m, "note": "valid", "sharpe": rv["sharpe_ratio"],
                     "annret": rv["annualized_return"], "maxdd": rv["max_drawdown"],
                     "nav": rv["final_nav"], "turnover": rv["mean_turnover"], "winrate": rv["hit_rate"]})
        rows.append({"section": "single", "model": m, "note": "test", "sharpe": rt["sharpe_ratio"],
                     "annret": rt["annualized_return"], "maxdd": rt["max_drawdown"],
                     "nav": rt["final_nav"], "turnover": rt["mean_turnover"], "winrate": rt["hit_rate"]})
        print(f"  {m:15s}: valid_Sharpe={rv['sharpe_ratio']:.4f}  test_Sharpe={rt['sharpe_ratio']:.4f}")

    # 2. Equal-Weight
    valid["_yp"] = valid[MODELS].mean(axis=1)
    test["_yp"] = test[MODELS].mean(axis=1)
    rv = bt(valid, returns)
    rt = bt(test, returns)
    for note, r in [("valid", rv), ("test", rt)]:
        rows.append({"section": "equal_weight", "model": "ensemble", "note": note,
                     "sharpe": r["sharpe_ratio"], "annret": r["annualized_return"],
                     "maxdd": r["max_drawdown"], "nav": r["final_nav"],
                     "turnover": r["mean_turnover"], "winrate": r["hit_rate"]})
    print(f"  Equal-Weight     : valid_Sharpe={rv['sharpe_ratio']:.4f}  test_Sharpe={rt['sharpe_ratio']:.4f}")

    # 3. Rank-Ridge fitted on valid
    for m in MODELS:
        valid[f"{m}_r"] = valid.groupby("time")[m].rank(pct=True)
    valid["yt_r"] = valid.groupby("time")["y_true"].rank(pct=True)
    rank_cols = [f"{m}_r" for m in MODELS]
    ridge = RidgeCV(alphas=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0], fit_intercept=False)
    ridge.fit(valid[rank_cols].values, valid["yt_r"].values)
    w_rr = pd.Series(ridge.coef_, index=MODELS)
    w_rr = w_rr / w_rr.abs().sum()

    valid["_yp"] = sum(w_rr[m] * valid[m] for m in MODELS)
    test["_yp"] = sum(w_rr[m] * test[m] for m in MODELS)
    rv = bt(valid, returns)
    rt = bt(test, returns)
    for note, r in [("valid", rv), ("test", rt)]:
        rows.append({"section": "rank_ridge", "model": "ensemble", "note": note,
                     "sharpe": r["sharpe_ratio"], "annret": r["annualized_return"],
                     "maxdd": r["max_drawdown"], "nav": r["final_nav"],
                     "turnover": r["mean_turnover"], "winrate": r["hit_rate"]})
    print(f"  Rank-Ridge       : alpha={ridge.alpha_:.1f}  weights={dict(w_rr.round(4))}")
    print(f"                    valid_Sharpe={rv['sharpe_ratio']:.4f}  test_Sharpe={rt['sharpe_ratio']:.4f}")

    # 4. Grid on valid → top 5 on test
    print(f"\n=== Grid on VALID ({len(valid):,} rows, {valid['time'].nunique()} dates) ===")
    grid = []
    for w_lgbm, w_xgb, w_dlin, w_it, w_tsm in COMBO_WEIGHTS:
        valid["_yp"] = (w_lgbm * valid["lightgbm"] + w_xgb * valid["xgboost"]
                        + w_dlin * valid["dlinear"] + w_it * valid["itransformer"]
                        + w_tsm * valid["tsmixer"])
        s = bt(valid, returns)
        grid.append((w_lgbm, w_xgb, w_dlin, w_it, w_tsm, s["sharpe_ratio"]))
    grid.sort(key=lambda x: x[5], reverse=True)

    print("All grid → both valid & test:")
    for w_lgbm, w_xgb, w_dlin, w_it, w_tsm, sv in grid:
        test["_yp"] = (w_lgbm * test["lightgbm"] + w_xgb * test["xgboost"]
                       + w_dlin * test["dlinear"] + w_it * test["itransformer"]
                       + w_tsm * test["tsmixer"])
        st = bt(test, returns)
        note = f"LGBM={w_lgbm:.2f}_XGB={w_xgb:.2f}_DL={w_dlin:.2f}_iT={w_it:.2f}_TSM={w_tsm:.2f}"
        rows.append({"section": "grid", "model": "ensemble", "note": note,
                     "valid_sharpe": sv, "sharpe": st["sharpe_ratio"],
                     "annret": st["annualized_return"], "maxdd": st["max_drawdown"],
                     "nav": st["final_nav"], "turnover": st["mean_turnover"],
                     "winrate": st["hit_rate"]})

    # Sort and print top 5 for quick view
    grid_rows = [r for r in rows if r["section"] == "grid"]
    grid_rows.sort(key=lambda x: x["valid_sharpe"], reverse=True)
    print("Top 5 by valid Sharpe:")
    for r in grid_rows[:5]:
        print(f"  {r['note']:40s}  valid={r['valid_sharpe']:.4f}  test={r['sharpe']:.4f}  MaxDD={r['maxdd']:.4f}")
    grid_rows.sort(key=lambda x: x["sharpe"], reverse=True)
    print("Top 5 by test Sharpe:")
    for r in grid_rows[:5]:
        print(f"  {r['note']:40s}  test={r['sharpe']:.4f}  valid={r['valid_sharpe']:.4f}  MaxDD={r['maxdd']:.4f}")

    pd.DataFrame(rows).to_csv("temp/ensemble_results.csv", index=False)
    print(f"\nSaved: temp/ensemble_results.csv")


if __name__ == "__main__":
    main()
