"""
Run a multi-model end-to-end experiment.

Phase 2B goal:
raw data -> preprocess -> dataset -> train -> predict -> backtest -> compare models
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.backtest.engine import TransactionCostConfig, run_backtest
from src.backtest.portfolio import PortfolioConfig
from src.data.dataset_builder import PanelDatasetBuilder
from src.data.loader import load_panel_data
from src.data.preprocess import PreprocessConfig, preprocess_panel_data
from src.models.dlinear import DLinearConfig, build_dlinear_model
from src.models.itransformer import ITransformerConfig, build_itransformer_model
from src.models.lightgbm_model import LightGBMConfig, build_lightgbm_model
from src.models.patchtst import PatchTSTConfig, build_patchtst_model
from src.models.tsmixer import TSMixerConfig, build_tsmixer_model
from src.models.xgboost_model import XGBoostConfig, build_xgboost_model
from src.predict.generate_predictions import (
    PredictionConfig,
    generate_predictions,
    save_predictions,
)
from src.train.train_tabular import train_tabular_model
from src.train.train_torch import TorchTrainConfig, train_torch_model
from src.utils.seed import set_seed


DEFAULT_FEATURE_COLS = (
    "SIZE",
    "SIZENL",
    "LIQUIDITY",
    "BETA",
    "RESVOL",
    "MOMENTUM",
    "LEVERAGE",
    "VALUE",
    "EARNYLD",
    "GROWTH",
    "LTREV",
    "STREV",
)


@dataclass
class ExperimentConfig:
    data_path: str = "dataset/processed/factor_panel_1500_ind.parquet"

    date_col: str = "time"
    stock_col: str = "stock_id"
    target_col: str = "5d_next_raw"
    return_col: str = "return_1d"
    backtest_return_source: str = "1d_next_raw"

    feature_cols: Sequence[str] = (
        "SIZE", "SIZENL", "LIQUIDITY", "BETA", "RESVOL",
        "MOMENTUM", "LTREV", "STREV", "LEVERAGE", "VALUE",
        "EARNYLD", "GROWTH",
    )
    meta_cols: Sequence[str] = ("industry_csrc_2012", "list_date")

    model_names: Sequence[str] = (
        "lightgbm", "dlinear",
    )

    train_end: str = ""
    valid_end: str = ""

    seed: int = 42
    top_n: int = 50
    periods_per_year: int = 252

    seq_len: int = 20
    torch_epochs: int = 10
    torch_patience: int = 1
    torch_batch_size: int = 4096
    torch_learning_rate: float = 7.5e-4
    torch_weight_decay: float = 0.0
    torch_device: str = "auto"

    model_params: dict = dataclasses.field(default_factory=lambda: {
        "dlinear":      {"seq_len": 15, "epochs": 15, "lr": 7.5e-4,   "wd": 0.0, "patience": 2},
        "itransformer": {"seq_len": 40, "epochs": 10, "lr": 1e-4,   "wd": 0.0, "patience": 2},
        "tsmixer":      {"seq_len": 40, "epochs": 15, "lr": 4e-3,   "wd": 0.0, "patience": 2},
    })
    predict_batch_size: int = 8192

    output_dir: str = "dataset/output/experiment_003"
    report_path: str = "reports/experiment_003.md"


def _split_clean_df_7_1_2(
    df: pd.DataFrame,
    date_col: str,
    stock_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """
    Split a cleaned panel DataFrame into strict 7/1/2 chronological date slices.

    The split is based on unique dates, not row counts, to avoid splitting the
    same trading day across train/valid/test.
    """
    required_cols = [date_col, stock_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"df missing required columns: {missing_cols}")

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="raise")
    work = work.sort_values([date_col, stock_col], kind="mergesort").reset_index(
        drop=True
    )

    unique_dates = pd.DatetimeIndex(work[date_col].drop_duplicates()).sort_values()
    n_dates = len(unique_dates)

    if n_dates < 3:
        raise ValueError(f"Need at least 3 unique dates for 7/1/2 split, got {n_dates}")

    train_end_idx = int(n_dates * 0.7)
    valid_end_idx = int(n_dates * 0.8)

    if train_end_idx <= 0 or valid_end_idx <= train_end_idx or valid_end_idx >= n_dates:
        raise ValueError(
            "Unable to derive a non-empty 7/1/2 date split. "
            f"n_dates={n_dates}, train_end_idx={train_end_idx}, "
            f"valid_end_idx={valid_end_idx}"
        )

    train_dates = unique_dates[:train_end_idx]
    valid_dates = unique_dates[train_end_idx:valid_end_idx]
    test_dates = unique_dates[valid_end_idx:]

    train_df = work[work[date_col].isin(train_dates)].copy().reset_index(drop=True)
    valid_df = work[work[date_col].isin(valid_dates)].copy().reset_index(drop=True)
    test_df = work[work[date_col].isin(test_dates)].copy().reset_index(drop=True)

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError(
            "Strict 7/1/2 date split produced an empty subset. "
            f"train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}"
        )

    train_end = str(pd.Timestamp(train_dates[-1]))
    valid_end = str(pd.Timestamp(valid_dates[-1]))

    return train_df, valid_df, test_df, train_end, valid_end


def _load_experiment_raw_data(data_path: str) -> pd.DataFrame:
    """
    Load the experiment input data.

    If the aggregate parquet is unreadable, fall back to split parquet files in
    the same directory.
    """
    try:
        return load_panel_data(data_path)
    except OSError as exc:
        error_message = str(exc)

        if "Couldn't deserialize thrift" not in error_message:
            raise

        data_file = Path(data_path)
        required_fallback_paths = [
            data_file.with_name("df_response_daily_train.parquet"),
            data_file.with_name("df_response_daily_validate.parquet"),
        ]
        optional_fallback_paths = [
            data_file.with_name("df_response_daily_test.parquet"),
        ]

        missing_paths = [path for path in required_fallback_paths if not path.exists()]
        if missing_paths:
            missing_text = ", ".join(str(path) for path in missing_paths)
            raise ValueError(
                "Failed to read aggregate parquet and fallback files are missing: "
                f"{missing_text}"
            ) from exc

        fallback_paths = required_fallback_paths + [
            path for path in optional_fallback_paths if path.exists()
        ]

        frames = [load_panel_data(path) for path in fallback_paths]

        return pd.concat(frames, ignore_index=True)


def build_returns_frame_from_next_target(
    df: pd.DataFrame,
    date_col: str,
    stock_col: str,
    target_col: str,
    return_col: str,
) -> pd.DataFrame:
    """
    Convert a next-period target column into an engine-compatible return frame.

    If target_col at signal date t means the return from t to the next market
    trading date, this function assigns that value to the next global date.
    """
    required_cols = [date_col, stock_col, target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"df missing required columns: {missing_cols}")

    work = df[required_cols].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="raise")
    work = work.sort_values([date_col, stock_col], kind="mergesort").reset_index(
        drop=True
    )

    unique_dates = pd.DatetimeIndex(work[date_col].drop_duplicates()).sort_values()

    next_date_map = {
        unique_dates[i]: unique_dates[i + 1]
        for i in range(len(unique_dates) - 1)
    }

    returns_df = work.copy()
    returns_df[date_col] = returns_df[date_col].map(next_date_map)
    returns_df = returns_df.loc[returns_df[date_col].notna()].copy()

    returns_df = returns_df.rename(columns={target_col: return_col})
    returns_df = returns_df[[date_col, stock_col, return_col]]

    returns_df[stock_col] = returns_df[stock_col].astype(str)
    returns_df[return_col] = returns_df[return_col].astype(float)

    finite_mask = np.isfinite(returns_df[return_col].to_numpy())
    returns_df = returns_df.loc[finite_mask].reset_index(drop=True)

    if returns_df.duplicated(subset=[date_col, stock_col]).any():
        raise ValueError(
            f"returns_df contains duplicated {date_col}-{stock_col} rows"
        )

    return returns_df



def _align_preds_to_returns(pred_df, returns_df, config):
    """Filter predictions to keep only stocks with valid next-day returns."""
    import pandas as pd
    ret_dates = pd.DatetimeIndex(returns_df[config.date_col].dropna().unique()).sort_values()
    pred_dates = pd.DatetimeIndex(pred_df[config.date_col].dropna().unique()).sort_values()
    nd_map = {}
    for sd in pred_dates:
        fut = ret_dates[ret_dates > sd]
        if len(fut): nd_map[sd] = fut[0]
    keep = pd.Series(False, index=pred_df.index)
    for sd, nd in nd_map.items():
        stocks_on_nd = returns_df.loc[returns_df[config.date_col] == nd, config.stock_col].unique()
        keep[(pred_df[config.date_col] == sd) & (pred_df[config.stock_col].isin(stocks_on_nd))] = True
    return pred_df[keep].copy()


def _get_model_params(model_name: str, config: ExperimentConfig) -> dict:
    defaults = {"seq_len": config.seq_len, "epochs": config.torch_epochs,
                "lr": config.torch_learning_rate, "wd": config.torch_weight_decay}
    overrides = config.model_params.get(model_name, {})
    defaults.update(overrides)
    return defaults


def get_model_family(model_name: str) -> str:
    """Return the model family used by the experiment runner."""
    normalized_name = model_name.lower()

    if normalized_name in {"lightgbm", "xgboost"}:
        return "tabular"

    if normalized_name in {"dlinear", "itransformer", "patchtst", "tsmixer"}:
        return "torch"

    raise ValueError(f"Unsupported model_name: {model_name}")


def build_experiment_model(
    model_name: str,
    seed: int,
    seq_len: int,
    n_features: int,
) -> Any:
    """Build one experiment model by name."""
    normalized_name = model_name.lower()

    if normalized_name == "lightgbm":
        return build_lightgbm_model(
            LightGBMConfig(
                n_estimators=500,
                learning_rate=0.01,
                num_leaves=31,
                early_stopping_rounds=50,
                verbose_eval=False,
                random_state=seed,
            )
        )

    if normalized_name == "xgboost":
        return build_xgboost_model(
            XGBoostConfig(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=-1,
            )
        )

    if normalized_name == "dlinear":
        return build_dlinear_model(
            DLinearConfig(
                seq_len=seq_len,
                n_features=n_features,
                moving_avg_kernel=25,
                dropout=0.1,
            )
        )

    if normalized_name == "itransformer":
        return build_itransformer_model(
            ITransformerConfig(
                seq_len=seq_len,
                n_features=n_features,
                d_model=128,
                nhead=4,
                num_layers=2,
                dim_feedforward=256,
                dropout=0.1,
            )
        )

    if normalized_name == "patchtst":
        return build_patchtst_model(
            PatchTSTConfig(
                seq_len=seq_len,
                n_features=n_features,
                patch_len=8,
                stride=4,
                d_model=128,
                nhead=4,
                num_layers=2,
                dim_feedforward=256,
                dropout=0.1,
            )
        )

    if normalized_name == "tsmixer":
        return build_tsmixer_model(
            TSMixerConfig(
                seq_len=seq_len,
                n_features=n_features,
                num_blocks=2,
                dropout=0.1,
            )
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value)


def _extract_train_rmse(train_summary: dict[str, Any]) -> float | None:
    if train_summary.get("train_rmse") is not None:
        return float(train_summary["train_rmse"])

    final_train_loss = train_summary.get("final_train_loss")
    if final_train_loss is None:
        return None

    final_train_loss = float(final_train_loss)
    if not np.isfinite(final_train_loss) or final_train_loss < 0.0:
        return None

    return float(np.sqrt(final_train_loss))


def _extract_valid_rmse(train_summary: dict[str, Any]) -> float | None:
    for key in ("valid_rmse", "best_valid_rmse", "final_valid_rmse"):
        if train_summary.get(key) is not None:
            return float(train_summary[key])

    return None


def _build_model_comparison_df(model_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    comparison_rows = []

    for model_name, result in model_results.items():
        summary = result["backtest_summary"]
        train_summary = result["train_summary"]

        comparison_rows.append(
            {
                "model": model_name,
                "model_family": result.get("model_family"),
                "train_rmse": _extract_train_rmse(train_summary),
                "valid_rmse": _extract_valid_rmse(train_summary),
                "sharpe_ratio": summary["sharpe_ratio"],
                "annualized_return": summary["annualized_return"],
                "max_drawdown": summary["max_drawdown"],
                "hit_rate": summary.get("hit_rate"),
                "final_nav": summary["final_nav"],
                "mean_turnover": summary["mean_turnover"],
            }
        )

    return pd.DataFrame(comparison_rows)


def save_markdown_report(
    report_path: str | Path,
    config: ExperimentConfig,
    preprocess_report: dict,
    split_sizes: dict[str, int],
    model_results: dict[str, dict[str, Any]],
    comparison_df: pd.DataFrame,
) -> Path:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Experiment 001: Multi-Model Comparison",
        "",
        "## Config",
        "",
        f"- data_path: `{config.data_path}`",
        f"- date_col: `{config.date_col}`",
        f"- stock_col: `{config.stock_col}`",
        f"- target_col: `{config.target_col}`",
        f"- split: `70/10/20 by unique dates`",
        f"- seq_len: `{config.seq_len}`",
        f"- torch_epochs: `{config.torch_epochs}`",
        f"- train_end: `{config.train_end}`",
        f"- valid_end: `{config.valid_end}`",
        f"- models: `{', '.join(config.model_names)}`",
        f"- top_n: `{config.top_n}`",
        "",
        "## Data Summary",
        "",
        f"- preprocess input rows: {preprocess_report.get('input_rows')}",
        f"- preprocess output rows: {preprocess_report.get('output_rows')}",
        f"- train rows: {split_sizes['train_rows']}",
        f"- valid rows: {split_sizes['valid_rows']}",
        f"- test rows: {split_sizes['test_rows']}",
        "",
        "## Model Comparison",
        "",
        "| Model | Family | Train RMSE | Valid RMSE | Sharpe | Ann. Return | MaxDD | Hit Rate | Final NAV | Mean Turnover |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in comparison_df.iterrows():
        lines.append(
            "| "
            f"{row['model']} | "
            f"{row['model_family']} | "
            f"{_format_value(row['train_rmse'])} | "
            f"{_format_value(row['valid_rmse'])} | "
            f"{_format_value(row['sharpe_ratio'])} | "
            f"{_format_value(row['annualized_return'])} | "
            f"{_format_value(row['max_drawdown'])} | "
            f"{_format_value(row['hit_rate'])} | "
            f"{_format_value(row['final_nav'])} | "
            f"{_format_value(row['mean_turnover'])} |"
        )

    for model_name, result in model_results.items():
        train_summary = result["train_summary"]
        backtest_summary = result["backtest_summary"]

        lines.extend(
            [
                "",
                f"## {model_name} Details",
                "",
                "### Output Files",
                "",
                f"- prediction_path: `{result['prediction_path']}`",
                f"- daily_returns_path: `{result['daily_returns_path']}`",
                f"- daily_nav_path: `{result['daily_nav_path']}`",
                f"- daily_weights_path: `{result['daily_weights_path']}`",
                f"- daily_turnover_path: `{result['daily_turnover_path']}`",
                "",
                "### Training Summary",
                "",
            ]
        )

        for key, value in train_summary.items():
            lines.append(f"- {key}: {value}")

        lines.extend(
            [
                "",
                "### Backtest Summary",
                "",
                "| Metric | Value |",
                "|---|---:|",
            ]
        )

        for key, value in backtest_summary.items():
            lines.append(f"| {key} | {_format_value(value)} |")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    return report_path


def main() -> None:
    config = ExperimentConfig()

    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = _load_experiment_raw_data(config.data_path)

    if config.date_col not in raw_df.columns and config.date_col in raw_df.index.names:
        raw_df = raw_df.reset_index()

    preprocess_config = PreprocessConfig(
        date_col=config.date_col,
        stock_col=config.stock_col,
        feature_cols=list(config.feature_cols),
        target_col=config.target_col,
        meta_cols=list(config.meta_cols),
        replace_inf_with_nan=True,
        drop_rows_with_missing_keys=True,
        drop_rows_with_missing_features=False,
        drop_rows_with_missing_target=True,
        duplicate_policy="raise",
        sort_values=True,
    )

    preprocess_result = preprocess_panel_data(
        df=raw_df,
        config=preprocess_config,
    )

    clean_df = preprocess_result.df

    train_df, valid_df, test_df, config.train_end, config.valid_end = (
        _split_clean_df_7_1_2(
            clean_df,
            config.date_col,
            config.stock_col,
        )
    )

    split_sizes = {
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "test_rows": len(test_df),
    }

    builder = PanelDatasetBuilder(
        feature_cols=list(config.feature_cols),
        target_col=config.target_col,
        date_col=config.date_col,
        stock_col=config.stock_col,
        seq_len=config.seq_len,
        meta_cols=list(config.meta_cols),
    )

    returns_df = clean_df[[config.date_col, config.stock_col, "ret_daily"]].rename(
        columns={"ret_daily": config.return_col}).copy()
    returns_df[config.date_col] = pd.to_datetime(returns_df[config.date_col])
    returns_df[config.stock_col] = returns_df[config.stock_col].astype(str)
    returns_df[config.return_col] = returns_df[config.return_col].astype(float)

    portfolio_config = PortfolioConfig(
        strategy="top_n",
        top_n=config.top_n,
        pred_col="y_pred",
        stock_col=config.stock_col,
    )
    cost_config = TransactionCostConfig()

    model_results: dict[str, dict[str, Any]] = {}
    tabular_names = [n for n in config.model_names if get_model_family(n) == "tabular"]
    torch_names = [n for n in config.model_names if get_model_family(n) == "torch"]

    # ----- phase 1: tabular models (low memory) -----
    if tabular_names:
        tabular_train_data = builder.build_tabular_dataset(train_df)
        tabular_valid_data = builder.build_tabular_dataset(valid_df)
        tabular_test_data = builder.build_tabular_dataset(test_df)

        for model_name in tabular_names:
            print(f"\nRunning model: {model_name}")

            model = build_experiment_model(
                model_name=model_name,
                seed=config.seed,
                seq_len=config.seq_len,
                n_features=len(config.feature_cols),
            )

            train_summary = train_tabular_model(
                model=model,
                train_data=tabular_train_data,
                valid_data=tabular_valid_data,
                output_dir=output_dir / "models",
            )

            pred_df = generate_predictions(
                model=model,
                dataset=tabular_test_data,
                model_name=model_name,
                required_meta_cols=(config.date_col, config.stock_col),
            )

            prediction_path = save_predictions(
                pred_df=pred_df,
                output_path=output_dir / f"predictions_{model_name}.parquet",
            )

            pred_df_bt = _align_preds_to_returns(pred_df, returns_df, config)
            backtest_result = run_backtest(
                pred_df=pred_df_bt,
                returns_df=returns_df,
                portfolio_config=portfolio_config,
                return_col=config.return_col,
                date_col=config.date_col,
                stock_col=config.stock_col,
                periods_per_year=config.periods_per_year,
                cost_config=cost_config,
            )

            daily_returns_path = output_dir / f"daily_returns_{model_name}.csv"
            daily_nav_path = output_dir / f"daily_nav_{model_name}.csv"
            daily_weights_path = output_dir / f"daily_weights_{model_name}.csv"
            daily_turnover_path = output_dir / f"daily_turnover_{model_name}.csv"

            backtest_result["daily_returns"].to_csv(daily_returns_path, header=True)
            backtest_result["daily_nav"].to_csv(daily_nav_path, header=True)
            backtest_result["daily_weights"].to_csv(daily_weights_path)
            backtest_result["daily_turnover"].to_csv(daily_turnover_path, header=True)

            model_results[model_name] = {
                "model_family": "tabular",
                "train_summary": train_summary,
                "backtest_summary": backtest_result["summary"],
                "prediction_path": str(prediction_path),
                "daily_returns_path": str(daily_returns_path),
                "daily_nav_path": str(daily_nav_path),
                "daily_weights_path": str(daily_weights_path),
                "daily_turnover_path": str(daily_turnover_path),
            }

            print(f"{model_name} completed.")
            print(backtest_result["summary"])

        # Release tabular datasets before building heavy sequence data
        del tabular_train_data, tabular_valid_data, tabular_test_data

    # ----- phase 2: torch models (build sequence data per seq_len group) -----
    if torch_names:
        seq_groups = {}
        for model_name in torch_names:
            p = _get_model_params(model_name, config)
            seq_groups.setdefault(p["seq_len"], []).append(model_name)

        prediction_config = PredictionConfig(
            batch_size=config.predict_batch_size,
            device=config.torch_device,
        )

        for sl, models in seq_groups.items():
            print(f"\nBuilding sequence datasets (seq_len={sl}) for: {', '.join(models)}...")
            builder.seq_len = sl
            seq_train = builder.build_sequence_dataset(train_df)
            seq_valid = builder.build_sequence_dataset(valid_df)
            seq_test = builder.build_sequence_dataset(test_df)

            for model_name in models:
                print(f"\nRunning model: {model_name}")
                p = _get_model_params(model_name, config)

                model = build_experiment_model(
                    model_name=model_name,
                    seed=config.seed,
                    seq_len=p["seq_len"],
                    n_features=len(config.feature_cols),
                )

                train_summary = train_torch_model(
                    model=model,
                    train_data=seq_train,
                    valid_data=seq_valid,
                    output_dir=output_dir / "models",
                    config=TorchTrainConfig(
                        epochs=p["epochs"],
                        patience=p.get("patience", config.torch_patience),
                        batch_size=config.torch_batch_size,
                        learning_rate=p["lr"],
                        weight_decay=p["wd"],
                        device=config.torch_device,
                    ),
                )

                pred_df = generate_predictions(
                model=model,
                dataset=seq_test,
                model_name=model_name,
                config=prediction_config,
                required_meta_cols=(config.date_col, config.stock_col),
                )
    
                prediction_path = save_predictions(
                pred_df=pred_df,
                output_path=output_dir / f"predictions_{model_name}.parquet",
                )
    
                # Filter preds to stocks with valid next-day returns
                ret_dates = pd.DatetimeIndex(returns_df[config.date_col].dropna().unique()).sort_values()
                pred_dates = pd.DatetimeIndex(pred_df[config.date_col].dropna().unique()).sort_values()
                nd_map = {}
                for sd in pred_dates:
                    fut = ret_dates[ret_dates > sd]
                    if len(fut): nd_map[sd] = fut[0]
                keep = pd.Series(False, index=pred_df.index)
                for sd, nd in nd_map.items():
                    stocks_on_nd = returns_df.loc[
                        returns_df[config.date_col] == nd, config.stock_col
                    ].unique()
                    keep[(pred_df[config.date_col] == sd) & (pred_df[config.stock_col].isin(stocks_on_nd))] = True
                pred_df_bt = pred_df[keep].copy()

                backtest_result = run_backtest(
                pred_df=pred_df_bt,
                returns_df=returns_df,
                portfolio_config=portfolio_config,
                return_col=config.return_col,
                date_col=config.date_col,
                stock_col=config.stock_col,
                periods_per_year=config.periods_per_year,
                cost_config=cost_config,
                )

                daily_returns_path = output_dir / f"daily_returns_{model_name}.csv"
                daily_nav_path = output_dir / f"daily_nav_{model_name}.csv"
                daily_weights_path = output_dir / f"daily_weights_{model_name}.csv"
                daily_turnover_path = output_dir / f"daily_turnover_{model_name}.csv"

                backtest_result["daily_returns"].to_csv(daily_returns_path, header=True)
                backtest_result["daily_nav"].to_csv(daily_nav_path, header=True)
                backtest_result["daily_weights"].to_csv(daily_weights_path)
                backtest_result["daily_turnover"].to_csv(daily_turnover_path, header=True)
    
                model_results[model_name] = {
                "model_family": "torch",
                "train_summary": train_summary,
                "backtest_summary": backtest_result["summary"],
                "prediction_path": str(prediction_path),
                "daily_returns_path": str(daily_returns_path),
                "daily_nav_path": str(daily_nav_path),
                "daily_weights_path": str(daily_weights_path),
                "daily_turnover_path": str(daily_turnover_path),
                }
    
                print(f"{model_name} completed.")
                print(backtest_result["summary"])
    
    comparison_df = _build_model_comparison_df(model_results)
    comparison_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    report_path = save_markdown_report(
        report_path=config.report_path,
        config=config,
        preprocess_report=preprocess_result.report,
        split_sizes=split_sizes,
        model_results=model_results,
        comparison_df=comparison_df,
    )

    print("\nExperiment completed.")
    print(f"Training target:  {config.target_col}")
    print(f"Backtest return: {config.backtest_return_source} -> {config.return_col}")
    print("Split sizes:")
    print(split_sizes)
    print("\nModel comparison:")
    print(comparison_df)
    print(f"\nModel comparison saved to: {comparison_path}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
