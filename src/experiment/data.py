"""
Experiment data preparation.

This module loads raw experiment data, applies the standard preprocessing
pipeline, creates chronological train/valid/test splits, and builds the
backtest return frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.loader import load_panel_data
from src.data.preprocess import PreprocessConfig, preprocess_panel_data
from src.experiment.config import ExperimentConfig
from src.experiment.returns import build_experiment_returns
from src.experiment.split import split_panel_by_date_ratio


@dataclass
class ExperimentData:
    """
    Prepared data bundle for one experiment run.
    """

    raw_df: pd.DataFrame
    clean_df: pd.DataFrame

    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame

    returns_df: pd.DataFrame

    preprocess_report: dict
    split_sizes: dict[str, int]


def load_experiment_raw_data(data_path: str | Path) -> pd.DataFrame:
    """
    Load experiment input data.

    If the aggregate parquet is unreadable because of the known parquet thrift
    issue, fall back to split parquet files in the same directory.

    This preserves the behavior from the original scripts/run_experiment.py.
    """

    data_path = Path(data_path)

    try:
        return load_panel_data(data_path)
    except OSError as exc:
        error_message = str(exc)

        if "Couldn't deserialize thrift" not in error_message:
            raise

        required_fallback_paths = [
            data_path.with_name("df_response_daily_train.parquet"),
            data_path.with_name("df_response_daily_validate.parquet"),
        ]
        optional_fallback_paths = [
            data_path.with_name("df_response_daily_test.parquet"),
        ]

        missing_paths = [
            path for path in required_fallback_paths
            if not path.exists()
        ]
        if missing_paths:
            missing_text = ", ".join(str(path) for path in missing_paths)
            raise ValueError(
                "Failed to read aggregate parquet and fallback files are missing: "
                f"{missing_text}"
            ) from exc

        fallback_paths = required_fallback_paths + [
            path for path in optional_fallback_paths
            if path.exists()
        ]

        frames = [load_panel_data(path) for path in fallback_paths]

        return pd.concat(frames, ignore_index=True)


def preprocess_experiment_data(
    raw_df: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply the standard panel preprocessing pipeline for experiments.
    """

    work = raw_df.copy()

    if config.date_col not in work.columns and config.date_col in work.index.names:
        work = work.reset_index()

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

    result = preprocess_panel_data(
        df=work,
        config=preprocess_config,
    )

    return result.df, result.report


def prepare_experiment_data(config: ExperimentConfig) -> ExperimentData:
    """
    Prepare all data frames needed by the experiment runner.

    Steps:
        1. Load raw data.
        2. Preprocess panel data.
        3. Split by unique dates.
        4. Build backtest returns.
    """

    raw_df = load_experiment_raw_data(config.data_path)

    clean_df, preprocess_report = preprocess_experiment_data(
        raw_df=raw_df,
        config=config,
    )

    train_df, valid_df, test_df, train_end, valid_end = split_panel_by_date_ratio(
        df=clean_df,
        date_col=config.date_col,
        stock_col=config.stock_col,
        split_ratio=config.split_ratio,
    )

    config.train_end = train_end
    config.valid_end = valid_end

    returns_df = build_experiment_returns(
        df=clean_df,
        date_col=config.date_col,
        stock_col=config.stock_col,
        return_col=config.return_col,
        source_col=config.backtest_return_source,
        source_mode=config.backtest_return_mode,
    )

    split_sizes = {
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "test_rows": len(test_df),
    }

    return ExperimentData(
        raw_df=raw_df,
        clean_df=clean_df,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        returns_df=returns_df,
        preprocess_report=preprocess_report,
        split_sizes=split_sizes,
    )