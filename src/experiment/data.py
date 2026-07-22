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
from src.experiment.split import split_panel_by_date_ratio, split_panel_by_dates


@dataclass
class ExperimentData:
    """
    Prepared data bundle for one experiment run.
    """

    raw_df: pd.DataFrame
    clean_df: pd.DataFrame
    inference_df: pd.DataFrame

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
    active_feature_cols: list[str] | None = None,
    drop_missing_target: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply the standard panel preprocessing pipeline for experiments.

    If active_feature_cols is provided, only those factor columns are validated
    and kept (reducing memory for models with restricted factor pools).
    """

    work = raw_df.copy()

    if config.date_col not in work.columns and config.date_col in work.index.names:
        work = work.reset_index()

    feature_cols = active_feature_cols if active_feature_cols is not None else list(config.feature_cols)

    preprocess_config = PreprocessConfig(
        date_col=config.date_col,
        stock_col=config.stock_col,
        feature_cols=feature_cols,
        target_col=config.target_col,
        meta_cols=list(config.meta_cols),
        replace_inf_with_nan=True,
        drop_rows_with_missing_keys=True,
        drop_rows_with_missing_features=False,
        drop_rows_with_missing_target=drop_missing_target,
        duplicate_policy="raise",
        sort_values=True,
    )

    result = preprocess_panel_data(
        df=work,
        config=preprocess_config,
    )

    return result.df, result.report


def _active_column_set(config: ExperimentConfig) -> set[str]:
    """Compute the minimal set of columns needed by active models."""
    needed = set()
    for model_name in config.model_names:
        if config.model_feature_cols and model_name in config.model_feature_cols:
            needed.update(config.model_feature_cols[model_name])
        else:
            needed.update(config.feature_cols)
    needed.add(config.date_col)
    needed.add(config.stock_col)
    needed.add(config.target_col)
    needed.update(config.meta_cols)
    if config.backtest_return_mode == "column":
        needed.add(config.return_col)
    if config.backtest_return_source:
        needed.add(config.backtest_return_source)
    # Force-load columns destined for one-hot expansion so they survive
    # column filtering even when excluded from per-model feature lists.
    for model_name in config.model_names:
        if config.one_hot_features and model_name in config.one_hot_features:
            needed.update(config.one_hot_features[model_name])
    return needed


def expand_onehot_columns(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """One-hot encode a categorical column fit on train, applied to all splits.

    Categories are inferred from *train* only (no leakage).  Missing /
    unseen values safely produce an all-zero row.  uint8 keeps memory low.
    """
    cats = sorted(train_df[col].dropna().unique())
    prefix = f"{col}_"
    new_cols = [f"{prefix}{int(c)}" for c in cats]

    for df in (train_df, valid_df, test_df):
        for i, c in enumerate(cats):
            df[new_cols[i]] = (df[col] == c).astype("uint8")
        df.drop(columns=[col], inplace=True)

    return train_df, valid_df, test_df, new_cols


def prepare_experiment_data(config: ExperimentConfig) -> ExperimentData:
    """
    Prepare all data frames needed by the experiment runner.

    Steps:
        1. Load raw data (only columns needed by active models).
        2. Preprocess panel data.
        3. Split by unique dates.
        4. Build backtest returns.
    """

    raw_df = load_experiment_raw_data(config.data_path)
    active_cols = _active_column_set(config)
    available = [c for c in active_cols if c in raw_df.columns]
    active_factors = sorted(active_cols & set(config.feature_cols))
    # Halve memory before any deep copy: float64 → float32 in-place
    for col in raw_df.columns:
        if raw_df[col].dtype == "float64":
            raw_df[col] = raw_df[col].astype("float32")

    raw_df = raw_df[available]

    inference_df, preprocess_report = preprocess_experiment_data(
        raw_df=raw_df,
        config=config,
        active_feature_cols=active_factors,
        drop_missing_target=False,
    )
    missing_target = inference_df[config.target_col].isna()
    clean_df = inference_df.loc[~missing_target].copy().reset_index(drop=True)
    preprocess_report["dropped_missing_target_rows"] = int(missing_target.sum())
    preprocess_report["output_rows"] = int(len(clean_df))

    if config.end_date:
        clean_df = clean_df[
            clean_df[config.date_col] <= pd.Timestamp(config.end_date)
        ].reset_index(drop=True)
        preprocess_report["end_date_filtered_rows"] = int(len(clean_df))

    if config.date_boundaries is not None:
        train_df, valid_df, test_df, train_end, valid_end = split_panel_by_dates(
            df=clean_df,
            date_col=config.date_col,
            stock_col=config.stock_col,
            boundaries=config.date_boundaries,
            purge=config.purge_days,
            target_horizon=config.target_horizon_days,
        )
    else:
        train_df, valid_df, test_df, train_end, valid_end = split_panel_by_date_ratio(
            df=clean_df,
            date_col=config.date_col,
            stock_col=config.stock_col,
            split_ratio=config.split_ratio,
            purge=config.purge_days,
        )

    config.train_end = train_end
    config.valid_end = valid_end

    # Use inference_df (all dates, ret_daily intact) rather than clean_df
    # (which loses the last ~5 dates when 5d label is still NaN).  The engine
    # maps each signal T → return_date T+1, so even the latest signal dates
    # need a valid return_1d that is already realized.
    returns_df = build_experiment_returns(
        df=inference_df,
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
        inference_df=inference_df,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        returns_df=returns_df,
        preprocess_report=preprocess_report,
        split_sizes=split_sizes,
    )
