"""
Experiment runner.

This module orchestrates the end-to-end experiment flow:

    raw data
        -> preprocess
        -> train / valid / test split
        -> dataset building
        -> model training
        -> valid/test prediction generation
        -> valid/test evaluation
        -> comparison table
        -> markdown report

Model definitions, training loops, prediction generation, metrics, and
backtest engines remain in their own src modules.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd

# LightGBM uses np.array for features; feature names warning is just noise.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from src.backtest.engine import TransactionCostConfig
from src.backtest.portfolio import PortfolioConfig
from src.data.dataset_builder import PanelDatasetBuilder
from src.experiment.config import ExperimentConfig
from src.experiment.data import ExperimentData, prepare_experiment_data
from src.experiment.evaluation import evaluate_prediction_split
from src.experiment.model_factory import (
    build_experiment_model,
    get_active_tabular_models,
    get_active_torch_models,
    get_model_params,
)
from src.experiment.report import build_model_comparison_df, save_markdown_report
from src.predict.generate_predictions import (
    PredictionConfig,
    generate_predictions,
    save_predictions,
)
from src.train.train_tabular import train_tabular_model
from src.train.train_torch import TorchTrainConfig, train_torch_model
from src.utils.seed import set_seed


def _model_features(
    model_name: str, config: ExperimentConfig
) -> tuple[str, ...]:
    """Return feature columns for a specific model."""
    if config.model_feature_cols and model_name in config.model_feature_cols:
        return tuple(config.model_feature_cols[model_name])
    return tuple(config.feature_cols)


def _apply_feature_signs(
    df: pd.DataFrame, model_name: str, config: ExperimentConfig
) -> pd.DataFrame:
    """Flip configured factor signs for one model on a caller-owned copy.

    The function is intentionally strict. A typo in model_feature_signs should
    fail fast instead of silently producing a different experiment.
    """
    if not config.model_feature_signs or model_name not in config.model_feature_signs:
        return df

    for col, sign in config.model_feature_signs[model_name].items():
        if sign not in (-1, 1):
            raise ValueError(
                f"Invalid feature sign for {model_name}.{col}: {sign!r}. "
                "Expected -1 or 1."
            )
        if col not in df.columns:
            raise KeyError(
                f"Feature sign configured for {model_name}.{col}, "
                "but the column is absent from the experiment dataframe."
            )
        if sign == -1:
            df[col] = -df[col]
    return df


def _feature_sign_key(
    model_name: str, config: ExperimentConfig
) -> tuple[tuple[str, int], ...]:
    """Return a hashable representation of a model's sign overrides."""
    if not config.model_feature_signs or model_name not in config.model_feature_signs:
        return ()
    return tuple(
        sorted((str(col), int(sign)) for col, sign in config.model_feature_signs[model_name].items())
    )


def run_experiment(config: ExperimentConfig | None = None) -> dict[str, Any]:
    """
    Run one full experiment.

    Parameters
    ----------
    config:
        ExperimentConfig. If None, use ExperimentConfig().

    Returns
    -------
    dict[str, Any]
        Experiment outputs, including prepared data, model results,
        comparison DataFrame, and report path.
    """

    if config is None:
        config = ExperimentConfig()

    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_experiment_data(config)

    portfolio_config = PortfolioConfig(
        strategy="top_n",
        top_n=config.top_n,
        pred_col="y_pred",
        stock_col=config.stock_col,
    )
    cost_config = TransactionCostConfig()

    model_results: dict[str, dict[str, Any]] = {}

    tabular_names = get_active_tabular_models(config)
    torch_names = get_active_torch_models(config)

    if tabular_names:
        tabular_results = _run_tabular_models(
            config=config,
            data=data,
            model_names=tabular_names,
            output_dir=output_dir,
            portfolio_config=portfolio_config,
            cost_config=cost_config,
        )
        model_results.update(tabular_results)

    if torch_names:
        torch_results = _run_torch_models(
            config=config,
            data=data,
            model_names=torch_names,
            output_dir=output_dir,
            portfolio_config=portfolio_config,
            cost_config=cost_config,
        )
        model_results.update(torch_results)

    comparison_df = build_model_comparison_df(model_results)
    comparison_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    report_path = save_markdown_report(
        report_path=config.report_path,
        config=config,
        preprocess_report=data.preprocess_report,
        split_sizes=data.split_sizes,
        model_results=model_results,
        comparison_df=comparison_df,
    )

    print("\nExperiment completed.")
    print(f"Training target:  {config.target_col}")
    print(
        f"Backtest return: {config.backtest_return_mode}:"
        f"{config.backtest_return_source} -> {config.return_col}"
    )
    print("Split sizes:")
    print(data.split_sizes)
    print("\nModel comparison:")
    print(comparison_df)
    print(f"\nModel comparison saved to: {comparison_path}")
    print(f"Report saved to: {report_path}")

    return {
        "config": config,
        "data": data,
        "model_results": model_results,
        "comparison_df": comparison_df,
        "comparison_path": comparison_path,
        "report_path": report_path,
    }


def _run_tabular_models(
    config: ExperimentConfig,
    data: ExperimentData,
    model_names: tuple[str, ...],
    output_dir: Path,
    portfolio_config: PortfolioConfig,
    cost_config: TransactionCostConfig,
) -> dict[str, dict[str, Any]]:
    """
    Train, predict, and evaluate active tabular models.
    """

    results: dict[str, dict[str, Any]] = {}

    for model_name in model_names:
        print(f"\nRunning model: {model_name}")

        cols = _model_features(model_name, config)
        train_df = data.train_df.copy()
        valid_df = data.valid_df.copy()
        test_df = data.test_df.copy()
        _apply_feature_signs(train_df, model_name, config)
        _apply_feature_signs(valid_df, model_name, config)
        _apply_feature_signs(test_df, model_name, config)

        builder = PanelDatasetBuilder(
            feature_cols=list(cols),
            target_col=config.target_col,
            date_col=config.date_col,
            stock_col=config.stock_col,
            seq_len=1,
            meta_cols=list(config.meta_cols),
        )

        train_data = builder.build_tabular_dataset(train_df)
        valid_data = builder.build_tabular_dataset(valid_df)
        test_data = builder.build_tabular_dataset(test_df)

        # Isolate each model from RNG consumed by any previous model.
        set_seed(config.seed)
        model = build_experiment_model(
            model_name=model_name,
            seed=config.seed,
            seq_len=1,
            n_features=len(cols),
            config=config,
        )

        # Reset again so training-time randomness, if any, also starts from the
        # same state in single-model and multi-model runs.
        set_seed(config.seed)
        train_summary = train_tabular_model(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            # Keep checkpoints for each experiment model isolated. This avoids
            # accidental overwrites when two model names share the same class,
            # and makes full-vs-single debugging cleaner.
            output_dir=output_dir / "models" / model_name,
        )

        valid_pred_df = generate_predictions(
            model=model,
            dataset=valid_data,
            model_name=model_name,
            required_meta_cols=(config.date_col, config.stock_col),
        )

        test_pred_df = generate_predictions(
            model=model,
            dataset=test_data,
            model_name=model_name,
            required_meta_cols=(config.date_col, config.stock_col),
        )

        prediction_paths = _save_prediction_outputs(
            valid_pred_df=valid_pred_df,
            test_pred_df=test_pred_df,
            model_name=model_name,
            output_dir=output_dir,
        )

        valid_eval = _evaluate_and_save_split(
            pred_df=valid_pred_df,
            returns_df=data.returns_df,
            model_name=model_name,
            split_name="valid",
            config=config,
            output_dir=output_dir,
            portfolio_config=portfolio_config,
            cost_config=cost_config,
        )

        test_eval = _evaluate_and_save_split(
            pred_df=test_pred_df,
            returns_df=data.returns_df,
            model_name=model_name,
            split_name="test",
            config=config,
            output_dir=output_dir,
            portfolio_config=portfolio_config,
            cost_config=cost_config,
        )

        results[model_name] = _build_model_result_record(
            model_family="tabular",
            train_summary=train_summary,
            prediction_paths=prediction_paths,
            valid_eval=valid_eval,
            test_eval=test_eval,
        )

        print(f"{model_name} completed.")
        print("Valid metrics:")
        print(valid_eval["backtest_summary"])
        print("Test metrics:")
        print(test_eval["backtest_summary"])

    return results


def _run_torch_models(
    config: ExperimentConfig,
    data: ExperimentData,
    model_names: tuple[str, ...],
    output_dir: Path,
    portfolio_config: PortfolioConfig,
    cost_config: TransactionCostConfig,
) -> dict[str, dict[str, Any]]:
    """
    Train, predict, and evaluate active torch sequence models.

    Models are grouped by seq_len to avoid rebuilding identical sequence
    datasets repeatedly.
    """

    results: dict[str, dict[str, Any]] = {}

    # Group by (seq_len, feature_cols, feature_signs).
    # Signs must be part of the key; otherwise a signed model can silently reuse
    # an unsigned dataset from another model with the same feature list.
    group_key: dict[str, tuple[int, tuple[str, ...], tuple[tuple[str, int], ...]]] = {}
    for model_name in model_names:
        params = get_model_params(model_name, config)
        cols = _model_features(model_name, config)
        group_key[model_name] = (int(params["seq_len"]), cols, _feature_sign_key(model_name, config))

    groups: dict[tuple[int, tuple[str, ...], tuple[tuple[str, int], ...]], list[str]] = {}
    for model_name in model_names:
        groups.setdefault(group_key[model_name], []).append(model_name)

    prediction_config = PredictionConfig(
        batch_size=config.predict_batch_size,
        device=config.torch_device,
    )

    for (seq_len, cols, _sign_key), names in groups.items():
        print(
            f"\nBuilding sequence datasets "
            f"(seq_len={seq_len}, n_feat={len(cols)}) for: {', '.join(names)}..."
        )

        # Apply signs once per compatible group, on copies only.
        sign_model = names[0]
        train_df = data.train_df.copy()
        valid_df = data.valid_df.copy()
        test_df = data.test_df.copy()
        _apply_feature_signs(train_df, sign_model, config)
        _apply_feature_signs(valid_df, sign_model, config)
        _apply_feature_signs(test_df, sign_model, config)

        builder = PanelDatasetBuilder(
            feature_cols=list(cols),
            target_col=config.target_col,
            date_col=config.date_col,
            stock_col=config.stock_col,
            seq_len=seq_len,
            meta_cols=list(config.meta_cols),
        )

        # Build sequence windows on the concatenated chronological panel, then
        # split by the window end date. This keeps validation/test warm-up
        # history available without leaking future targets: each sample still
        # belongs to the split of its endpoint row.
        train_data, valid_data, test_data = builder.build_sequence_splits_from_frames(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
        )
        if valid_data is None or test_data is None:
            raise RuntimeError(
                "Failed to build non-empty valid/test sequence datasets. "
                f"seq_len={seq_len}, models={names}"
            )

        for model_name in names:
            print(f"\nRunning model: {model_name}")

            params = get_model_params(model_name, config)

            # IMPORTANT: reset BEFORE model construction.
            # Otherwise the initial weights depend on RNG consumed by earlier
            # models (e.g. DLinear before GatedDWTCN in the full run).
            set_seed(config.seed)
            model = build_experiment_model(
                model_name=model_name,
                seed=config.seed,
                seq_len=int(params["seq_len"]),
                n_features=len(cols),
                config=config,
            )

            # Reset again so training-time randomness (dropout, CUDA kernels,
            # DataLoader shuffling if enabled) is also independent of model order.
            set_seed(config.seed)
            train_summary = train_torch_model(
                model=model,
                train_data=train_data,
                valid_data=valid_data,
                output_dir=output_dir / "models" / model_name,
                config=TorchTrainConfig(
                    epochs=int(params["epochs"]),
                    patience=int(params.get("patience", config.torch_patience)),
                    batch_size=config.torch_batch_size,
                    learning_rate=float(params["lr"]),
                    weight_decay=float(params["wd"]),
                    device=config.torch_device,
                    seed=config.seed,
                    date_col=config.date_col,
                ),
            )

            valid_pred_df = generate_predictions(
                model=model,
                dataset=valid_data,
                model_name=model_name,
                config=prediction_config,
                required_meta_cols=(config.date_col, config.stock_col),
            )

            test_pred_df = generate_predictions(
                model=model,
                dataset=test_data,
                model_name=model_name,
                config=prediction_config,
                required_meta_cols=(config.date_col, config.stock_col),
            )

            prediction_paths = _save_prediction_outputs(
                valid_pred_df=valid_pred_df,
                test_pred_df=test_pred_df,
                model_name=model_name,
                output_dir=output_dir,
            )

            valid_eval = _evaluate_and_save_split(
                pred_df=valid_pred_df,
                returns_df=data.returns_df,
                model_name=model_name,
                split_name="valid",
                config=config,
                output_dir=output_dir,
                portfolio_config=portfolio_config,
                cost_config=cost_config,
            )

            test_eval = _evaluate_and_save_split(
                pred_df=test_pred_df,
                returns_df=data.returns_df,
                model_name=model_name,
                split_name="test",
                config=config,
                output_dir=output_dir,
                portfolio_config=portfolio_config,
                cost_config=cost_config,
            )

            results[model_name] = _build_model_result_record(
                model_family="torch",
                train_summary=train_summary,
                prediction_paths=prediction_paths,
                valid_eval=valid_eval,
                test_eval=test_eval,
            )

            print(f"{model_name} completed.")
            print("Valid metrics:")
            print(valid_eval["backtest_summary"])
            print("Test metrics:")
            print(test_eval["backtest_summary"])

    return results


def _save_prediction_outputs(
    valid_pred_df,
    test_pred_df,
    model_name: str,
    output_dir: Path,
) -> dict[str, str]:
    """
    Save valid/test predictions.

    For backward compatibility:
        predictions_<model>.parquet is still written as the test prediction file.
    """

    valid_prediction_path = save_predictions(
        pred_df=valid_pred_df,
        output_path=output_dir / f"predictions_valid_{model_name}.parquet",
    )

    test_prediction_path = save_predictions(
        pred_df=test_pred_df,
        output_path=output_dir / f"predictions_test_{model_name}.parquet",
    )

    legacy_prediction_path = save_predictions(
        pred_df=test_pred_df,
        output_path=output_dir / f"predictions_{model_name}.parquet",
    )

    return {
        "prediction_valid_path": str(valid_prediction_path),
        "prediction_test_path": str(test_prediction_path),
        "prediction_path": str(legacy_prediction_path),
    }


def _evaluate_and_save_split(
    pred_df,
    returns_df,
    model_name: str,
    split_name: str,
    config: ExperimentConfig,
    output_dir: Path,
    portfolio_config: PortfolioConfig,
    cost_config: TransactionCostConfig,
) -> dict[str, Any]:
    """
    Evaluate one split and save backtest outputs.
    """

    eval_result = evaluate_prediction_split(
        pred_df=pred_df,
        returns_df=returns_df,
        config=config,
        portfolio_config=portfolio_config,
        cost_config=cost_config,
        split_name=split_name,
        pred_col="y_pred",
        y_true_col="y_true",
        min_obs=10,  # match training-side _compute_icir default
        top_frac=0.1,
    )

    backtest_result = eval_result["backtest_result"]

    paths = _save_backtest_outputs(
        backtest_result=backtest_result,
        model_name=model_name,
        split_name=split_name,
        output_dir=output_dir,
    )

    return {
        "metrics": eval_result["metrics"],
        "backtest_summary": backtest_result["summary"],
        "paths": paths,
    }


def _save_backtest_outputs(
    backtest_result: dict[str, Any],
    model_name: str,
    split_name: str,
    output_dir: Path,
) -> dict[str, str]:
    """
    Save split-specific backtest output files.

    For test split, also write legacy file names without the split prefix.
    """

    daily_returns_path = output_dir / f"daily_returns_{split_name}_{model_name}.csv"
    daily_nav_path = output_dir / f"daily_nav_{split_name}_{model_name}.csv"
    daily_weights_path = output_dir / f"daily_weights_{split_name}_{model_name}.csv"
    daily_turnover_path = output_dir / f"daily_turnover_{split_name}_{model_name}.csv"

    backtest_result["daily_returns"].to_csv(daily_returns_path, header=True)
    backtest_result["daily_nav"].to_csv(daily_nav_path, header=True)
    backtest_result["daily_weights"].to_csv(daily_weights_path)
    backtest_result["daily_turnover"].to_csv(daily_turnover_path, header=True)

    paths = {
        f"{split_name}_daily_returns_path": str(daily_returns_path),
        f"{split_name}_daily_nav_path": str(daily_nav_path),
        f"{split_name}_daily_weights_path": str(daily_weights_path),
        f"{split_name}_daily_turnover_path": str(daily_turnover_path),
    }

    if split_name == "test":
        legacy_returns_path = output_dir / f"daily_returns_{model_name}.csv"
        legacy_nav_path = output_dir / f"daily_nav_{model_name}.csv"
        legacy_weights_path = output_dir / f"daily_weights_{model_name}.csv"
        legacy_turnover_path = output_dir / f"daily_turnover_{model_name}.csv"

        backtest_result["daily_returns"].to_csv(legacy_returns_path, header=True)
        backtest_result["daily_nav"].to_csv(legacy_nav_path, header=True)
        backtest_result["daily_weights"].to_csv(legacy_weights_path)
        backtest_result["daily_turnover"].to_csv(legacy_turnover_path, header=True)

        paths.update(
            {
                "daily_returns_path": str(legacy_returns_path),
                "daily_nav_path": str(legacy_nav_path),
                "daily_weights_path": str(legacy_weights_path),
                "daily_turnover_path": str(legacy_turnover_path),
            }
        )

    return paths


def _build_model_result_record(
    model_family: str,
    train_summary: dict[str, Any],
    prediction_paths: dict[str, str],
    valid_eval: dict[str, Any],
    test_eval: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the model_results record used by report.py and tuning.py.
    """

    return {
        "model_family": model_family,
        "train_summary": train_summary,

        "valid_metrics": valid_eval["metrics"],
        "test_metrics": test_eval["metrics"],

        "valid_backtest_summary": valid_eval["backtest_summary"],
        "test_backtest_summary": test_eval["backtest_summary"],

        # Backward compatibility: old code expects backtest_summary to be test.
        "backtest_summary": test_eval["backtest_summary"],

        **prediction_paths,
        **valid_eval["paths"],
        **test_eval["paths"],
    }