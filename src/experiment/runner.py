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
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

# LightGBM uses np.array for features; feature names warning is just noise.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from src.backtest.engine import TransactionCostConfig
from src.backtest.portfolio import PortfolioConfig
from src.data.dataset_builder import PanelDatasetBuilder
from src.experiment.config import ExperimentConfig
from src.experiment.data import ExperimentData, expand_onehot_columns, prepare_experiment_data
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


def _save_model_features(output_dir: Path, model_name: str, cols: tuple[str, ...]) -> None:
    """Save per-model feature list for audit reproducibility."""
    import json as _json
    path = output_dir / f"model_features_{model_name}.json"
    path.write_text(_json.dumps(list(cols), ensure_ascii=False), encoding="utf-8")


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


def _with_model_param(
    config: ExperimentConfig,
    model_name: str,
    key: str,
    value: Any,
) -> ExperimentConfig:
    """Return a config copy with one extra model param; never mutates input."""
    merged = dict(config.model_params)
    merged[model_name] = {**merged.get(model_name, {}), key: value}
    return replace(config, model_params=merged)


def _persist_industry_mapping(
    output_dir: Path,
    n_industries: int,
    ind_rank: int,
    id_to_code: dict[str, int],
) -> None:
    import json

    path = Path(output_dir) / "models" / "dlinear" / "industry_mapping.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"n_industries": n_industries, "ind_rank": ind_rank,
             "id_to_code": id_to_code},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
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


def _get_total_trees(model) -> int:
    inner = model.model
    if hasattr(inner, "booster_"):
        return inner.booster_.num_trees()
    try:
        return inner.get_booster().num_boosted_rounds()
    except Exception:
        return 1


def _tabular_predict_at_n(model, dataset, n, date_col, stock_col):
    """Predict using first N trees, return pred_df with meta columns."""
    inner = model.model
    X = dataset.X
    n = min(int(n), _get_total_trees(model))
    if hasattr(inner, "best_iteration_"):
        y_pred = inner.predict(X, num_iteration=n)
    else:
        y_pred = inner.predict(X, iteration_range=(0, n))
    pred_df = dataset.meta.copy()
    pred_df["y_pred"] = y_pred
    pred_df["y_true"] = dataset.y
    return pred_df


def _tabular_n_scan(
    model, valid_data, data, config, portfolio_config,
    cost_config, model_name, output_dir,
):
    """Scan candidate n_tree values, pick best by valid total return."""
    total_trees = _get_total_trees(model)
    # 5 evenly-spaced candidates from 20%→100% of total trees
    step = max(1, total_trees // 5)
    candidates = sorted(set(
        [step, step * 2, step * 3, step * 4, total_trees]
    ))
    candidates = [c for c in candidates if c <= total_trees]
    if not candidates:
        candidates = [total_trees]
    best_n = candidates[0]
    best_valid_ret = -float("inf")
    scan_records = []

    print("  n_scan:", end="", flush=True)
    for n in candidates:
        valid_pred = _tabular_predict_at_n(
            model, valid_data, n, config.date_col, config.stock_col,
        )
        valid_eval = evaluate_prediction_split(
            pred_df=valid_pred,
            returns_df=data.returns_df,
            config=config,
            portfolio_config=portfolio_config,
            cost_config=cost_config,
            split_name="valid",
        )
        v_ret = valid_eval["backtest_result"]["summary"]["total_return"]
        v_sr = valid_eval["backtest_result"]["summary"]["sharpe_ratio"]
        v_to = valid_eval["backtest_result"]["summary"]["mean_turnover"]

        scan_records.append({
            "n_tree": n,
            "valid_total_return": v_ret,
            "valid_sharpe": v_sr,
            "valid_turnover": v_to,
        })
        marker = "*" if v_ret > best_valid_ret else ""
        print(f" {n}:{v_ret:.3f}{marker}", end="", flush=True)
        if v_ret > best_valid_ret:
            best_valid_ret = v_ret
            best_n = n
    print(flush=True)

    scan_df = pd.DataFrame(scan_records)
    scan_path = output_dir / f"n_scan_{model_name}.csv"
    scan_df.to_csv(scan_path, index=False)
    print(f"  best_n={best_n}  valid_ret={best_valid_ret:.4f}"
          f"  total_trees={total_trees}")
    print(f"  scan saved to {scan_path}")

    return best_n, best_valid_ret, scan_records


def _filter_train_limit_up_entry(
    train_df: pd.DataFrame,
    date_col: str,
    stock_col: str,
) -> pd.DataFrame:
    """Remove training samples whose signal-day close is limit-up.

    With T-close/post-close entry, a limit-up on T is generally not executable.
    A limit-up on T+1 is an outcome after entry and must remain in the target.
    """
    if "limit_status" not in train_df.columns:
        return train_df

    df = train_df.sort_values([stock_col, date_col])
    before = len(df)
    keep = df["limit_status"].ne(1) | df["limit_status"].isna()
    df = df[keep].copy()
    after = len(df)
    pct = (1 - after / before) * 100 if before else 0
    print(f"  [limit-up filter] train: {before:,} → {after:,} rows ({pct:.1f}% removed)")
    return df


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
        train_df = _filter_train_limit_up_entry(
            data.train_df.copy(), config.date_col, config.stock_col,
        )
        valid_df = data.valid_df.copy()
        test_df = data.test_df.copy()
        _apply_feature_signs(train_df, model_name, config)
        _apply_feature_signs(valid_df, model_name, config)
        _apply_feature_signs(test_df, model_name, config)

        # Append industry_id as native categorical for LightGBM
        _cat_indices: list[int] = []
        _expanded_cols = list(cols)
        if model_name == "lightgbm" and "industry_sw" in train_df.columns:
            all_ind = pd.concat([
                train_df["industry_sw"], valid_df["industry_sw"], test_df["industry_sw"]
            ]).dropna().astype(int)
            if not all_ind.empty:
                unique_ind = sorted(all_ind.unique())
                ind_to_id = {v: i + 1 for i, v in enumerate(unique_ind)}
                for df in (train_df, valid_df, test_df):
                    df["industry_id"] = (
                        df["industry_sw"].fillna(-1).astype(int).map(ind_to_id).fillna(0).astype("int64")
                    )
                _expanded_cols.append("industry_id")
                _cat_indices = [len(_expanded_cols) - 1]
                print(f"  [{model_name}] industry_id categorical: idx={_cat_indices[0]}, {len(unique_ind)} categories")

        builder_meta = list(config.meta_cols)
        if _cat_indices:
            builder_meta.append("industry_id")

        builder = PanelDatasetBuilder(
            feature_cols=_expanded_cols,
            target_col=config.target_col,
            date_col=config.date_col,
            stock_col=config.stock_col,
            seq_len=1,
            meta_cols=builder_meta,
        )

        train_data = builder.build_tabular_dataset(train_df)
        valid_data = builder.build_tabular_dataset(valid_df)
        test_data = builder.build_tabular_dataset(test_df)

        live_date = pd.Timestamp(data.inference_df[config.date_col].max())
        live_df = data.inference_df.loc[
            data.inference_df[config.date_col].eq(live_date)
        ].copy()
        _apply_feature_signs(live_df, model_name, config)

        if _cat_indices and "industry_sw" in live_df.columns:
            all_ind = pd.concat([
                train_df["industry_sw"], valid_df["industry_sw"], test_df["industry_sw"]
            ]).dropna().astype(int)
            unique_ind = sorted(all_ind.unique())
            ind_to_id = {v: i + 1 for i, v in enumerate(unique_ind)}
            live_df["industry_id"] = (
                live_df["industry_sw"].fillna(-1).astype(int).map(ind_to_id).fillna(0).astype("int64")
            )

        live_data = builder.build_tabular_dataset(live_df, require_target=False)

        build_config = (
            _with_model_param(config, model_name, "categorical_feature", _cat_indices)
            if _cat_indices
            else config
        )

        # Isolate each model from RNG consumed by any previous model.
        set_seed(config.seed)
        model = build_experiment_model(
            model_name=model_name,
            seed=config.seed,
            seq_len=1,
            n_features=len(cols),
            config=build_config,
        )

        # Reset again so training-time randomness, if any, also starts from the
        # same state in single-model and multi-model runs.
        set_seed(config.seed)
        train_summary = train_tabular_model(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            output_dir=output_dir / "models" / model_name,
        )

        # Save feature list alongside model for audit reproducibility
        _save_model_features(output_dir, model_name, cols)

        # ── n_tree scan: pick best_n by valid total return ──
        best_n, best_valid_ret, scan_records = _tabular_n_scan(
            model=model,
            valid_data=valid_data,
            data=data,
            config=config,
            portfolio_config=portfolio_config,
            cost_config=cost_config,
            model_name=model_name,
            output_dir=output_dir,
        )
        train_summary["best_n"] = best_n
        train_summary["best_valid_ret"] = best_valid_ret

        # ── Predict with best_n, bypass generate_predictions ──
        valid_pred_df = _tabular_predict_at_n(
            model, valid_data, best_n, config.date_col, config.stock_col,
        )
        test_pred_df = _tabular_predict_at_n(
            model, test_data, best_n, config.date_col, config.stock_col,
        )
        live_pred_df = _tabular_predict_at_n(
            model, live_data, best_n, config.date_col, config.stock_col,
        )

        prediction_paths = _save_prediction_outputs(
            valid_pred_df=valid_pred_df,
            test_pred_df=test_pred_df,
            model_name=model_name,
            output_dir=output_dir,
        )
        live_path = save_predictions(
            pred_df=live_pred_df,
            output_path=output_dir / f"predictions_live_{model_name}.parquet",
        )
        prediction_paths["prediction_live_path"] = str(live_path)

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

        print(f"{model_name} completed."
              f"\n  best_n={best_n}"
              f"  best_valid_ret={best_valid_ret:.4f}"
              f"  total_n={train_summary.get('total_n', 'N/A')}")
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
        train_df = _filter_train_limit_up_entry(
            data.train_df.copy(), config.date_col, config.stock_col,
        )
        valid_df = data.valid_df.copy()
        test_df = data.test_df.copy()
        _apply_feature_signs(train_df, sign_model, config)
        _apply_feature_signs(valid_df, sign_model, config)
        _apply_feature_signs(test_df, sign_model, config)

        # One-hot expansion: per-model categorical → dummy columns.
        # Expands feature_cols in-place so the builder sees the dummies.
        expanded_cols = list(cols)
        _onehot_meta: dict[str, tuple[list[str], list]] = {}  # col → (names, cat_values)
        for model_name in names:
            if config.one_hot_features and model_name in config.one_hot_features:
                for oh_col in config.one_hot_features[model_name]:
                    if oh_col in train_df.columns:
                        cats = sorted(train_df[oh_col].dropna().unique())
                        train_df, valid_df, test_df, new_cols = expand_onehot_columns(
                            train_df, valid_df, test_df, oh_col,
                        )
                        expanded_cols = [c for c in expanded_cols if c != oh_col] + new_cols
                        _onehot_meta[oh_col] = (new_cols, cats)
                        print(f"  [{model_name}] one-hot {oh_col} → {len(new_cols)} columns "
                              f"({len(expanded_cols)} total features)")

        # Create sequential industry_id from industry_sw for DLinear embedding.
        # industry_sw is already in meta_cols. Map to 0..N-1 for nn.Embedding.
        _n_industries = 0
        _ind_to_id: dict = {}
        if "industry_sw" in train_df.columns and not train_df["industry_sw"].dropna().empty:
            all_ind = pd.concat([
                train_df["industry_sw"], valid_df["industry_sw"], test_df["industry_sw"]
            ]).dropna().astype(int)
            unique_ind = sorted(all_ind.unique())
            # UNKNOWN → 0, real industries → 1..N
            _ind_to_id = {v: i + 1 for i, v in enumerate(unique_ind)}
            for df in (train_df, valid_df, test_df):
                df["industry_id"] = (
                    df["industry_sw"].fillna(-1).astype(int).map(_ind_to_id).fillna(0).astype("int64")
                )
            _n_industries = len(unique_ind) + 1  # +1 for UNKNOWN=0
            print(f"  industry_id: {_n_industries} categories (0=UNKNOWN, 1..{len(unique_ind)}={unique_ind[:5]}...)")

            if "dlinear" in names:
                dlinear_params = get_model_params("dlinear", config)
                _persist_industry_mapping(
                    output_dir=output_dir,
                    n_industries=_n_industries,
                    ind_rank=int(dlinear_params.get("ind_rank", 0) or 0),
                    id_to_code={str(v): int(k) for k, v in _ind_to_id.items()},
                )

        builder_meta = list(config.meta_cols)
        if _n_industries > 0:
            builder_meta.append("industry_id")

        builder = PanelDatasetBuilder(
            feature_cols=expanded_cols,
            target_col=config.target_col,
            date_col=config.date_col,
            stock_col=config.stock_col,
            seq_len=seq_len,
            meta_cols=builder_meta,
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

        live_date = pd.Timestamp(data.inference_df[config.date_col].max())
        live_frame = data.inference_df.copy()
        _apply_feature_signs(live_frame, sign_model, config)
        live_frame = (
            live_frame.sort_values([config.stock_col, config.date_col])
            .groupby(config.stock_col, sort=False)
            .tail(seq_len)
            .copy()
        )
        live_frame["_live_endpoint"] = live_frame[config.date_col].eq(live_date)

        if _ind_to_id and "industry_sw" in live_frame.columns:
            live_frame["industry_id"] = (
                live_frame["industry_sw"].fillna(-1).astype(int).map(_ind_to_id).fillna(0).astype("int64")
            )

        for oh_col, (oh_names, cats) in _onehot_meta.items():
            if oh_col in live_frame.columns:
                for name, cat in zip(oh_names, cats):
                    live_frame[name] = (live_frame[oh_col] == cat).astype("uint8")
                live_frame.drop(columns=[oh_col], inplace=True)

        live_data = builder.build_sequence_dataset(
            live_frame,
            end_filter_col="_live_endpoint",
            end_filter_value=True,
            require_target=False,
        )

        for model_name in names:
            print(f"\nRunning model: {model_name}")

            params = get_model_params(model_name, config)

            build_config = (
                _with_model_param(config, model_name, "n_industries", _n_industries)
                if _n_industries > 0
                else config
            )

            # IMPORTANT: reset BEFORE model construction.
            # Otherwise the initial weights depend on RNG consumed by earlier
            # models (e.g. DLinear before GatedDWTCN in the full run).
            set_seed(config.seed)
            model = build_experiment_model(
                model_name=model_name,
                seed=config.seed,
                seq_len=int(params["seq_len"]),
                n_features=len(expanded_cols),
                config=build_config,
            )

            # Reset again so training-time randomness (dropout, CUDA kernels,
            # DataLoader shuffling if enabled) is also independent of model order.
            set_seed(config.seed)
            ttc = TorchTrainConfig(
                epochs=int(params["epochs"]),
                patience=int(params.get("patience", config.torch_patience)),
                batch_size=config.torch_batch_size,
                learning_rate=float(params["lr"]),
                weight_decay=float(params["wd"]),
                device=config.torch_device,
                seed=config.seed,
                date_col=config.date_col,
                enable_compile=bool(params.get("enable_compile", False)),
                loss_type=str(params.get("loss_type", "mse")),
                # Forward RankNet-specific params (use defaults when absent)
                rank_pos_quantile=float(params.get("rank_pos_quantile", 0.95)),
                rank_neg_quantile=float(params.get("rank_neg_quantile", 0.80)),
                rank_tau=float(params.get("rank_tau", 0.20)),
                rank_weight_mode=str(params.get("rank_weight_mode", "return_diff_log_mad")),
                rank_hard_neg_frac=float(params.get("rank_hard_neg_frac", 0.20)),
                rank_hard_neg_score_quantile=float(params.get("rank_hard_neg_score_quantile", 0.90)),
                rank_hard_neg_y_quantile=float(params.get("rank_hard_neg_y_quantile", 0.30)),
                rank_hard_neg_mode=str(params.get("rank_hard_neg_mode", "score")),
                rank_hard_neg_warmup_epochs=int(params.get("rank_hard_neg_warmup_epochs", 3)),
                rank_tail_neg_frac=float(params.get("rank_tail_neg_frac", 0.10)),
                early_stop_metric=str(params.get("early_stop_metric", "auto")),
                pairs_per_pos=int(params.get("pairs_per_pos", 10)),
                dates_per_batch=int(params.get("dates_per_batch", 8)),
            )
            train_summary = train_torch_model(
                model=model,
                train_data=train_data,
                valid_data=valid_data,
                output_dir=output_dir / "models" / model_name,
                config=ttc,
            )

            # Save feature list alongside model for audit reproducibility
            _save_model_features(output_dir, model_name, cols)

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

            live_pred_df = generate_predictions(
                model=model,
                dataset=live_data,
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
            live_path = save_predictions(
                pred_df=live_pred_df,
                output_path=output_dir / f"predictions_live_{model_name}.parquet",
            )
            prediction_paths["prediction_live_path"] = str(live_path)

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

            print(f"{model_name} completed."
                  f"\n  best_composite_epoch={train_summary.get('best_composite_epoch', 'N/A')}"
                  f"  selection_metric={train_summary.get('selection_metric', 'N/A')}")
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
