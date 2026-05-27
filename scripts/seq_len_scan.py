"""
Scan seq_len values for TSMixer and collect comparison metrics.

Runs experiment_002 (seq_len=30), 003 (seq_len=40), 004 (seq_len=60)
with only lightgbm + tsmixer. Collects results automatically.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_experiment import (
    ExperimentConfig,
    _split_clean_df_7_1_2,
    _load_experiment_raw_data,
    build_returns_frame_from_next_target,
    build_experiment_model,
    get_model_family,
    save_markdown_report,
    _build_model_comparison_df,
)
from src.backtest.engine import TransactionCostConfig, run_backtest
from src.backtest.portfolio import PortfolioConfig
from src.data.dataset_builder import PanelDatasetBuilder
from src.data.preprocess import PreprocessConfig, preprocess_panel_data
from src.predict.generate_predictions import (
    PredictionConfig,
    generate_predictions,
    save_predictions,
)
from src.backtest.metrics import prediction_ic_summary, top_bottom_spread
from src.train.train_tabular import train_tabular_model
from src.train.train_torch import TorchTrainConfig, train_torch_model
from src.utils.seed import set_seed


def run_one_experiment(config: ExperimentConfig) -> dict:
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

    result = preprocess_panel_data(raw_df, preprocess_config)
    clean_df = result.df

    train_df, valid_df, test_df, config.train_end, config.valid_end = (
        _split_clean_df_7_1_2(clean_df, config.date_col, config.stock_col)
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

    returns_df = build_returns_frame_from_next_target(
        df=clean_df,
        date_col=config.date_col,
        stock_col=config.stock_col,
        target_col=config.backtest_return_source,
        return_col=config.return_col,
    )

    portfolio_config = PortfolioConfig(
        strategy="top_n",
        top_n=config.top_n,
        pred_col="y_pred",
        stock_col=config.stock_col,
    )
    cost_config = TransactionCostConfig()

    model_results = {}
    tabular_names = [n for n in config.model_names if get_model_family(n) == "tabular"]
    torch_names = [n for n in config.model_names if get_model_family(n) == "torch"]

    # ---- tabular phase ----
    if tabular_names:
        t_train = builder.build_tabular_dataset(train_df)
        t_valid = builder.build_tabular_dataset(valid_df)
        t_test = builder.build_tabular_dataset(test_df)

        for model_name in tabular_names:
            print(f"\nRunning model: {model_name}")
            model = build_experiment_model(
                model_name, seed=config.seed, seq_len=config.seq_len,
                n_features=len(config.feature_cols),
            )
            train_summary = train_tabular_model(
                model=model, train_data=t_train, valid_data=t_valid,
                output_dir=output_dir / "models",
            )
            pred_df = generate_predictions(
                model=model, dataset=t_test, model_name=model_name,
                required_meta_cols=(config.date_col, config.stock_col),
            )
            save_predictions(pred_df, output_dir / f"predictions_{model_name}.parquet")
            bt = run_backtest(
                pred_df=pred_df, returns_df=returns_df,
                portfolio_config=portfolio_config,
                return_col=config.return_col,
                date_col=config.date_col, stock_col=config.stock_col,
                periods_per_year=config.periods_per_year,
                cost_config=cost_config,
            )
            for suffix, data in [
                ("daily_returns", bt["daily_returns"]),
                ("daily_nav", bt["daily_nav"]),
                ("daily_weights", bt["daily_weights"]),
                ("daily_turnover", bt["daily_turnover"]),
            ]:
                data.to_csv(output_dir / f"{suffix}_{model_name}.csv", header=True)
            model_results[model_name] = {
                "model_family": "tabular",
                "train_summary": train_summary,
                "backtest_summary": bt["summary"],
                "prediction_path": str(output_dir / f"predictions_{model_name}.parquet"),
                "daily_returns_path": str(output_dir / f"daily_returns_{model_name}.csv"),
                "daily_nav_path": str(output_dir / f"daily_nav_{model_name}.csv"),
                "daily_weights_path": str(output_dir / f"daily_weights_{model_name}.csv"),
                "daily_turnover_path": str(output_dir / f"daily_turnover_{model_name}.csv"),
            }
            print(f"{model_name} completed.")
            print(bt["summary"])

        del t_train, t_valid, t_test

    # ---- torch phase ----
    if torch_names:
        s_train = builder.build_sequence_dataset(train_df)
        s_valid = builder.build_sequence_dataset(valid_df)
        s_test = builder.build_sequence_dataset(test_df)

        p_config = PredictionConfig(
            batch_size=config.predict_batch_size,
            device=config.torch_device,
        )

        for model_name in torch_names:
            print(f"\nRunning model: {model_name}")
            model = build_experiment_model(
                model_name, seed=config.seed, seq_len=config.seq_len,
                n_features=len(config.feature_cols),
            )
            train_summary = train_torch_model(
                model=model, train_data=s_train, valid_data=s_valid,
                output_dir=output_dir / "models",
                config=TorchTrainConfig(
                    epochs=config.torch_epochs,
                    patience=config.torch_patience,
                    batch_size=config.torch_batch_size,
                    learning_rate=config.torch_learning_rate,
                    weight_decay=config.torch_weight_decay,
                    device=config.torch_device,
                ),
            )

            # Validation backtest + IC — primary metrics for model selection
            valid_pred_df = generate_predictions(
                model=model, dataset=s_valid, model_name=model_name,
                config=p_config,
                required_meta_cols=(config.date_col, config.stock_col),
            )
            valid_ic = prediction_ic_summary(
                pred_df=valid_pred_df,
                date_col=config.date_col,
                y_true_col="y_true", y_pred_col="y_pred",
                min_obs=50,
            )
            valid_spread = top_bottom_spread(
                pred_df=valid_pred_df,
                date_col=config.date_col,
                y_true_col="y_true", y_pred_col="y_pred",
                top_frac=0.1, min_obs=50,
            )
            bt_valid = run_backtest(
                pred_df=valid_pred_df, returns_df=returns_df,
                portfolio_config=portfolio_config,
                return_col=config.return_col,
                date_col=config.date_col, stock_col=config.stock_col,
                periods_per_year=config.periods_per_year,
                cost_config=cost_config,
            )
            valid_summary = bt_valid["summary"]

            # Test backtest — final evaluation only
            test_pred_df = generate_predictions(
                model=model, dataset=s_test, model_name=model_name,
                config=p_config,
                required_meta_cols=(config.date_col, config.stock_col),
            )
            save_predictions(test_pred_df, output_dir / f"predictions_{model_name}.parquet")
            bt_test = run_backtest(
                pred_df=test_pred_df, returns_df=returns_df,
                portfolio_config=portfolio_config,
                return_col=config.return_col,
                date_col=config.date_col, stock_col=config.stock_col,
                periods_per_year=config.periods_per_year,
                cost_config=cost_config,
            )
            test_summary = bt_test["summary"]

            for suffix, data in [
                ("daily_returns", bt_test["daily_returns"]),
                ("daily_nav", bt_test["daily_nav"]),
                ("daily_weights", bt_test["daily_weights"]),
                ("daily_turnover", bt_test["daily_turnover"]),
            ]:
                data.to_csv(output_dir / f"{suffix}_{model_name}.csv", header=True)
            model_results[model_name] = {
                "model_family": "torch",
                "train_summary": train_summary,
                "valid_ic": valid_ic,
                "valid_spread": valid_spread,
                "valid_backtest_summary": valid_summary,
                "test_backtest_summary": test_summary,
                "backtest_summary": test_summary,  # legacy key for save_markdown_report
                "prediction_path": str(output_dir / f"predictions_{model_name}.parquet"),
                "daily_returns_path": str(output_dir / f"daily_returns_{model_name}.csv"),
                "daily_nav_path": str(output_dir / f"daily_nav_{model_name}.csv"),
                "daily_weights_path": str(output_dir / f"daily_weights_{model_name}.csv"),
                "daily_turnover_path": str(output_dir / f"daily_turnover_{model_name}.csv"),
            }
            print(f"{model_name} completed.")
            print(f"  valid_rIC={valid_ic['rank_ic_mean']:.4f}  "
                  f"valid_spread={valid_spread['spread_sharpe']:.4f}  "
                  f"valid_sharpe={valid_summary['sharpe_ratio']:.4f}  "
                  f"valid_rmse={train_summary.get('best_valid_rmse', float('nan')):.4f}  "
                  f"valid_turnover={valid_summary['mean_turnover']:.4f}")
            print(f"  test_sharpe ={test_summary['sharpe_ratio']:.4f}  "
                  f"test_turnover={test_summary['mean_turnover']:.4f}")

    comparison_df = _build_model_comparison_df(model_results)
    comparison_df["seq_len"] = config.seq_len

    save_markdown_report(
        report_path=config.report_path,
        config=config,
        preprocess_report=result.report,
        split_sizes=split_sizes,
        model_results=model_results,
        comparison_df=comparison_df,
    )

    return {
        "seq_len": config.seq_len,
        "model_results": model_results,
        "comparison_df": comparison_df,
    }


def main():
    epoch_values = [7, 8, 9]

    results = {}
    all_rows = []
    for epoch in epoch_values:
        print(f"\n{'='*60}")
        print(f"  DLinear epochs = {epoch}")
        print(f"{'='*60}\n")

        config = ExperimentConfig()
        config.seq_len = 20
        config.model_names = ("dlinear",)
        config.torch_epochs = epoch
        config.torch_patience = 0
        config.torch_learning_rate = 1e-3
        config.output_dir = f"dataset/output/experiment_ep{epoch:02d}"
        config.report_path = f"reports/experiment_ep{epoch:02d}.md"

        result = run_one_experiment(config)
        results[epoch] = result

    # ---- summary with composite score ----
    print("\n\n" + "=" * 80)
    print("  DLinear EPOCH SCAN (5d, seq=20, lr=1e-3)  — Composite Score Test")
    print("=" * 80)

    rows = []
    for epoch in epoch_values:
        m = results[epoch]["model_results"]["dlinear"]
        train = m["train_summary"]
        vs = m["valid_backtest_summary"]
        vic = m["valid_ic"]
        vsp = m["valid_spread"]
        rows.append({
            "epoch": epoch,
            "v_rmse": train["best_valid_rmse"],
            "v_icir": vic["rank_ic_ir"],
            "v_sharpe": vs["sharpe_ratio"],
            "t_sharpe": m["test_backtest_summary"]["sharpe_ratio"],
            "t_turn": m["test_backtest_summary"]["mean_turnover"],
        })

    # Percentile-rank normalize (RMSE lower is better → invert)
    rmses = pd.Series([r["v_rmse"] for r in rows])
    icirs = pd.Series([r["v_icir"] for r in rows])
    sharpes = pd.Series([r["v_sharpe"] for r in rows])

    pct = lambda s: (s.rank() - 1) / (len(s) - 1) if len(s) > 1 else pd.Series([0.5])
    for i, r in enumerate(rows):
        r["p_rmse"]  = 1.0 - pct(rmses)[i]   # lower rmse = better
        r["p_icir"]  = pct(icirs)[i]
        r["p_sharpe"] = pct(sharpes)[i]
        r["composite"] = (
            r["p_icir"] * 0.50
            + r["p_sharpe"] * 0.30
            + r["p_rmse"] * 0.20
        )

    header = (
        f"{'ep':>4}  "
        f"{'v_rmse':>8}  {'p_rmse':>7}  "
        f"{'v_icir':>8}  {'p_icir':>7}  "
        f"{'v_sharpe':>9}  {'p_sh':>7}  "
        f"{'composite':>9}  {'t_sharpe':>9}  {'pick':>4}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    best_composite = max(rows, key=lambda r: r["composite"])
    best_test = max(rows, key=lambda r: r["t_sharpe"])

    for r in rows:
        pick = ""
        if r["epoch"] == best_composite["epoch"]:
            pick += "C"
        if r["epoch"] == best_test["epoch"]:
            pick += "T"
        print(
            f"{r['epoch']:>4}  "
            f"{r['v_rmse']:>8.4f}  {r['p_rmse']:>7.3f}  "
            f"{r['v_icir']:>8.4f}  {r['p_icir']:>7.3f}  "
            f"{r['v_sharpe']:>9.4f}  {r['p_sharpe']:>7.3f}  "
            f"{r['composite']:>9.4f}  {r['t_sharpe']:>9.4f}  {pick:>4}"
        )

    print(f"\nC = composite pick, T = test Sharpe pick")
    print(f"Composite selects epoch {best_composite['epoch']} (t_sharpe={best_composite['t_sharpe']:.4f})")
    print(f"Oracle    selects epoch {best_test['epoch']} (t_sharpe={best_test['t_sharpe']:.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
