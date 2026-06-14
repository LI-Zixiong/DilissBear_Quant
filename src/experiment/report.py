"""
Experiment report utilities.

This module builds model comparison tables and writes markdown reports.
It should not train models, generate predictions, or run backtests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.experiment.config import ExperimentConfig


CORE_COMPARISON_COLUMNS = [
    "model",
    "model_family",
    "train_rmse",
    "valid_rmse",
    "best_icir",
    "best_icir_epoch",

    "valid_ic_mean",
    "valid_ic_ir",
    "valid_rank_ic_mean",
    "valid_rank_ic_ir",
    "valid_spread_mean",
    "valid_spread_sharpe",
    "valid_sharpe_ratio",
    "valid_annualized_return",
    "valid_max_drawdown",
    "valid_final_nav",
    "valid_mean_turnover",

    "test_ic_mean",
    "test_ic_ir",
    "test_rank_ic_mean",
    "test_rank_ic_ir",
    "test_spread_mean",
    "test_spread_sharpe",
    "test_sharpe_ratio",
    "test_annualized_return",
    "test_max_drawdown",
    "test_final_nav",
    "test_mean_turnover",
]


def format_value(value: Any) -> str:
    """Format metric values for markdown output."""

    if value is None:
        return ""

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{float(value):.6f}"

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    return str(value)


def extract_train_rmse(train_summary: dict[str, Any]) -> float | None:
    """Extract train RMSE from a training summary."""

    if train_summary.get("train_rmse") is not None:
        return float(train_summary["train_rmse"])

    final_train_loss = train_summary.get("final_train_loss")
    if final_train_loss is None:
        return None

    final_train_loss = float(final_train_loss)
    if not np.isfinite(final_train_loss) or final_train_loss < 0.0:
        return None

    return float(np.sqrt(final_train_loss))


def extract_valid_rmse(train_summary: dict[str, Any]) -> float | None:
    """Extract validation RMSE from a training summary."""

    for key in ("valid_rmse", "best_valid_rmse", "final_valid_rmse"):
        if train_summary.get(key) is not None:
            return float(train_summary[key])

    return None


def build_model_comparison_df(
    model_results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """
    Build a model comparison table from model results.

    This function expands:
        - train_summary
        - valid_metrics
        - test_metrics

    For backward compatibility, old columns such as sharpe_ratio and final_nav
    still point to test metrics.
    """

    comparison_rows = []

    for model_name, result in model_results.items():
        train_summary = result.get("train_summary", {})
        summary = result.get("backtest_summary", {})

        valid_metrics = result.get("valid_metrics", {})
        test_metrics = result.get("test_metrics", {})

        row = {
            "model": model_name,
            "model_family": result.get("model_family"),

            "train_rmse": extract_train_rmse(train_summary),
            "valid_rmse": extract_valid_rmse(train_summary),

            # Torch-specific training diagnostics if available.
            "best_icir": train_summary.get("best_icir"),
            "best_icir_epoch": train_summary.get("best_icir_epoch"),
            "best_rmse_epoch": train_summary.get("best_rmse_epoch"),

            # Tree model diagnostics.
            "best_n": train_summary.get("best_n"),
            "total_n": train_summary.get("total_n"),

            **valid_metrics,
            **test_metrics,

            # Backward compatibility: old columns point to test metrics.
            "sharpe_ratio": test_metrics.get(
                "test_sharpe_ratio",
                summary.get("sharpe_ratio"),
            ),
            "annualized_return": test_metrics.get(
                "test_annualized_return",
                summary.get("annualized_return"),
            ),
            "max_drawdown": test_metrics.get(
                "test_max_drawdown",
                summary.get("max_drawdown"),
            ),
            "hit_rate": test_metrics.get(
                "test_hit_rate",
                summary.get("hit_rate"),
            ),
            "final_nav": test_metrics.get(
                "test_final_nav",
                summary.get("final_nav"),
            ),
            "mean_turnover": test_metrics.get(
                "test_mean_turnover",
                summary.get("mean_turnover"),
            ),

            # Useful output paths.
            "prediction_valid_path": result.get("prediction_valid_path"),
            "prediction_test_path": result.get("prediction_test_path"),
            "prediction_path": result.get("prediction_path"),
        }

        comparison_rows.append(row)

    return pd.DataFrame(comparison_rows)


def save_markdown_report(
    report_path: str | Path,
    config: ExperimentConfig,
    preprocess_report: dict,
    split_sizes: dict[str, int],
    model_results: dict[str, dict[str, Any]],
    comparison_df: pd.DataFrame,
) -> Path:
    """
    Save an experiment markdown report.
    """

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    split_text = "/".join(f"{x:.0%}" for x in config.split_ratio)

    lines = [
        "# Experiment Report",
        "",
        "## Config",
        "",
        f"- data_path: `{config.data_path}`",
        f"- date_col: `{config.date_col}`",
        f"- stock_col: `{config.stock_col}`",
        f"- target_col: `{config.target_col}`",
        f"- return_col: `{config.return_col}`",
        f"- backtest_return_mode: `{config.backtest_return_mode}`",
        f"- backtest_return_source: `{config.backtest_return_source}`",
        f"- split: `{split_text} by unique dates`",
        f"- split_ratio: `{config.split_ratio}`",
        f"- models: `{', '.join(config.model_names)}`",
        f"- top_n: `{config.top_n}`",
        f"- seed: `{config.seed}`",
        "",
        "## Data Summary",
        "",
        f"- preprocess input rows: {preprocess_report.get('input_rows')}",
        f"- preprocess output rows: {preprocess_report.get('output_rows')}",
        f"- train rows: {split_sizes['train_rows']}",
        f"- valid rows: {split_sizes['valid_rows']}",
        f"- test rows: {split_sizes['test_rows']}",
        "",
        "## Core Model Comparison",
        "",
    ]

    lines.extend(_markdown_table(comparison_df, CORE_COMPARISON_COLUMNS))

    lines.extend(
        [
            "",
            "## Full Comparison CSV",
            "",
            "The complete metric table is saved as `model_comparison.csv` in the experiment output directory.",
            "",
        ]
    )

    for model_name, result in model_results.items():
        lines.extend(_model_detail_section(model_name, result))

    report_path.write_text("\n".join(lines), encoding="utf-8")

    return report_path


def _markdown_table(df: pd.DataFrame, preferred_columns: list[str]) -> list[str]:
    """Create a markdown table using available preferred columns."""

    available_cols = [col for col in preferred_columns if col in df.columns]

    if not available_cols:
        return ["No comparison columns available."]

    lines = []

    header = "| " + " | ".join(available_cols) + " |"
    align = "| " + " | ".join(["---"] * len(available_cols)) + " |"

    lines.append(header)
    lines.append(align)

    for _, row in df[available_cols].iterrows():
        values = [format_value(row[col]) for col in available_cols]
        lines.append("| " + " | ".join(values) + " |")

    return lines


def _model_detail_section(model_name: str, result: dict[str, Any]) -> list[str]:
    """Build markdown detail section for one model."""

    train_summary = result.get("train_summary", {})
    valid_summary = result.get("valid_backtest_summary", {})
    test_summary = result.get("test_backtest_summary", {})

    lines = [
        "",
        f"## {model_name} Details",
        "",
        "### Output Files",
        "",
    ]

    output_keys = [
        "prediction_valid_path",
        "prediction_test_path",
        "prediction_path",
        "valid_daily_returns_path",
        "valid_daily_nav_path",
        "valid_daily_weights_path",
        "valid_daily_turnover_path",
        "test_daily_returns_path",
        "test_daily_nav_path",
        "test_daily_weights_path",
        "test_daily_turnover_path",
        "daily_returns_path",
        "daily_nav_path",
        "daily_weights_path",
        "daily_turnover_path",
    ]

    for key in output_keys:
        value = result.get(key)
        if value is not None:
            lines.append(f"- {key}: `{value}`")

    lines.extend(
        [
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
            "### Valid Backtest Summary",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )

    for key, value in valid_summary.items():
        lines.append(f"| {key} | {format_value(value)} |")

    lines.extend(
        [
            "",
            "### Test Backtest Summary",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )

    for key, value in test_summary.items():
        lines.append(f"| {key} | {format_value(value)} |")

    return lines