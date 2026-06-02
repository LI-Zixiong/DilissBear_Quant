"""
Experiment tuning utilities.

This module runs a grid of ExperimentConfig variants and summarizes results.

Design:
    - tuning.py does not train models directly.
    - tuning.py does not calculate metrics directly.
    - It builds ExperimentConfig objects.
    - It calls run_experiment(config).
    - It collects each trial's comparison_df into one tuning summary table.
"""

from __future__ import annotations

import copy
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.experiment.config import ExperimentConfig
from src.experiment.runner import run_experiment


@dataclass
class TuningResult:
    """
    Result bundle returned by run_grid_search.
    """

    results_df: pd.DataFrame
    results_path: Path
    sorted_results_path: Path


def run_grid_search(
    base_config: ExperimentConfig,
    grid: Mapping[str, Sequence[Any]],
    tuning_name: str,
    output_root: str | Path = "dataset/output/tuning",
    report_root: str | Path = "reports/tuning",
    sort_by: str = "valid_rank_ic_ir",
    ascending: bool = False,
    continue_on_error: bool = True,
) -> TuningResult:
    """
    Run a grid search over ExperimentConfig parameters.

    Parameters
    ----------
    base_config:
        Base ExperimentConfig. Each trial starts from a deep copy of it.

    grid:
        Parameter grid.

        Supported examples:

            "model_names": [
                ("lightgbm",),
                ("dlinear",),
                ("lightgbm", "dlinear"),
            ]

            "split_ratio": [
                (0.7, 0.1, 0.2),
                (0.6, 0.2, 0.2),
            ]

            "top_n": [30, 50, 80]

            "model_params.dlinear.seq_len": [15, 20, 30]
            "model_params.dlinear.epochs": [7, 8, 9]
            "model_params.dlinear.lr": [1e-3, 7.5e-4]

            "model_params.lightgbm.num_leaves": [31, 63]
            "model_params.lightgbm.learning_rate": [0.01, 0.005]

    tuning_name:
        Name used for output folders and summary CSV files.

    output_root:
        Root folder for all trial outputs.

    report_root:
        Root folder for all trial reports.

    sort_by:
        Column used to sort final results.
        Recommended:
            valid_rank_ic_ir
            valid_spread_sharpe
            valid_sharpe_ratio
            valid_rmse

    ascending:
        Whether smaller sort_by is better.

    continue_on_error:
        If True, failed trials are recorded and the grid continues.
        If False, the first failed trial raises immediately.
    """

    output_root = Path(output_root)
    report_root = Path(report_root)

    tuning_output_dir = output_root / tuning_name
    tuning_report_dir = report_root / tuning_name

    tuning_output_dir.mkdir(parents=True, exist_ok=True)
    tuning_report_dir.mkdir(parents=True, exist_ok=True)

    keys = list(grid.keys())
    values_product = list(itertools.product(*(grid[key] for key in keys)))

    rows: list[dict[str, Any]] = []

    print("=" * 88)
    print(f"Tuning: {tuning_name}")
    print(f"Number of trials: {len(values_product)}")
    print(f"Sort by: {sort_by}, ascending={ascending}")
    print("=" * 88)

    for trial_idx, values in enumerate(values_product, start=1):
        params = dict(zip(keys, values))
        trial_name = make_trial_name(trial_idx, params)

        print("\n" + "-" * 88)
        print(f"Trial {trial_idx}/{len(values_product)}: {trial_name}")
        print(json.dumps(_json_safe(params), ensure_ascii=False, indent=2))
        print("-" * 88)

        config = copy.deepcopy(base_config)

        for key, value in params.items():
            apply_grid_param(config, key, value)

        config.output_dir = str(tuning_output_dir / trial_name)
        config.report_path = str(tuning_report_dir / f"{trial_name}.md")

        try:
            result = run_experiment(config)

            comparison_df = result["comparison_df"].copy()

            for _, model_row in comparison_df.iterrows():
                row = {
                    "trial_idx": trial_idx,
                    "trial_name": trial_name,
                    "status": "ok",
                    "output_dir": config.output_dir,
                    "report_path": config.report_path,
                    **flatten_params(params),
                    **model_row.to_dict(),
                }
                rows.append(row)

        except Exception as exc:
            error_row = {
                "trial_idx": trial_idx,
                "trial_name": trial_name,
                "status": "failed",
                "error": repr(exc),
                "output_dir": config.output_dir,
                "report_path": config.report_path,
                **flatten_params(params),
            }
            rows.append(error_row)

            print(f"[FAILED] {trial_name}")
            print(repr(exc))

            if not continue_on_error:
                raise

    results_df = pd.DataFrame(rows)

    results_path = tuning_output_dir / "tuning_results.csv"
    sorted_results_path = tuning_output_dir / "tuning_results_sorted.csv"

    results_df.to_csv(results_path, index=False)

    sorted_df = sort_tuning_results(
        results_df=results_df,
        sort_by=sort_by,
        ascending=ascending,
    )
    sorted_df.to_csv(sorted_results_path, index=False)

    print("\n" + "=" * 88)
    print("Tuning completed.")
    print(f"Results: {results_path}")
    print(f"Sorted : {sorted_results_path}")

    if sort_by in sorted_df.columns:
        print("\nTop rows:")
        display_cols = [
            col for col in [
                "trial_idx",
                "trial_name",
                "status",
                "model",
                sort_by,
                "valid_rank_ic_ir",
                "valid_spread_sharpe",
                "valid_sharpe_ratio",
                "test_rank_ic_ir",
                "test_spread_sharpe",
                "test_sharpe_ratio",
                "output_dir",
            ]
            if col in sorted_df.columns
        ]
        print(sorted_df[display_cols].head(10).to_string(index=False))

    print("=" * 88)

    return TuningResult(
        results_df=results_df,
        results_path=results_path,
        sorted_results_path=sorted_results_path,
    )


def apply_grid_param(
    config: ExperimentConfig,
    key: str,
    value: Any,
) -> None:
    """
    Apply one grid parameter to ExperimentConfig.

    Supports:
        normal config field:
            top_n
            split_ratio
            model_names
            feature_cols
            torch_epochs

        nested model_params:
            model_params.dlinear.seq_len
            model_params.lightgbm.num_leaves
    """

    if key.startswith("model_params."):
        parts = key.split(".")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid model_params key: {key}. "
                "Expected format: model_params.<model_name>.<param_name>"
            )

        _, model_name, param_name = parts

        if model_name not in config.model_params:
            config.model_params[model_name] = {}

        config.model_params[model_name][param_name] = value
        return

    if not hasattr(config, key):
        raise ValueError(f"ExperimentConfig has no field: {key}")

    setattr(config, key, value)


def sort_tuning_results(
    results_df: pd.DataFrame,
    sort_by: str,
    ascending: bool,
) -> pd.DataFrame:
    """
    Sort tuning results.

    Failed trials are always pushed to the bottom if possible.
    """

    if results_df.empty:
        return results_df.copy()

    work = results_df.copy()

    if sort_by not in work.columns:
        return work

    if "status" in work.columns:
        work["_status_rank"] = (work["status"] != "ok").astype(int)
        work = work.sort_values(
            by=["_status_rank", sort_by],
            ascending=[True, ascending],
            na_position="last",
        ).drop(columns=["_status_rank"])
    else:
        work = work.sort_values(
            by=sort_by,
            ascending=ascending,
            na_position="last",
        )

    return work.reset_index(drop=True)


def make_trial_name(
    trial_idx: int,
    params: Mapping[str, Any],
    max_len: int = 180,
) -> str:
    """
    Create a filesystem-friendly trial name.
    """

    parts = [f"trial_{trial_idx:03d}"]

    for key, value in params.items():
        clean_key = (
            key.replace("model_params.", "")
            .replace(".", "_")
            .replace(" ", "")
        )
        clean_value = value_to_name(value)
        parts.append(f"{clean_key}-{clean_value}")

    name = "__".join(parts)

    if len(name) > max_len:
        name = name[:max_len]

    return name


def value_to_name(value: Any) -> str:
    """
    Convert a grid value into a filesystem-friendly string.
    """

    if isinstance(value, float):
        return f"{value:g}".replace(".", "p").replace("-", "m")

    if isinstance(value, (tuple, list)):
        return "-".join(value_to_name(v) for v in value)

    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "")
        .replace(".", "p")
        .replace("-", "m")
    )


def flatten_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """
    Flatten params for CSV output.
    """

    flat: dict[str, Any] = {}

    for key, value in params.items():
        if isinstance(value, (tuple, list)):
            flat[key] = ",".join(str(v) for v in value)
        else:
            flat[key] = value

    return flat


def _json_safe(params: Mapping[str, Any]) -> dict[str, Any]:
    """
    Convert params into JSON-printable values.
    """

    safe: dict[str, Any] = {}

    for key, value in params.items():
        if isinstance(value, tuple):
            safe[key] = list(value)
        else:
            safe[key] = value

    return safe