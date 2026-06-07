"""
Central path configuration for the entire project.

Import as:   from src.config.paths import PATHS
Usage:       pd.read_parquet(PATHS.factor_panel)
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    # ── Input data ──
    zz500_constituents: Path = Path("dataset/input/zz500.xls")
    zz1000_constituents: Path = Path("dataset/input/zz1000.xls")
    zz500_daily: Path = Path("dataset/input/zz500daily.xlsx")
    zz1000_daily: Path = Path("dataset/input/zz1000daily.xlsx")
    hs300_index: Path = Path("dataset/input/original_data/TRD_Index.csv")
    industry_mapping: Path = Path("dataset/input/Aindustry.xlsx")

    # ── Processed data (pipeline outputs) ──
    unified_daily: Path = Path("dataset/processed/unified_daily_panel.parquet")
    financial_quarterly: Path = Path("dataset/processed/financial_quarterly_panel.parquet")
    factor_panel: Path = Path("dataset/processed/factor_panel_54_ind.parquet")
    universe: Path = Path("dataset/processed/factor_panel_1500_54_ind.parquet")

    # ── Experiment outputs ──
    experiment_dir: Path = Path("dataset/output/experiment_003")
    tuning_dir: Path = Path("dataset/output/tuning")

    # ── Ensemble outputs (separate from model predictions) ──
    ensemble_dir: Path = Path("dataset/output/ensemble")
    best_ridge_score: Path = Path("dataset/output/ensemble/best_ridge_score_test.parquet")
    best_ridge_weights: Path = Path("dataset/output/ensemble/best_ridge_weights.csv")
    ensemble_results: Path = Path("dataset/output/ensemble/ensemble_results.csv")
    ensemble_summary: Path = Path("dataset/output/ensemble/ensemble_summary.txt")

    # ── Strategy outputs ──
    strategy_dir: Path = Path("reports/strategy_v1")
    strategy_results: Path = Path("reports/strategy_v1/strategy_results.csv")
    strategy_summary: Path = Path("reports/strategy_v1/strategy_summary.txt")
    strategy_daily_returns: Path = Path("reports/strategy_v1/strategy_daily_returns.csv")
    strategy_holdings: Path = Path("reports/strategy_v1/strategy_holdings.parquet")


PATHS = DataPaths()
