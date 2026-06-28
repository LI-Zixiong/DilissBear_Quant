"""
Calculate 24-factor panel.

This is the editable entry point for full factor computation.
"""

from src.pipeline.factor_panel import (
    DEFAULT_FACTOR_NAMES,
    FactorPanelConfig,
    compute_factor_panel_full,
)


def main() -> None:
    config = FactorPanelConfig(
        base_panel_path="dataset/processed/unified_daily_panel.parquet",
        financial_panel_path="dataset/processed/financial_quarterly_panel.parquet",
        output_path="dataset/processed/factor_panel_1500_54_ind.parquet",
        metadata_path="dataset/processed/factor_panel_1500_54_ind_metadata.json",

        model_start_date="2015-06-01",
        date_col="time",
        stock_col="stock_id",

        factor_names=DEFAULT_FACTOR_NAMES,

        target_names=(
            "1d_next_raw",
            "5d_next_raw",
        ),

        # research: calculate future-return targets.
        # live: targets are set to NaN.
        mode="research",

        beta_window=252,
        beta_min_periods=60,
        liquidity_window=63,
        liquidity_min_periods=20,
        incremental_window=800,

        resid_window=252,
        resid_min_periods=60,
        rev5_window=5,
        mom_long_window=120,
        mom_short_window=20,
        volrev_short_window=5,
        volrev_long_window=60,
        maxret_window=20,
        ivol_short_window=20,
        ivol_long_window=60,
        amihud_short_window=20,
        amihud_long_window=60,
        vwap_window=120,

        winsorize_lower=0.01,
        winsorize_upper=0.99,

        # Industry neutralization OFF by default.
        industry_col=None,
        neutralize_factors=(),

        # Default output keeps final standardized factors only.
        save_raw_factors=False,
        save_winsorized_factors=False,

        factor_version="v1_100f",
        seed=42,

        # ZZ500+ZZ1000 universe (~1500 stocks). Leave empty for full universe.
        universe_paths=(
            "dataset/input/zz500.xls",
            "dataset/input/zz1000.xls",
        ),
    )

    compute_factor_panel_full(config)


if __name__ == "__main__":
    main()