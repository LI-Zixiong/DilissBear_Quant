"""
Run grid tuning experiments.

This is the editable tuning entry point.

Change BASE_CONFIG and GRID below to tune different models or parameters.
"""

from src.experiment.config import ExperimentConfig
from src.experiment.tuning import run_grid_search


if __name__ == "__main__":
    BASE_CONFIG = ExperimentConfig(
        # ──────────────────────────────────────────────────────────
        # Data paths & columns
        # ──────────────────────────────────────────────────────────
        data_path="dataset/processed/factor_panel_1500_54_ind.parquet",
        date_col="time",
        stock_col="stock_id",

        # ──────────────────────────────────────────────────────────
        # Target & backtest return
        # ──────────────────────────────────────────────────────────
        target_col="5d_next_raw",
        return_col="return_1d",
        backtest_return_mode="column",
        backtest_return_source="ret_daily",

        # ──────────────────────────────────────────────────────────
        # Factors
        # ──────────────────────────────────────────────────────────
        feature_cols=(
            "F001SIZE", "F002SIZENL", "F003LIQUIDITY", "F004BETA",
            "F005RESVOL", "F006MOMENTUM", "F007LTREV", "F008STREV",
            "F009LEVERAGE", "F010VALUE", "F011EARNYLD", "F012GROWTH",
            "F013REV5", "F014MOM120_20", "F015VOLREV", "F016MAXRET",
            "F017IVOL", "F018AMIHUD", "F019COSTDEV", "F020BP",
            "F021CFP", "F022GPTA", "F023ACCRUAL", "F024ASSETGR",
            "F025GAP", "F026KLEN", "F027KUP", "F028KLOW", "F029KSFT",
            "F030RSV20", "F031RSV60", "F032RANGEZ20", "F033GAPREV5",
            "F034HIGHDEV20", "F035LOWDEV20", "F036VOLSHOCK5", "F037VOLSHOCK20",
            "F038TURNZ20", "F039VSTD20", "F040PVCORR20", "F041RETVOLCORR20",
            "F042AMTCORR20", "F043SLOPE20", "F044RSQR20", "F045RESI20",
            "F046LIMITUP20", "F047LIMITDN20", "F048LIMITSTREAKUP",
            "F049ROE", "F050ROA", "F051GPM", "F052CFOA",
            "F053RD_INTENSITY", "F054RECEIVABLE_RATIO",
        ),
        meta_cols=("industry_sw", "list_date"),

        # This can be overwritten by GRID.
        model_names=("dlinear",),
        model_feature_cols={
            "dlinear": (
                "F001SIZE", "F003LIQUIDITY", "F005RESVOL", "F008STREV",
                "F010VALUE", "F013REV5", "F015VOLREV", "F019COSTDEV",
                "F020BP", "F021CFP",
                "F025GAP", "F030RSV20", "F031RSV60",
                "F043SLOPE20", "F045RESI20",
            ),
        },

        # ──────────────────────────────────────────────────────────
        # Split, portfolio, seed
        # ──────────────────────────────────────────────────────────
        split_ratio=(0.7, 0.1, 0.2),
        top_n=50,
        periods_per_year=252,
        seed=42,

        # ──────────────────────────────────────────────────────────
        # Torch training defaults
        # ──────────────────────────────────────────────────────────
        seq_len=20,
        torch_epochs=10,
        torch_patience=1,
        torch_batch_size=4096,
        torch_learning_rate=7.5e-4,
        torch_weight_decay=0.0,
        torch_device="auto",
        predict_batch_size=8192,

        # ──────────────────────────────────────────────────────────
        # Per-model parameters
        # ──────────────────────────────────────────────────────────
        model_params={
            "lightgbm": {
                "n_estimators": 500,
                "learning_rate": 0.01,
                "num_leaves": 31,
                "early_stopping_rounds": 50,
            },
            "xgboost": {
                "n_estimators": 500,
                "max_depth": 5,
                "learning_rate": 0.01,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            "dlinear": {
                "seq_len": 15,
                "epochs": 15,
                "lr": 7.5e-4,
                "wd": 0.0,
                "patience": 2,
            },
            "itransformer": {
                "seq_len": 40,
                "epochs": 10,
                "lr": 1e-4,
                "wd": 0.0,
                "patience": 2,
            },
            "tsmixer": {
                "seq_len": 40,
                "epochs": 15,
                "lr": 4e-3,
                "wd": 0.0,
                "patience": 2,
            },
            "patchtst": {
                "seq_len": 40,
                "epochs": 10,
                "lr": 1e-4,
                "wd": 0.0,
                "patience": 2,
            },
        },

        # These are overwritten by run_grid_search per trial.
        output_dir="dataset/output/tuning/_base",
        report_path="reports/tuning/_base.md",
    )

    # ──────────────────────────────────────────────────────────────
    # Grid
    # ──────────────────────────────────────────────────────────────
    # Supported key formats:
    #
    # Normal ExperimentConfig field:
    #   "model_names"
    #   "top_n"
    #   "split_ratio"
    #   "feature_cols"
    #   "torch_epochs"
    #
    # Nested model parameter:
    #   "model_params.dlinear.seq_len"
    #   "model_params.dlinear.epochs"
    #   "model_params.dlinear.lr"
    #   "model_params.lightgbm.num_leaves"
    #
    GRID = {
        "model_names": [
            ("dlinear",),
        ],
        "model_params.dlinear.seq_len": [10, 15, 20, 25, 30, 40],
    }

    tuning_name = "dlinear_seq_scan"

    run_grid_search(
        base_config=BASE_CONFIG,
        grid=GRID,
        tuning_name=tuning_name,
        output_root="dataset/output/tuning",
        report_root="reports/tuning",
        sort_by="valid_rank_ic_ir",
        ascending=False,
        continue_on_error=True,
    )