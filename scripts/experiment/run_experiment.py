"""
Run a multi-model end-to-end experiment.

This is the editable experiment entry point. Change the config below to run
different experiments — single model, model pairs, or full five-model benchmark.
"""
from src.experiment.config import ExperimentConfig
from src.experiment.runner import run_experiment


if __name__ == "__main__":
    config = ExperimentConfig(
        # ──────────────────────────────────────────────────────────
        # Data paths & columns
        # ──────────────────────────────────────────────────────────
        data_path="dataset/processed/factor_panel_1500_54_ind.parquet",
        date_col="time",
        stock_col="stock_id",

        # ──────────────────────────────────────────────────────────
        # Target & backtest return
        # ──────────────────────────────────────────────────────────
        target_col="5d_next_raw",                # model learns this
        return_col="return_1d",                  # backtest PnL column
        backtest_return_mode="column",           # "column" = ret_daily, "next_target" = 1d_next_raw
        backtest_return_source="ret_daily",

        # ──────────────────────────────────────────────────────────
        # Factors — 12-factor V1 selection (2026-05-29)
        # ──────────────────────────────────────────────────────────
        # Full 54-factor V1 set (2026-06-01)
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
            "F055IND",
        ),
        meta_cols=("industry_sw", "list_date", "ret_daily"),

        # ──────────────────────────────────────────────────────────
        # Active models
        # ──────────────────────────────────────────────────────────
        #  Current default: LightGBM + Xgboost +DLinear + GatedDWTCN (2026-06-04)
        model_names=("lightgbm", "xgboost", "dlinear", "gated_dwtcn"),

        #  To run a single model (e.g. LGBM only):
        #  model_names=("lightgbm",)

        #  To restore the full five-model benchmark:
        #  model_names=("lightgbm", "xgboost", "dlinear", "itransformer", "tsmixer"),

        # ──────────────────────────────────────────────────────────
        # Per-model factor sets (2026-06-04 audit, ZZ500+1000 universe)
        #   LGBM / iTransformer / TSMixer: |ICIR| >= 0.02, corr < 0.98  (46 factors)
        #   DLinear: |ICIR| >= 0.10, corr < 0.85  (~31 factors, see audit_dlinear)
        # ──────────────────────────────────────────────────────────
        model_feature_signs={},
        model_feature_cols={
            "lightgbm": (
                "F002SIZENL", "F003LIQUIDITY", "F004BETA",
                "F005RESVOL", "F007LTREV", "F008STREV",
                "F010VALUE", "F011EARNYLD", "F012GROWTH",
                "F013REV5", "F015VOLREV", "F016MAXRET",
                "F017IVOL", "F018AMIHUD", "F019COSTDEV", "F020BP",
                "F021CFP", "F023ACCRUAL", "F024ASSETGR",
                "F025GAP", "F026KLEN", "F027KUP", "F028KLOW", "F029KSFT",
                "F030RSV20", "F031RSV60", "F032RANGEZ20", "F033GAPREV5",
                "F035LOWDEV20", "F036VOLSHOCK5", "F037VOLSHOCK20",
                "F038TURNZ20", "F039VSTD20", "F040PVCORR20", "F041RETVOLCORR20",
                "F042AMTCORR20", "F043SLOPE20", "F044RSQR20", "F045RESI20",
                # "F046LIMITUP20", "F047LIMITDN20", "F048LIMITSTREAKUP",
                "F051GPM", "F052CFOA", "F054RECEIVABLE_RATIO",
                "F055IND",
            ),
            "itransformer": (
                "F002SIZENL", "F003LIQUIDITY", "F004BETA",
                "F005RESVOL", "F007LTREV", "F008STREV",
                "F010VALUE", "F011EARNYLD", "F012GROWTH",
                "F013REV5", "F015VOLREV", "F016MAXRET",
                "F017IVOL", "F018AMIHUD", "F019COSTDEV", "F020BP",
                "F021CFP", "F023ACCRUAL", "F024ASSETGR",
                "F025GAP", "F026KLEN", "F027KUP", "F028KLOW", "F029KSFT",
                "F030RSV20", "F031RSV60", "F032RANGEZ20", "F033GAPREV5",
                "F035LOWDEV20", "F036VOLSHOCK5", "F037VOLSHOCK20",
                "F038TURNZ20", "F039VSTD20", "F040PVCORR20", "F041RETVOLCORR20",
                "F042AMTCORR20", "F043SLOPE20", "F044RSQR20", "F045RESI20",
                "F046LIMITUP20", "F047LIMITDN20", "F048LIMITSTREAKUP",
                "F051GPM", "F052CFOA", "F054RECEIVABLE_RATIO",
                "F055IND",
            ),
            "tsmixer": (
                "F002SIZENL", "F003LIQUIDITY", "F004BETA",
                "F005RESVOL", "F007LTREV", "F008STREV",
                "F010VALUE", "F011EARNYLD", "F012GROWTH",
                "F013REV5", "F015VOLREV", "F016MAXRET",
                "F017IVOL", "F018AMIHUD", "F019COSTDEV", "F020BP",
                "F021CFP", "F023ACCRUAL", "F024ASSETGR",
                "F025GAP", "F026KLEN", "F027KUP", "F028KLOW", "F029KSFT",
                "F030RSV20", "F031RSV60", "F032RANGEZ20", "F033GAPREV5",
                "F035LOWDEV20", "F036VOLSHOCK5", "F037VOLSHOCK20",
                "F038TURNZ20", "F039VSTD20", "F040PVCORR20", "F041RETVOLCORR20",
                "F042AMTCORR20", "F043SLOPE20", "F044RSQR20", "F045RESI20",
                "F046LIMITUP20", "F047LIMITDN20", "F048LIMITSTREAKUP",
                "F051GPM", "F052CFOA", "F054RECEIVABLE_RATIO",
                "F055IND",
            ),
            # DLinear (30-factor, current active)
            "dlinear": (
                "F039VSTD20", "F041RETVOLCORR20",
                "F016MAXRET", "F040PVCORR20",
                "F017IVOL", "F018AMIHUD", "F002SIZENL",
                "F026KLEN", "F028KLOW", "F035LOWDEV20",
                "F008STREV", "F019COSTDEV", "F042AMTCORR20",
                "F031RSV60",
                "F005RESVOL", "F003LIQUIDITY", "F025GAP",
                "F027KUP", "F013REV5", "F010VALUE",
                "F038TURNZ20", "F033GAPREV5", "F029KSFT",
                "F021CFP", "F020BP", "F023ACCRUAL",
                "F030RSV20", "F015VOLREV", "F011EARNYLD",
                "F007LTREV", "F052CFOA",
            ),
            "gated_dwtcn": (
                "F002SIZENL", "F003LIQUIDITY", "F004BETA",
                "F005RESVOL", "F007LTREV", "F008STREV",
                "F010VALUE", "F011EARNYLD", "F012GROWTH",
                "F013REV5", "F015VOLREV", "F016MAXRET",
                "F017IVOL", "F018AMIHUD", "F019COSTDEV", "F020BP",
                "F021CFP", "F023ACCRUAL", "F024ASSETGR",
                "F025GAP", "F026KLEN", "F027KUP", "F028KLOW", "F029KSFT",
                "F030RSV20", "F031RSV60", "F032RANGEZ20", "F033GAPREV5",
                "F035LOWDEV20", "F036VOLSHOCK5", "F037VOLSHOCK20",
                "F038TURNZ20", "F039VSTD20", "F040PVCORR20", "F041RETVOLCORR20",
                "F042AMTCORR20", "F043SLOPE20", "F044RSQR20", "F045RESI20",
                "F046LIMITUP20", "F047LIMITDN20", "F048LIMITSTREAKUP",
                "F051GPM", "F052CFOA", "F054RECEIVABLE_RATIO",
                "F055IND",
            ),
        },

        # ──────────────────────────────────────────────────────────
        # Split, portfolio, seed
        # ──────────────────────────────────────────────────────────
        top_n=50,                               # top-N equal-weight portfolio
        periods_per_year=252,

        # ──────────────────────────────────────────────────────────
        # Torch training defaults (fallback for all torch models)
        # ──────────────────────────────────────────────────────────
        seq_len=20,                             # default lookback window
        torch_epochs=10,                        # default max epochs
        torch_patience=1,                       # default early-stopping patience
        torch_batch_size=4096,
        torch_learning_rate=7.5e-4,
        torch_weight_decay=0.0,
        torch_device="auto",                    # "auto" | "cpu" | "cuda"
        predict_batch_size=8192,

        # ──────────────────────────────────────────────────────────
        # Per-model parameters (overrides torch defaults + tree settings)
        # ──────────────────────────────────────────────────────────
        model_params={
            # ---- Tree models ----
            "lightgbm": {
                "n_estimators":          4000,
                "learning_rate":         0.01,
                "num_leaves":            31,
                "early_stopping_rounds": None,
                "feature_fraction":      0.8,
            },
            "xgboost": {
                "n_estimators":          4000,
                "max_depth":             5,
                "learning_rate":         0.02,
                "subsample":             0.8,
                "colsample_bytree":      0.12,
                "early_stopping_rounds": None,
            },

            # ---- Torch models (override torch defaults above) ----
            "dlinear": {
                "seq_len":   20,
                "epochs":    80,
                "lr":        2e-4,
                "wd":        5e-5,
                "patience":  10,
            },
            "gated_dwtcn": {
                "seq_len":   20,
                "epochs":    25,
                "lr":        2e-4,
                "wd":        0.0,
                "patience":  5,
                "kernel_size": 3,
                "dilations": (1, 2, 4),
                "hidden_dim": 32,
                "gate_rank": 8,
            },
            "itransformer": {
                "seq_len":   40,
                "epochs":    10,
                "lr":        1e-4,
                "wd":        0.0,
                "patience":  2,
            },
            "tsmixer": {
                "seq_len":   40,
                "epochs":    15,
                "lr":        4e-3,
                "wd":        0.0,
                "patience":  2,
            },
            "patchtst": {
                "seq_len":   40,
                "epochs":    10,
                "lr":        1e-4,
                "wd":        0.0,
                "patience":  2,
            },
        },

        # ──────────────────────────────────────────────────────────
        # Output
        # ──────────────────────────────────────────────────────────
    )

    run_experiment(config)
