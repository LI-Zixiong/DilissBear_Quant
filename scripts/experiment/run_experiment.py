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
        backtest_return_mode="next_target",       # open-to-open: 1d_next_raw → return_1d
        backtest_return_source="1d_next_raw",

        # ──────────────────────────────────────────────────────────
        # Factors — Full 100-factor set (2026-06-25)
        # ──────────────────────────────────────────────────────────
        feature_cols=(
            "F001SIZE", "F002SIZENL", "F003LIQUIDITY", "F004BETA", "F005RESVOL",
            "F006MOMENTUM", "F007LTREV", "F008STREV", "F009LEVERAGE", "F010VALUE",
            "F011EARNYLD", "F012GROWTH", "F013REV5", "F014MOM120_20", "F015VOLREV",
            "F016MAXRET", "F017IVOL", "F018AMIHUD", "F019COSTDEV", "F020LIMITUP_RECENCY", "F021CFP",
            "F022GPTA", "F023ACCRUAL", "F024ASSETGR", "F025GAP", "F026KLEN", "F027KUP",
            "F028KLOW", "F029KSFT", "F030RSV20", "F031RSV60", "F032RANGEZ20",
            "F033GAPREV5", "F034HIGHDEV20", "F035LOWDEV20", "F036VOLSHOCK5",
            "F037VOLSHOCK20", "F038TURNZ20", "F039VSTD20", "F040PVCORR20",
            "F041RETVOLCORR20", "F042AMTCORR20", "F043SLOPE20", "F044RSQR20",
            "F045RESI20", "F046LIMITUP20", "F047LIMITDN20", "F048LIMITSTREAKUP",
            "F049ROE", "F050ROA", "F051GPM", "F052CFOA", "F053RD_INTENSITY",
            "F054RECEIVABLE_RATIO", "F055IND", "F056GAP_UP_FAIL", "F057INTRA1", "F058O2O_RET5", "F059GK_VOL20", "F060ON_INTRA_DIV5",
            "F061GAP_UP_HOLD", "F062GAP_DN_RECOVER", "F063RET5D_SKIP1", "F064RET_ACCEL20", "F065MAXDD20", "F066EFFICIENCY20", "F067TAIL_LOSS20", "F068SKEW20", "F069DNVOL20",
            "F070UP_DN_VOL", "F071VOL_OF_VOL", "F072CORR_60D", "F073KURT_60D", "F074BETA_20D", "F075VOLUME_RATIO", "F076SIGNED_AMT20", "F077AMP_VOL20", "F078TURN_SIZE",
            "F079TURN_ACCEL", "F080VWAP_DEV", "F081STRONG_CLOSE", "F082LOCKED_PCT", "F083TURN_FREE", "F084AMT_FREE20", "F085SP_TTM", "F086DIV_TTM", "F087LIST_AGE",
            "F088CF_SALES_Q", "F089CASH_PROFIT", "F090CRR", "F091CF_VOL", "F092EARN_STAB", "F093FCF_YIELD", "F094CAPEX_INT", "F095NET_FIN", "F096DILUTION",
            "F097INT_BURDEN", "F098DIV_PAYOUT", "F099AR_MINUS_REV", "F100INV_MINUS_REV"
        ),
        meta_cols=("industry_sw", "list_date", "ret_daily", "1d_next_raw"),

        # ──────────────────────────────────────────────────────────
        # Active models — full benchmark
        # ──────────────────────────────────────────────────────────
        model_names=("lightgbm", "xgboost", "dlinear", "gated_dwtcn",),

        model_feature_signs={},
        # LGBM: 100 factors minus limit-up/down (leaf-wise growth can't handle them)
        model_feature_cols={
            "lightgbm": tuple(
                f for f in (
                    "F001SIZE", "F002SIZENL", "F003LIQUIDITY", "F004BETA", "F005RESVOL",
                    "F006MOMENTUM", "F007LTREV", "F008STREV", "F009LEVERAGE", "F010VALUE",
                    "F011EARNYLD", "F012GROWTH", "F013REV5", "F014MOM120_20", "F015VOLREV",
                    "F016MAXRET", "F017IVOL", "F018AMIHUD", "F019COSTDEV",
                    "F021CFP", "F022GPTA", "F023ACCRUAL", "F024ASSETGR",
                    "F025GAP", "F026KLEN", "F027KUP", "F028KLOW", "F029KSFT",
                    "F030RSV20", "F031RSV60", "F032RANGEZ20", "F033GAPREV5",
                    "F034HIGHDEV20", "F035LOWDEV20", "F036VOLSHOCK5", "F037VOLSHOCK20",
                    "F038TURNZ20", "F039VSTD20", "F040PVCORR20", "F041RETVOLCORR20",
                    "F042AMTCORR20", "F043SLOPE20", "F044RSQR20", "F045RESI20",
                    "F049ROE", "F050ROA", "F051GPM", "F052CFOA",
                    "F053RD_INTENSITY", "F054RECEIVABLE_RATIO", "F055IND",
                    "F056GAP_UP_FAIL", "F057INTRA1", "F058O2O_RET5", "F059GK_VOL20", "F060ON_INTRA_DIV5", "F061GAP_UP_HOLD", "F062GAP_DN_RECOVER",
                    "F063RET5D_SKIP1", "F064RET_ACCEL20", "F065MAXDD20", "F066EFFICIENCY20", "F067TAIL_LOSS20", "F068SKEW20", "F069DNVOL20",
                    "F070UP_DN_VOL", "F071VOL_OF_VOL", "F072CORR_60D", "F073KURT_60D", "F074BETA_20D", "F075VOLUME_RATIO", "F076SIGNED_AMT20",
                    "F077AMP_VOL20", "F078TURN_SIZE", "F079TURN_ACCEL", "F080VWAP_DEV", "F081STRONG_CLOSE", "F082LOCKED_PCT", "F083TURN_FREE",
                    "F084AMT_FREE20", "F085SP_TTM", "F086DIV_TTM", "F087LIST_AGE", "F088CF_SALES_Q", "F089CASH_PROFIT", "F090CRR",
                    "F091CF_VOL", "F092EARN_STAB", "F093FCF_YIELD", "F094CAPEX_INT", "F095NET_FIN", "F096DILUTION", "F097INT_BURDEN",
                    "F098DIV_PAYOUT", "F099AR_MINUS_REV", "F100INV_MINUS_REV"
                ) if f not in ("F046LIMITUP20", "F047LIMITDN20", "F048LIMITSTREAKUP", "F020LIMITUP_RECENCY")
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
                "colsample_bytree":      0.20,
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
