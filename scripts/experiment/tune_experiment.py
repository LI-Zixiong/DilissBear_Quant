"""
Grid-search tuning for LightGBM and XGBoost hyperparameters.

Usage:
    python -m scripts.experiment.tune_trees   # tune both models
    python -m scripts.experiment.tune_trees --model lightgbm
    python -m scripts.experiment.tune_trees --model xgboost
"""
import argparse
from copy import deepcopy

from src.experiment.config import ExperimentConfig
from src.experiment.tuning import run_grid_search

# ── Base config from run_experiment.py ────────────────────────────
# Only the model under test is active. All other settings are frozen.
BASE = ExperimentConfig(
    data_path="dataset/processed/factor_panel_1500_54_ind.parquet",
    date_col="time",
    stock_col="stock_id",
    target_col="5d_next_raw",
    return_col="return_1d",
    backtest_return_mode="column",
    backtest_return_source="ret_daily",
    feature_cols=(
        "F001SIZE", "F002SIZENL", "F003LIQUIDITY", "F004BETA", "F005RESVOL",
        "F006MOMENTUM", "F007LTREV", "F008STREV", "F009LEVERAGE", "F010VALUE",
        "F011EARNYLD", "F012GROWTH", "F013REV5", "F014MOM120_20", "F015VOLREV",
        "F016MAXRET", "F017IVOL", "F018AMIHUD", "F019COSTDEV", "F021CFP",
        "F022GPTA", "F023ACCRUAL", "F024ASSETGR", "F025GAP", "F026KLEN", "F027KUP",
        "F028KLOW", "F029KSFT", "F030RSV20", "F031RSV60", "F032RANGEZ20",
        "F033GAPREV5", "F034HIGHDEV20", "F035LOWDEV20", "F036VOLSHOCK5",
        "F037VOLSHOCK20", "F038TURNZ20", "F039VSTD20", "F040PVCORR20",
        "F041RETVOLCORR20", "F042AMTCORR20", "F043SLOPE20", "F044RSQR20",
        "F045RESI20", "F049ROE", "F050ROA", "F051GPM", "F052CFOA",
        "F053RD_INTENSITY", "F054RECEIVABLE_RATIO", 
        "F056GAP_UP_FAIL", "F057INTRA1", "F058O2O_RET5", "F059GK_VOL20",
        "F060ON_INTRA_DIV5", "F061GAP_UP_HOLD", "F062GAP_DN_RECOVER",
        "F063RET5D_SKIP1", "F064RET_ACCEL20", "F065MAXDD20", "F066EFFICIENCY20",
        "F067TAIL_LOSS20", "F068SKEW20", "F069DNVOL20", "F070UP_DN_VOL",
        "F071VOL_OF_VOL", "F072CORR_60D", "F073KURT_60D", "F074BETA_20D",
        "F075VOLUME_RATIO", "F076SIGNED_AMT20", "F077AMP_VOL20", "F078TURN_SIZE",
        "F079TURN_ACCEL", "F080VWAP_DEV", "F081STRONG_CLOSE", "F082LOCKED_PCT",
        "F083TURN_FREE", "F084AMT_FREE20", "F085SP_TTM", "F086DIV_TTM",
        "F087LIST_AGE", "F088CF_SALES_Q", "F089CASH_PROFIT", "F090CRR",
        "F091CF_VOL", "F092EARN_STAB", "F093FCF_YIELD", "F094CAPEX_INT",
        "F095NET_FIN", "F096DILUTION", "F097INT_BURDEN", "F098DIV_PAYOUT",
        "F099AR_MINUS_REV", "F100INV_MINUS_REV"
    ),
    meta_cols=("industry_sw", "list_date", "ret_daily", "1d_next_raw", "limit_status"),
    model_feature_cols={
        "lightgbm": (
                "F001SIZE","F002SIZENL","F003LIQUIDITY","F004BETA","F005RESVOL",
                "F006MOMENTUM","F007LTREV","F008STREV","F009LEVERAGE","F010VALUE",
                "F011EARNYLD","F012GROWTH","F013REV5","F014MOM120_20","F015VOLREV",
                "F016MAXRET","F017IVOL","F018AMIHUD","F019COSTDEV","F021CFP",
                "F022GPTA","F023ACCRUAL","F024ASSETGR","F025GAP","F026KLEN",
                "F027KUP","F028KLOW","F029KSFT","F030RSV20","F031RSV60",
                "F032RANGEZ20","F033GAPREV5","F034HIGHDEV20","F035LOWDEV20","F036VOLSHOCK5",
                "F037VOLSHOCK20","F038TURNZ20","F039VSTD20","F040PVCORR20","F041RETVOLCORR20",
                "F042AMTCORR20","F043SLOPE20","F044RSQR20","F045RESI20","F049ROE",
                "F050ROA","F051GPM","F052CFOA","F053RD_INTENSITY","F054RECEIVABLE_RATIO",
                "F056GAP_UP_FAIL","F057INTRA1","F058O2O_RET5","F059GK_VOL20","F060ON_INTRA_DIV5",
                "F061GAP_UP_HOLD","F062GAP_DN_RECOVER","F063RET5D_SKIP1","F064RET_ACCEL20","F065MAXDD20",
                "F066EFFICIENCY20","F067TAIL_LOSS20","F068SKEW20","F069DNVOL20","F070UP_DN_VOL",
                "F071VOL_OF_VOL","F072CORR_60D","F073KURT_60D","F074BETA_20D","F075VOLUME_RATIO",
                "F076SIGNED_AMT20","F077AMP_VOL20","F078TURN_SIZE","F079TURN_ACCEL","F080VWAP_DEV",
                "F081STRONG_CLOSE","F082LOCKED_PCT","F083TURN_FREE","F084AMT_FREE20","F085SP_TTM",
                "F086DIV_TTM","F087LIST_AGE","F088CF_SALES_Q","F089CASH_PROFIT","F090CRR",
                "F091CF_VOL","F092EARN_STAB","F093FCF_YIELD","F094CAPEX_INT","F095NET_FIN",
                "F096DILUTION","F097INT_BURDEN","F098DIV_PAYOUT","F099AR_MINUS_REV","F100INV_MINUS_REV",
        ),
        "xgboost": (
            "F009LEVERAGE", "F010VALUE", "F011EARNYLD", "F012GROWTH", "F021CFP",
            "F022GPTA", "F023ACCRUAL", "F024ASSETGR", "F049ROE", "F050ROA",
            "F051GPM", "F052CFOA", "F053RD_INTENSITY", "F054RECEIVABLE_RATIO", "F055IND",
            "F085SP_TTM", "F086DIV_TTM", "F088CF_SALES_Q", "F089CASH_PROFIT", "F090CRR",
            "F091CF_VOL", "F092EARN_STAB", "F093FCF_YIELD", "F094CAPEX_INT", "F095NET_FIN",
            "F096DILUTION", "F097INT_BURDEN", "F098DIV_PAYOUT", "F099AR_MINUS_REV", "F100INV_MINUS_REV",
            "F001SIZE", "F002SIZENL", "F003LIQUIDITY",
            "F082LOCKED_PCT", "F083TURN_FREE", "F084AMT_FREE20", "F087LIST_AGE",
        ),
    },
    top_n=50,
    periods_per_year=252,
    model_params={
        "lightgbm": {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 255,
            "early_stopping_rounds": None,
            "feature_fraction": 0.8,
            "subsample": 0.8,
            "min_child_samples": 50,
            "reg_alpha": 1.0,
            "cat_smooth": 10.0,
            "cat_l2": 10.0,
        },
        "xgboost": {
            "n_estimators": 1000,
            "max_depth": 7,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "early_stopping_rounds": None,
        },
    },
)

# ── Parameter grids ────────────────────────────────────────────────
LGBM_GRID = {
    "model_params.lightgbm.max_depth": [10, 14],
}
# 2 trials — max_depth direction

XGBOOST_GRID = {
    "model_params.xgboost.max_depth": [4, 5, 6, 7],
}
# 4 trials — fresh tune on correct 37f fundamental

GATEDDW_GRID = {
    "model_params.gated_dwtcn.gate_rank": [16, 20, 24],
}

GATEDDW_BASE_OVERRIDE = {
    "gated_dwtcn": {
        "seq_len": 20,
        "epochs": 6,
        "lr": 1e-3,
        "wd": 0.0,
        "patience": 10,
        "kernel_size": 3,
        "dilations": (1, 2, 4),
        "hidden_dim": 32,
        "gate_rank": 8,
    },
}

XGBOOST_BASE_OVERRIDE = {
    "xgboost": {
        "n_estimators": 1000,
        "max_depth": 5,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "early_stopping_rounds": None,
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, nargs="+",
                        choices=["lightgbm", "xgboost", "gated_dwtcn"],
                        help="Tune one or more models (default: lightgbm xgboost)")
    args = parser.parse_args()

    models = args.model if args.model else ["lightgbm", "xgboost"]

    for model in models:
        base = deepcopy(BASE)
        base.model_names = (model,)
        base.output_dir = f"dataset/output/tuning/{model}_v1"
        base.report_path = f"reports/tuning/{model}_v1.md"

        if model == "xgboost" and XGBOOST_BASE_OVERRIDE:
            base.model_params.update(XGBOOST_BASE_OVERRIDE)
        elif model == "gated_dwtcn" and GATEDDW_BASE_OVERRIDE:
            base.model_params.update(GATEDDW_BASE_OVERRIDE)

        grids = {
            "lightgbm": LGBM_GRID,
            "xgboost": XGBOOST_GRID,
            "gated_dwtcn": GATEDDW_GRID,
        }
        grid = grids[model]

        run_grid_search(
            base_config=base,
            grid=grid,
            tuning_name=f"{model}_v1",
            sort_by="valid_sharpe_ratio",
            ascending=False,
            continue_on_error=True,
        )
