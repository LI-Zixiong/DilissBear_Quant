"""
Experiment configuration.

This module defines the public configuration object used by experiment runners,
ensemble scripts, scan scripts, and strategy evaluation scripts.

Design rules:
    1. config.py defines "what to run".
    2. It should not import models, trainers, data loaders, predictors, or backtest code.
    3. The first refactor should preserve the default behavior of scripts/run_experiment.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class ExperimentConfig:
    """
    Configuration for one end-to-end experiment.

    Current project convention:
        - The full benchmark model pool contains five models:
          LightGBM, XGBoost, DLinear, iTransformer, TSMixer.
        - The current default run activates only LightGBM + DLinear
          for speed and stability.
        - PatchTST is implemented in the codebase but treated as experimental
          for now.

    This class intentionally keeps all default paths and column names aligned
    with the current experiment_003 setup.
    """

    # ------------------------------------------------------------------
    # Input data
    # ------------------------------------------------------------------
    data_path: str = "dataset/processed/factor_panel_1500_54_ind.parquet"

    date_col: str = "time"
    stock_col: str = "stock_id"

    # Training target.
    # The model learns this column by default.
    target_col: str = "5d_next_raw"

    # Backtest return settings.
    #
    # backtest_return_mode:
    #   "column"      -> use an existing realized return column, e.g. ret_daily
    #   "next_target" -> map a next-period target, e.g. 1d_next_raw, to the next date
    #
    # Default: 1d_next_raw is open-to-open (t+1 open buy, t+2 open sell),
    # consistent with model training targets and Bayes H.
    return_col: str = "return_1d"
    backtest_return_mode: str = "next_target"
    backtest_return_source: str = "1d_next_raw"

    # ------------------------------------------------------------------
    # Features and metadata
    # ------------------------------------------------------------------
    # Current selected 12-factor set:
    #   4 old / Barra-style useful factors + 8 newer alpha factors.
    #
    # Note:
    #   Some negative-ICIR factors are already sign-flipped upstream,
    #   so the model can treat larger values as better signals.
    # Selected 12-factor set (F-format standard names).
    # Sign-flipped factors already have larger = better.
    # Full 54-factor V1 set (2026-06-01).
    # Use audit_factors.py to select subsets per model.
    # 87-factor set from audit_v2 (icir>=0.02, corr<=0.98, 2023-12-31 cutoff).
    # Audit kept 87/99 continuous factors.  Old factor names use padded format;
    # new F056GAP_UP_FAIL-F100INV_MINUS_REV use plain numeric format matching the factor panel columns.
    feature_cols: Sequence[str] = (
        "F046LIMITUP20",
        "F077AMP_VOL20",
        "F041RETVOLCORR20",
        "F016MAXRET",
        "F083TURN_FREE",
        "F039VSTD20",
        "F078TURN_SIZE",
        "F018AMIHUD",
        "F072CORR_60D",
        "F001SIZE",
        "F017IVOL",
        "F065MAXDD20",
        "F040PVCORR20",
        "F071VOL_OF_VOL",
        "F028KLOW",
        "F070UP_DN_VOL",
        "F059GK_VOL20",
        "F084AMT_FREE20",
        "F060ON_INTRA_DIV5",
        "F026KLEN",
        "F035LOWDEV20",
        "F008STREV",
        "F045RESI20",
        "F081STRONG_CLOSE",
        "F019COSTDEV",
        "F042AMTCORR20",
        "F025GAP",
        "F068SKEW20",
        "F005RESVOL",
        "F080VWAP_DEV",
        "F003LIQUIDITY",
        "F076SIGNED_AMT20",
        "F027KUP",
        "F031RSV60",
        "F090CRR",
        "F057INTRA1",
        "F086DIV_TTM",
        "F064RET_ACCEL20",
        "F033GAPREV5",
        "F062GAP_DN_RECOVER",
        "F029KSFT",
        "F010VALUE",
        "F021CFP",
        "F013REV5",
        "F085SP_TTM",
        "F007LTREV",
        "F069DNVOL20",
        "F038TURNZ20",
        "F011EARNYLD",
        "F037VOLSHOCK20",
        "F063RET5D_SKIP1",
        "F023ACCRUAL",
        "F061GAP_UP_HOLD",
        "F058O2O_RET5",
        "F048LIMITSTREAKUP",
        "F030RSV20",
        "F079TURN_ACCEL",
        "F015VOLREV",
        "F052CFOA",
        "F002SIZENL",
        "F082LOCKED_PCT",
        "F088CF_SALES_Q",
        "F004BETA",
        "F096DILUTION",
        "F047LIMITDN20",
        "F012GROWTH",
        "F066EFFICIENCY20",
        "F024ASSETGR",
        "F095NET_FIN",
        "F075VOLUME_RATIO",
        "F036VOLSHOCK5",
        "F099AR_MINUS_REV",
        "F032RANGEZ20",
        "F089CASH_PROFIT",
        "F055IND",
        "F087LIST_AGE",
        "F097INT_BURDEN",
        "F074BETA_20D",
        "F022GPTA",
        "F044RSQR20",
        "F054RECEIVABLE_RATIO",
        "F091CF_VOL",
        "F053RD_INTENSITY",
        "F006MOMENTUM",
        "F009LEVERAGE",
        "F050ROA",
        "F094CAPEX_INT",
    )

    meta_cols: Sequence[str] = (
        "industry_sw",
        "list_date",
        "ret_daily",
        "1d_next_raw",
    )

    # Per-model feature overrides. None = all models use feature_cols.
    # Example: {"lightgbm": ("F016MAXRET","F017IVOL",), "dlinear": ("F001SIZE",)}
    model_feature_cols: dict[str, tuple[str, ...]] | None = None

    # Per-model sign flips (applied before training). Does NOT modify factor panel.
    # Example: {"dlinear": {"F001SIZE": -1, "F003LIQUIDITY": -1}}
    model_feature_signs: dict[str, dict[str, int]] | None = None

    # ------------------------------------------------------------------
    # Model pool
    # ------------------------------------------------------------------
    # Full model pool used by the main project benchmark.
    supported_model_names: Sequence[str] = (
        "lightgbm",
        "xgboost",
        "dlinear",
        "gated_dwtcn",
        "itransformer",
        "tsmixer",
    )

    # Implemented but not part of the current main five-model benchmark.
    experimental_model_names: Sequence[str] = (
        "patchtst",
    )

    # Models activated in the current run.
    #
    # Default:
    #   Only LightGBM + DLinear are active for speed, stability, and cleaner
    #   comparison while the data/factor pipeline is still being refined.
    #
    # To restore the full five-model benchmark, set:
    #   model_names = (
    #       "lightgbm", "xgboost", "dlinear", "itransformer", "tsmixer"
    #   )
    model_names: Sequence[str] = (
        "lightgbm",
        "dlinear",
    )

    # Explicit date boundaries (half-open [start,end)). Overrides split_ratio.
    # Format: (train_start, train_end, valid_start, valid_end, test_start, test_end)
    date_boundaries: tuple[str, str, str, str, str, str] | None = None
    purge_days: int = 6
    target_horizon_days: int = 5

    # These are filled by the date split function during the run.
    train_end: str = ""
    valid_end: str = ""

    # ------------------------------------------------------------------
    # General experiment settings
    # ------------------------------------------------------------------
    # Date split ratio: (train, valid, test). Must sum to 1.0.
    split_ratio: Sequence[float] = (0.6, 0.2, 0.2)

    seed: int = 140503
    top_n: int = 50
    periods_per_year: int = 252

    # ------------------------------------------------------------------
    # Default torch training settings
    # ------------------------------------------------------------------
    # These are fallback defaults. Per-model overrides are stored in
    # model_params below.
    seq_len: int = 20
    torch_epochs: int = 10
    torch_patience: int = 1
    torch_batch_size: int = 4096
    torch_learning_rate: float = 7.5e-4
    torch_weight_decay: float = 0.0
    torch_device: str = "auto"

    # Per-model training overrides.
    #
    # Later, src/experiment/model_factory.py will provide:
    #   get_model_params(model_name, config)
    #
    # That function should merge:
    #   default torch settings + model-specific overrides.
    model_params: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        # Tree model parameters (used by model_factory.build_experiment_model)
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
        # Torch model parameters (override torch defaults above)
        "dlinear": {
            "seq_len": 15,
            "epochs": 15,
            "lr": 7.5e-4,
            "wd": 0.0,
            "patience": 2,
        },
        "gated_dwtcn": {
            "seq_len": 20,
            "epochs": 20,
            "lr": 2e-4,
            "wd": 0.0,
            "patience": 3,
            "kernel_size": 3,
            "dilations": (1, 2, 4),
            "hidden_dim": 32,
            "gate_rank": 8,
            "dropout": 0.1,
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
    })

    predict_batch_size: int = 8192

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    output_dir: str = "dataset/output/experiment_005"
    report_path: str = "reports/experiment_005.md"

    # ------------------------------------------------------------------
    # Lightweight validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Validate obvious configuration mistakes early."""

        allowed_models = set(self.supported_model_names) | set(self.experimental_model_names)
        unknown_models = [m for m in self.model_names if m not in allowed_models]
        if unknown_models:
            raise ValueError(
                "Unknown model name(s): "
                f"{unknown_models}. Allowed models are: {sorted(allowed_models)}"
            )

        duplicated_features = _find_duplicates(self.feature_cols)
        if duplicated_features:
            raise ValueError(f"Duplicated feature columns: {duplicated_features}")

        if self.top_n <= 0:
            raise ValueError(f"top_n must be positive, got {self.top_n}")

        if self.periods_per_year <= 0:
            raise ValueError(
                f"periods_per_year must be positive, got {self.periods_per_year}"
            )

        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")

        if self.torch_batch_size <= 0:
            raise ValueError(
                f"torch_batch_size must be positive, got {self.torch_batch_size}"
            )

        if self.predict_batch_size <= 0:
            raise ValueError(
                f"predict_batch_size must be positive, got {self.predict_batch_size}"
            )
        
        if self.backtest_return_mode not in {"column", "next_target"}:
            raise ValueError(
                "backtest_return_mode must be one of {'column', 'next_target'}, "
                f"got {self.backtest_return_mode}"
            )

        if len(self.split_ratio) != 3:
            raise ValueError(
                f"split_ratio must have three values: train, valid, test. "
                f"Got {self.split_ratio}"
            )

        if any(x <= 0 for x in self.split_ratio):
            raise ValueError(
                f"All split_ratio values must be positive. Got {self.split_ratio}"
            )

        ratio_sum = float(sum(self.split_ratio))
        if abs(ratio_sum - 1.0) > 1e-8:
            raise ValueError(
                f"split_ratio must sum to 1.0. Got {self.split_ratio}, sum={ratio_sum}"
            )

        if self.date_boundaries is not None:
            if len(self.date_boundaries) != 6:
                raise ValueError(
                    f"date_boundaries must have exactly 6 values, "
                    f"got {len(self.date_boundaries)}"
                )
        if self.purge_days < 0:
            raise ValueError(f"purge_days must be >= 0, got {self.purge_days}")
        if self.target_horizon_days < 0:
            raise ValueError(
                f"target_horizon_days must be >= 0, got {self.target_horizon_days}"
            )

    @property
    def active_model_set(self) -> set[str]:
        """Return active model names as a set."""

        return set(self.model_names)

    @property
    def tabular_model_names(self) -> tuple[str, ...]:
        """
        Return active tabular models.

        This helper is intentionally lightweight. The authoritative model-family
        logic will later live in src/experiment/model_factory.py.
        """

        return tuple(
            m for m in self.model_names
            if m in {"lightgbm", "xgboost"}
        )

    @property
    def torch_model_names(self) -> tuple[str, ...]:
        """
        Return active torch sequence models.

        This helper is intentionally lightweight. The authoritative model-family
        logic will later live in src/experiment/model_factory.py.
        """

        return tuple(
            m for m in self.model_names
            if m in {"dlinear", "gated_dwtcn", "itransformer", "tsmixer", "patchtst"}
        )


def _find_duplicates(values: Sequence[str]) -> list[str]:
    """Return duplicate strings while preserving first duplicate encounter order."""

    seen: set[str] = set()
    duplicates: list[str] = []

    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)

    return duplicates