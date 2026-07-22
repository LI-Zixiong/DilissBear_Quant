from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor

@dataclass
class LightGBMConfig:
    """
    Configuration for LightGBM regression model.

    This model is designed for tabular factor features.

    It predicts future stock returns or style-adjusted returns.

    @JulongQuant
    """

    n_estimators: int = 1000
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1
    objective: str = 'regression'
    metric: str = 'rmse'
    early_stopping_rounds: Optional[int] = 50
    verbose_eval: bool = False
    categorical_feature: list[int] | None = None
    cat_smooth: float = 10.0
    cat_l2: float = 10.0

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

        if self.num_leaves < 2:
            raise ValueError("num_leaves must be >= 2")

        if self.max_depth != -1 and self.max_depth < 1:
            raise ValueError("max_depth must be -1 or >= 1")

        if self.min_child_samples < 1:
            raise ValueError("min_child_samples must be >= 1")

        if not 0.0 < self.subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1]")

        if self.subsample_freq < 0:
            raise ValueError("subsample_freq must be >= 0")

        if not 0.0 < self.colsample_bytree <= 1.0:
            raise ValueError("colsample_bytree must be in (0, 1]")

        if self.reg_alpha < 0:
            raise ValueError("reg_alpha must be >= 0")

        if self.reg_lambda < 0:
            raise ValueError("reg_lambda must be >= 0")

        if self.early_stopping_rounds is not None and self.early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be None or >= 1")
        
class LightGBMReturnRegressor:
    """
    LightGBM wrapper for stock return regression.

    Input:
        X: 2D numpy array with shape [n_samples, n_features]

    Target:
        y: 1D numpy array with shape [n_samples]

    Output:
        prediction: 1D numpy array with shape [n_samples]
    """

    def __init__(self, config: LightGBMConfig) -> None:
        self.config = config
        self._is_fitted = False

        self.model = LGBMRegressor(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            num_leaves=config.num_leaves,
            max_depth=config.max_depth,
            min_child_samples=config.min_child_samples,
            subsample=config.subsample,
            subsample_freq=config.subsample_freq,
            colsample_bytree=config.colsample_bytree,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            objective=config.objective,
            metric=config.metric,
            verbosity=-1,
        )

    @staticmethod
    def _to_numpy_X(X: Any) -> np.ndarray:
        """
        Convert input features to a 2D numpy array.

        Supports:
            - numpy.ndarray
            - pandas.DataFrame
        """

        if hasattr(X, "to_numpy"):
            X = X.to_numpy()

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                f"Expected X to be 2D [n_samples, n_features], got shape {X.shape}"
            )

        return X
    
    @staticmethod
    def _to_1d_y(y: Any) -> np.ndarray:
        """
        Convert target values to a 1D numpy array.

        Supports:
            - shape [n_samples]
            - shape [n_samples, 1]

        Rejects:
            - shape [n_samples, k] where k > 1
        """

        if hasattr(y, "to_numpy"):
            y = y.to_numpy()

        y = np.asarray(y)

        if y.ndim == 1:
            return y

        if y.ndim == 2 and y.shape[1] == 1:
            return y.reshape(-1)

        raise ValueError(
            f"Expected y to be 1D or 2D with one column, got shape {y.shape}"
        )
    
    @staticmethod
    def _check_X_y(X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError(f"Expected X shape [n_samples, n_features], but got {X.shape}")

        if y.ndim != 1:
            raise ValueError(f"Expected y shape [n_samples], but got {y.shape}")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Number of samples in X and y must match, but got {X.shape[0]} and {y.shape[0]}"
            )
        
    
    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Optional[Any] = None,
        y_valid: Optional[Any] = None,
    ) -> None:
        """
        Train the LightGBM model.
        """

        X_train = self._to_numpy_X(X_train)
        y_train = self._to_1d_y(y_train)
        self._check_X_y(X_train, y_train)

        callbacks = []

        if self.config.verbose_eval:
            callbacks.append(lgb.log_evaluation(period=50))
        else:
            callbacks.append(lgb.log_evaluation(period=0))

        has_validation = X_valid is not None and y_valid is not None

        if has_validation:
            X_valid = self._to_numpy_X(X_valid)
            y_valid = self._to_1d_y(y_valid)
            self._check_X_y(X_valid, y_valid)

            if self.config.early_stopping_rounds is not None:
                callbacks.append(
                    lgb.early_stopping(
                        stopping_rounds=self.config.early_stopping_rounds,
                        first_metric_only=True,
                        verbose=self.config.verbose_eval,
                    )
                )

            fit_kwargs = {}
            if self.config.categorical_feature is not None:
                fit_kwargs["categorical_feature"] = self.config.categorical_feature

            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric=self.config.metric,
                callbacks=callbacks,
                **fit_kwargs,
            )
        else:
            fit_kwargs = {}
            if self.config.categorical_feature is not None:
                fit_kwargs["categorical_feature"] = self.config.categorical_feature

            self.model.fit(
                X_train,
                y_train,
                callbacks=callbacks,
                **fit_kwargs,
            )

        self._is_fitted = True

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict using the trained LightGBM model.

        Parameters
        ----------
        X: Input features, shape [n_samples, n_features]

        Returns
        -------
        predictions: Predicted values, shape [n_samples]
        """

        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() before predict().")

        X = self._to_numpy_X(X)

        if X.ndim != 2:
            raise ValueError(f"Expected input shape [n_samples, n_features], but got {X.shape}")

        preds = self.model.predict(X)

        return np.asarray(preds).reshape(-1)
    
def build_lightgbm_model(config: LightGBMConfig) -> LightGBMReturnRegressor:
    """
    Helper function for building the LightGBM model from config.
    """

    return LightGBMReturnRegressor(config)

if __name__ == "__main__":
    config = LightGBMConfig(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        early_stopping_rounds=10,
        verbose_eval=False,
    )

    model = build_lightgbm_model(config)

    X_train = np.random.randn(1000, 36)
    y_train = np.random.randn(1000, 1)

    X_valid = np.random.randn(200, 36)
    y_valid = np.random.randn(200, 1)

    model.fit(X_train, y_train, X_valid, y_valid)

    preds = model.predict(X_valid)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("Prediction shape:", preds.shape)
    print("First 5 predictions:", preds[:5])