from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from xgboost import XGBRegressor

@dataclass
class XGBoostConfig:
    """
    Configuration for XGBoost model.
    
    This model is designed for tabular factor features.

    It predicts future stock returns or style-adjusted returns.

    @JulongQuant
    """

    # Model parameters
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int | None = None
    random_state: int = 42
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")

        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

        if not 0.0 < self.subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1]")

        if not 0.0 < self.colsample_bytree <= 1.0:
            raise ValueError("colsample_bytree must be in (0, 1]")

        if self.reg_alpha < 0:
            raise ValueError("reg_alpha must be >= 0")

        if self.reg_lambda < 0:
            raise ValueError("reg_lambda must be >= 0")

class XGBoostReturnRegressor:
    """
    XGBoost wrapper for stock return regression.

    Input:
        X: 2D numpy array or pandas DataFrame with shape [n_samples, n_features]

    Target:
        y: 1D numpy array or pandas Series with shape [n_samples]

    Output:
        prediction: 1D numpy array with shape [n_samples]
    """

    def __init__(self, config: XGBoostConfig) -> None:
        self.config = config
        self._is_fitted = False
        kwargs = {}
        if config.early_stopping_rounds is not None:
            kwargs["early_stopping_rounds"] = config.early_stopping_rounds
        self.model = XGBRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            objective='reg:squarederror',
            **kwargs,
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
        Train the XGBoost model.
        """

        X_train = self._to_numpy_X(X_train)
        y_train = self._to_1d_y(y_train)
        self._check_X_y(X_train, y_train)

        if X_valid is not None and y_valid is not None:
            X_valid = self._to_numpy_X(X_valid)
            y_valid = self._to_1d_y(y_valid)
            self._check_X_y(X_valid, y_valid)
        
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
            )
        else:
            self.model.fit(
                X_train, 
                y_train, 
                verbose=False
            )
        
        self._is_fitted = True

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict using the trained XGBoost model.

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
        
def build_xgboost_model(config: XGBoostConfig) -> XGBoostReturnRegressor:
    """
    Helper function for building the XGBoost model from config.
    """

    return XGBoostReturnRegressor(config)

if __name__ == "__main__":
    # Quick smoke test.
    config = XGBoostConfig()
    model = build_xgboost_model(config)

    # Mock data
    X_train = np.random.rand(1000, 36)  # 1000 samples, 36 features
    y_train = np.random.rand(1000)      # 1000 target values

    model.fit(X_train, y_train)
    preds = model.predict(X_train)

    print(preds)