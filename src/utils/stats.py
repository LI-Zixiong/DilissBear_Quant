"""Statistical utilities: Newey-West HAC standard errors, performance stats."""
import numpy as np


def newey_west_se(
    X: np.ndarray, y: np.ndarray, beta: np.ndarray, maxlags: int = 5
) -> np.ndarray:
    """
    Newey-West heteroskedasticity-and-autocorrelation-consistent standard errors.

    Uses Bartlett kernel with `maxlags` lags.

    Parameters
    ----------
    X : (n, k) design matrix.
    y : (n,) response vector.
    beta : (k,) coefficient estimates.
    maxlags : int, number of autocorrelation lags to include.

    Returns
    -------
    (k,) array of HAC standard errors.
    """
    n, k = X.shape
    resid = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)

    # White HC0 (lag 0)
    V = X.T @ np.diag(resid ** 2) @ X

    # Autocorrelation terms with Bartlett kernel
    for lag in range(1, maxlags + 1):
        w = 1.0 - lag / (maxlags + 1.0)
        G = X[lag:].T @ np.diag(resid[lag:] * resid[:-lag]) @ X[:-lag]
        V += w * (G + G.T)

    cov = XtX_inv @ V @ XtX_inv
    se = np.sqrt(np.diag(cov))
    # Guard against zero or negative SE
    se = np.maximum(se, 1e-12)
    return se
