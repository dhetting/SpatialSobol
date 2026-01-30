from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression

@dataclass
class CoefficientModel:
    kind: str  # 'gp' | 'lin' | 'mean'
    model: object
    mean_: float

def _fit_gp(X: np.ndarray, y: np.ndarray, seed: int = 0) -> GaussianProcessRegressor:
    # Kernel approximates the "generalized exponential" behavior; Matern is robust.
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(X.shape[1]),
                                                       length_scale_bounds=(1e-2, 1e2),
                                                       nu=1.5) + WhiteKernel(noise_level=1e-6,
                                                                             noise_level_bounds=(1e-10, 1e-2))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed, n_restarts_optimizer=2)
    gp.fit(X, y)
    return gp

def _fit_linear_aic_subset(X: np.ndarray, y: np.ndarray):
    """Exhaustive subset selection by AIC for d<=12 (here d=8)."""
    n, d = X.shape
    # Always include intercept via LinearRegression(fit_intercept=True)
    best_aic = np.inf
    best_mask = None
    # Precompute residual variance for each subset.
    for mask_int in range(1, 1<<d):  # exclude empty -> mean model handled separately
        cols = [(mask_int>>j)&1 for j in range(d)]
        cols_idx = [j for j,c in enumerate(cols) if c]
        Xsub = X[:, cols_idx]
        lr = LinearRegression(fit_intercept=True)
        lr.fit(Xsub, y)
        yhat = lr.predict(Xsub)
        rss = np.sum((y - yhat)**2)
        k_params = 1 + len(cols_idx)  # intercept + betas
        # Gaussian AIC for linear regression
        sigma2 = rss / n
        if sigma2 <= 0:
            continue
        logL = -0.5*n*(np.log(2*np.pi*sigma2) + 1.0)
        aic = 2*k_params - 2*logL
        if aic < best_aic:
            best_aic = aic
            best_mask = cols_idx
            best_lr = lr
    return best_lr, best_mask

def fit_coefficient_models(X_train: np.ndarray,
                           coeffs_train: np.ndarray,
                           *,
                           k_gp: int,
                           k_lin: int,
                           seed: int = 0):
    """Fit models for wavelet coefficients according to Marrel et al. (2010) Method 3.

    Inputs:
      X_train: (n,d)
      coeffs_train: (n,K) flattened coefficient vectors, already ordered by decreasing variance.
    Returns:
      models: list[CoefficientModel] length K
    """
    n, K = coeffs_train.shape
    models = []
    # GP for first k_gp
    for j in range(k_gp):
        y = coeffs_train[:, j]
        gp = _fit_gp(X_train, y, seed=seed + j)
        models.append(CoefficientModel(kind="gp", model=gp, mean_=float(y.mean())))
    # Linear for next k_lin
    for j in range(k_gp, min(k_gp + k_lin, K)):
        y = coeffs_train[:, j]
        lr, cols = _fit_linear_aic_subset(X_train, y)
        models.append(CoefficientModel(kind="lin", model=(lr, cols), mean_=float(y.mean())))
    # Mean for the rest
    for j in range(min(k_gp + k_lin, K), K):
        y = coeffs_train[:, j]
        models.append(CoefficientModel(kind="mean", model=None, mean_=float(y.mean())))
    return models

def predict_coefficients(models, X_new: np.ndarray) -> np.ndarray:
    """Predict coefficient vectors for X_new using fitted models.

    X_new: (m,d)
    Returns: (m,K)
    """
    m = X_new.shape[0]
    K = len(models)
    out = np.empty((m, K), dtype=np.float64)
    for j, cm in enumerate(models):
        if cm.kind == "gp":
            out[:, j] = cm.model.predict(X_new)
        elif cm.kind == "lin":
            lr, cols = cm.model
            out[:, j] = lr.predict(X_new[:, cols])
        else:
            out[:, j] = cm.mean_
    return out
