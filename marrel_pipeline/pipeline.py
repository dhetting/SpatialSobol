from __future__ import annotations
import numpy as np
from .wavelet import dwt2_flatten
from .coeff_models import fit_coefficient_models
from .design import maximin_lhs
from .campbell2d import campbell2d_maps_batch, make_spatial_grid

def build_training_data(*, n: int, d: int, low: float, high: float,
                        Z1: np.ndarray, Z2: np.ndarray, seed: int = 12345):
    X = maximin_lhs(n, d, low, high, iters=200, seed=seed)
    Y = campbell2d_maps_batch(X, Z1, Z2, dtype=np.float32)
    return X, Y

def wavelet_decompose_training(Y_train: np.ndarray, wavelet: str, level: int):
    """Compute mu(z) and wavelet coefficients for each training map.
    Returns:
      mu_map: (n1,n2)
      coeffs: (n,K) coefficient vectors in ORIGINAL order
      coeff_slices, coeff_arr_shape: for reconstruction
    """
    mu_map = Y_train.mean(axis=0).astype(np.float32)
    n = Y_train.shape[0]
    # compute coeff metadata from first map
    coeff0, coeff_slices, coeff_arr_shape = dwt2_flatten((Y_train[0] - mu_map), wavelet=wavelet, level=level)
    K = coeff0.size
    coeffs = np.empty((n, K), dtype=np.float64)
    coeffs[0] = coeff0
    for i in range(1, n):
        coeffs[i], _, _ = dwt2_flatten((Y_train[i] - mu_map), wavelet=wavelet, level=level)
    return mu_map, coeffs, coeff_slices, coeff_arr_shape

def order_coefficients_by_variance(coeffs: np.ndarray):
    """Order coefficients by decreasing empirical variance across training sample (paper Eq. (9))."""
    var = np.var(coeffs, axis=0, ddof=1)
    order = np.argsort(-var)  # descending
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)
    coeffs_sorted = coeffs[:, order]
    return coeffs_sorted, order, inv_order, var[order]

def fit_functional_metamodel(X_train: np.ndarray, coeffs_sorted: np.ndarray,
                             *, k_gp: int, k_lin: int, seed: int = 0):
    return fit_coefficient_models(X_train, coeffs_sorted, k_gp=k_gp, k_lin=k_lin, seed=seed)


def frk_decompose_training_maps(Y_train: np.ndarray, Z1: np.ndarray, Z2: np.ndarray,
                                *,
                                levels: int,
                                knots_per_level: tuple[int, ...],
                                radius_factor: float,
                                lambda_pen: float,
                                add_intercept: bool):
    from .frk import frk_decompose_training
    return frk_decompose_training(Y_train, Z1, Z2,
                                 levels=levels,
                                 knots_per_level=knots_per_level,
                                 radius_factor=radius_factor,
                                 lambda_pen=lambda_pen,
                                 add_intercept=add_intercept)

def fit_frk_functional_metamodel(X_train: np.ndarray, eta_sorted: np.ndarray,
                                 *, k_gp: int, k_lin: int, seed: int = 0):
    # Reuse coefficient modeling machinery on FRK coefficients
    return fit_functional_metamodel(X_train, eta_sorted, k_gp=k_gp, k_lin=k_lin, seed=seed)
