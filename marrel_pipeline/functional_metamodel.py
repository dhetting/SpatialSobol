from __future__ import annotations
import numpy as np
from .wavelet import idwt2_unflatten
from .coeff_models import predict_coefficients

def predict_map_from_models(models, X_new: np.ndarray, *,
                            coeff_slices, coeff_arr_shape,
                            wavelet: str,
                            mu_map: np.ndarray,
                            order: np.ndarray):
    """Predict spatial maps using coefficient models + inverse DWT.

    `order` maps sorted coefficient index -> original wavelet coefficient index.
    If `order[j] = k`, then sorted coefficient j corresponds to original coefficient k.
    """
    coeffs_sorted = predict_coefficients(models, X_new)  # (m,K)
    m, K = coeffs_sorted.shape
    maps = []
    for i in range(m):
        coeff_full = np.empty(K, dtype=np.float64)
        coeff_full[order] = coeffs_sorted[i]
        Y_centered = idwt2_unflatten(coeff_full, coeff_slices, coeff_arr_shape, wavelet=wavelet)
        maps.append((mu_map + Y_centered).astype(np.float32, copy=False))
    return np.stack(maps, axis=0)
