from __future__ import annotations
import numpy as np
from .coeff_models import predict_coefficients
from .frk import frk_reconstruct_maps
from .frk_basis import FRKBasis

def predict_map_from_frk_models(models,
                                X_new: np.ndarray,
                                *,
                                mu_map: np.ndarray,
                                basis: FRKBasis,
                                order: np.ndarray):
    """Predict spatial maps using FRK coefficient models + basis reconstruction.

    `order` maps sorted coefficient index -> original coefficient index.
    """
    eta_sorted = predict_coefficients(models, X_new)  # (m,r)
    m, r = eta_sorted.shape
    eta_full = np.empty((m, r), dtype=np.float64)
    eta_full[:, order] = eta_sorted
    return frk_reconstruct_maps(mu_map, eta_full, basis)
