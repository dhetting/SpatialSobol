from __future__ import annotations
import numpy as np
from .functional_metamodel import predict_map_from_models

def _sample_uniform(n: int, d: int, low: float, high: float, rng: np.random.Generator):
    return low + (high - low) * rng.random((n, d))

def nested_mc_sobol_maps(models,
                         *,
                         Z1: np.ndarray, Z2: np.ndarray,
                         coeff_slices, coeff_arr_shape, wavelet: str,
                         mu_map: np.ndarray, order: np.ndarray,
                         low: float, high: float,
                         mc_outer_first: int, mc_inner_first: int,
                         mc_outer_total: int, mc_inner_total: int,
                         mc_varY: int,
                         seed: int = 7,
                         chunk: int = 2000):
    """Compute Sobol maps from the functional metamodel using Marrel et al. (2010) nested-MC scheme (Sec. 3.4).

    Returns:
      S_hat: (d,n1,n2)
      ST_hat: (d,n1,n2)
    """
    rng = np.random.default_rng(seed)
    d = 8
    n1, n2 = Z1.shape
    P = n1*n2

    # Estimate Var(Y) using mc_varY simulations of the metamodel
    Xv = _sample_uniform(mc_varY, d, low, high, rng)
    Yv = predict_map_from_models(models, Xv, coeff_slices=coeff_slices, coeff_arr_shape=coeff_arr_shape,
                                 wavelet=wavelet, mu_map=mu_map, order=order).reshape(-1, P).astype(np.float64)
    varY = np.var(Yv, axis=0, ddof=1)
    varY = np.maximum(varY, 1e-12)

    S = np.zeros((d, P), dtype=np.float64)
    ST = np.zeros((d, P), dtype=np.float64)

    # First-order indices
    for i in range(d):
        # Outer samples for Xi
        Xi = low + (high - low) * rng.random((mc_outer_first, 1))
        # For each Xi, sample inner X_-i and compute conditional mean
        cond_means = np.zeros((mc_outer_first, P), dtype=np.float64)
        for start in range(0, mc_outer_first, max(1, chunk//mc_inner_first)):
            end = min(mc_outer_first, start + max(1, chunk//mc_inner_first))
            m = end - start
            # Build all combos in a block: m * mc_inner_first rows
            Xblock = _sample_uniform(m*mc_inner_first, d, low, high, rng)
            # set Xi for each group
            for g in range(m):
                Xblock[g*mc_inner_first:(g+1)*mc_inner_first, i] = Xi[start+g, 0]
            Yblock = predict_map_from_models(models, Xblock,
                                             coeff_slices=coeff_slices, coeff_arr_shape=coeff_arr_shape,
                                             wavelet=wavelet, mu_map=mu_map, order=order).reshape(-1, P).astype(np.float64)
            # average within each group
            for g in range(m):
                cond_means[start+g] = Yblock[g*mc_inner_first:(g+1)*mc_inner_first].mean(axis=0)
        S[i] = np.var(cond_means, axis=0, ddof=1) / varY

    # Total-effect indices via Var(E[Y|X_-i]) and ST = 1 - Var(E[Y|X_-i]) / Var(Y)
    for i in range(d):
        Xminus = _sample_uniform(mc_outer_total, d-1, low, high, rng)
        cond_means = np.zeros((mc_outer_total, P), dtype=np.float64)
        for start in range(0, mc_outer_total, max(1, chunk//mc_inner_total)):
            end = min(mc_outer_total, start + max(1, chunk//mc_inner_total))
            m = end - start
            Xblock_full = _sample_uniform(m*mc_inner_total, d, low, high, rng)
            # fill X_-i columns for each group
            cols = [j for j in range(d) if j != i]
            for g in range(m):
                for jj, col in enumerate(cols):
                    Xblock_full[g*mc_inner_total:(g+1)*mc_inner_total, col] = Xminus[start+g, jj]
            # Xi already random in Xblock_full -> integrates over Xi
            Yblock = predict_map_from_models(models, Xblock_full,
                                             coeff_slices=coeff_slices, coeff_arr_shape=coeff_arr_shape,
                                             wavelet=wavelet, mu_map=mu_map, order=order).reshape(-1, P).astype(np.float64)
            for g in range(m):
                cond_means[start+g] = Yblock[g*mc_inner_total:(g+1)*mc_inner_total].mean(axis=0)
        ST[i] = 1.0 - (np.var(cond_means, axis=0, ddof=1) / varY)

    return S.reshape(d, n1, n2), ST.reshape(d, n1, n2)
