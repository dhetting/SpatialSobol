import numpy as np
import pytest

from marrel_pipeline.config import DEFAULTS
from marrel_pipeline.campbell2d import make_spatial_grid
from marrel_pipeline.pipeline import build_training_data, wavelet_decompose_training, order_coefficients_by_variance, fit_functional_metamodel
from marrel_pipeline.functional_metamodel import predict_map_from_models
from marrel_pipeline.diagnostics import rmae
from marrel_pipeline.sobol_exact import saltelli_exact_sobol_maps
from marrel_pipeline.sobol_metamodel import nested_mc_sobol_maps

@pytest.mark.slow
def test_table1_rmae_reasonable():
    cfg = DEFAULTS
    Z1, Z2 = make_spatial_grid(cfg.grid_n1, cfg.grid_n2, cfg.z_low, cfg.z_high)

    # Training data (n=200)
    Xtr, Ytr = build_training_data(n=cfg.n_train, d=cfg.d, low=cfg.x_low, high=cfg.x_high, Z1=Z1, Z2=Z2, seed=12345)
    mu_map, coeffs, coeff_slices, coeff_arr_shape = wavelet_decompose_training(Ytr, cfg.wavelet, cfg.wavelet_level)
    coeffs_sorted, order, inv_order, _ = order_coefficients_by_variance(coeffs)

    models = fit_functional_metamodel(Xtr, coeffs_sorted, k_gp=cfg.k_gp, k_lin=cfg.k_lin, seed=0)

    # "Exact" maps via Saltelli on true simulator (reduced N in unit test for speed)
    S_true, _ = saltelli_exact_sobol_maps(Z1, Z2, N=20_000, low=cfg.x_low, high=cfg.x_high, seed=2020, chunk=2000)

    # Metamodel Sobol maps (reduced counts for unit test)
    S_hat, _ = nested_mc_sobol_maps(models,
        Z1=Z1, Z2=Z2,
        coeff_slices=coeff_slices, coeff_arr_shape=coeff_arr_shape,
        wavelet=cfg.wavelet, mu_map=mu_map, order=order,
        low=cfg.x_low, high=cfg.x_high,
        mc_outer_first=80, mc_inner_first=200,
        mc_outer_total=80, mc_inner_total=200,
        mc_varY=4000,
        seed=7, chunk=2000
    )

    # Compute rMAE(Si) per variable
    rmae_vals = []
    for i in range(cfg.d):
        val = rmae(S_hat[i], S_true[i])
        rmae_vals.append(val)
    rmae_vals = np.array(rmae_vals)

    # Paper Table 1 (percent): X1 8.75, X2 16.25, X3 16.35, X4 12.8, X5 --, X6 13.17, X7 11.80, X8 9.96
    table1 = np.array([0.0875, 0.1625, 0.1635, 0.1280, np.nan, 0.1317, 0.1180, 0.0996])

    # We do not enforce X5 (true is identically 0)
    mask = ~np.isnan(table1)

    # Allow broad tolerance because our GP implementation differs from theirs;
    # the goal is to ensure the pipeline is in the right ballpark.
    assert np.all(np.abs(rmae_vals[mask] - table1[mask]) < 0.12)
