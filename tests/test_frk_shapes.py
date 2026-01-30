import numpy as np
from marrel_pipeline.config import DEFAULTS
from marrel_pipeline.campbell2d import make_spatial_grid
from marrel_pipeline.pipeline import build_training_data, frk_decompose_training_maps, order_coefficients_by_variance
from marrel_pipeline.functional_frk_metamodel import predict_map_from_frk_models
from marrel_pipeline.coeff_models import fit_coefficient_models

def test_frk_shapes_and_reconstruction():
    cfg = DEFAULTS
    Z1, Z2 = make_spatial_grid(cfg.grid_n1, cfg.grid_n2, cfg.z_low, cfg.z_high)
    Xtr, Ytr = build_training_data(n=50, d=cfg.d, low=cfg.x_low, high=cfg.x_high, Z1=Z1, Z2=Z2, seed=123)
    mu_map, eta, basis = frk_decompose_training_maps(
        Ytr, Z1, Z2,
        levels=cfg.frk_levels,
        knots_per_level=cfg.frk_knots_per_level,
        radius_factor=cfg.frk_radius_factor,
        lambda_pen=cfg.frk_lambda_pen,
        add_intercept=cfg.frk_add_intercept
    )
    assert eta.shape[0] == 50
    assert basis.B.shape[0] == cfg.grid_n1*cfg.grid_n2

    eta_sorted, order, inv_order, _ = order_coefficients_by_variance(eta)
    k_lin = min(cfg.k_lin, eta_sorted.shape[1] - cfg.k_gp)
    models = fit_coefficient_models(Xtr, eta_sorted, k_gp=cfg.k_gp, k_lin=k_lin, seed=0)
    Ypred = predict_map_from_frk_models(models, Xtr[:5], mu_map=mu_map, basis=basis, order=order)
    assert Ypred.shape == (5, cfg.grid_n1, cfg.grid_n2)
