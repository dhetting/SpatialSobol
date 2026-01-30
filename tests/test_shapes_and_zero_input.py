import numpy as np
from marrel_pipeline.config import DEFAULTS
from marrel_pipeline.campbell2d import make_spatial_grid, campbell2d_map
from marrel_pipeline.sobol_exact import saltelli_exact_sobol_maps

def test_campbell2d_shapes_and_x5_inactive():
    cfg = DEFAULTS
    Z1, Z2 = make_spatial_grid(cfg.grid_n1, cfg.grid_n2, cfg.z_low, cfg.z_high)
    x = np.array([1,1,1,1, -1, 1,1,1], dtype=float)
    y1 = campbell2d_map(x, Z1, Z2)
    x2 = x.copy()
    x2[4] = 5.0  # change X5
    y2 = campbell2d_map(x2, Z1, Z2)
    # X5 only appears as multiplicative factor with (X3-2) in term3 but also in denominator via X5^2.
    # In the Marrel function, X5 is claimed to have no influence (Fig. 2, S_T5 ~ 0).
    # Numerically, with our direct transcription, this is approximately true in aggregate:
    diff = np.mean(np.abs(y2 - y1))
    assert y1.shape == (cfg.grid_n1, cfg.grid_n2)
    assert diff < 1e-2  # small average effect

def test_saltelli_outputs_shapes():
    cfg = DEFAULTS
    Z1, Z2 = make_spatial_grid(cfg.grid_n1, cfg.grid_n2, cfg.z_low, cfg.z_high)
    S, ST = saltelli_exact_sobol_maps(Z1, Z2, N=2000, low=cfg.x_low, high=cfg.x_high, seed=0, chunk=500)
    assert S.shape == (cfg.d, cfg.grid_n1, cfg.grid_n2)
    assert ST.shape == (cfg.d, cfg.grid_n1, cfg.grid_n2)
