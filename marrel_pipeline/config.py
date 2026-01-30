from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class MarrelConfig:
    """Configuration matching the choices reported in Marrel et al. (2010) for Campbell2D."""
    d: int = 8
    x_low: float = -1.0
    x_high: float = 5.0

    # Spatial grid (nz = 64*64)
    grid_n1: int = 64
    grid_n2: int = 64
    z_low: float = -90.0
    z_high: float = 90.0

    # Paper choices for the functional metamodel build (Campbell2D study)
    n_train: int = 200             # learning sample size
    K: int = 4096                  # wavelet coefficients for 64x64
    k_gp: int = 30                 # k* in the paper (GP-modeled coefficients)
    k_lin: int = 500               # k' in the paper (linear-modeled coefficients), heuristic

    # "Exact" total indices in paper computed with Saltelli MC N=1e5 -> N*(d+2)=1e6 model evals
    saltelli_N_exact: int = 100_000

    # Metamodel-based Sobol estimation procedure (paper, Sec. 3.4)
    # First-order:
    mc_outer_first: int = 200       # integrate over Xi
    mc_inner_first: int = 1000      # integrate over X_-i (conditional mean)
    # Total-effect:
    mc_outer_total: int = 200       # integrate over Xi (outer on X_-i actually; see paper bullets)
    mc_inner_total: int = 1000      # integrate over X_-i (conditional mean)

    # Variance estimation Var(Y) in paper: 2*10^4 simulations
    mc_varY: int = 20_000

    # Wavelet basis: "Daubechies" (paper). We pick db4 as a widely used choice.
    wavelet: str = "db4"
    wavelet_level: int = 6

    # FRK-like basis (Python analog of FRK v2 basis/precision idea)
    frk_levels: int = 3
    frk_knots_per_level: tuple[int, ...] = (10, 15, 20)
    frk_radius_factor: float = 2.5
    frk_lambda_pen: float = 1e-2
    frk_add_intercept: bool = False          # for 64 -> max level depends on wavelet; we clamp safely

DEFAULTS = MarrelConfig()
