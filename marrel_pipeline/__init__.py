"""Reproducible implementation of Marrel et al. (2010) spatial Sobol index pipeline on the Campbell2D test case.

Main entry points are provided as composable functions (no monolithic run-all).
"""

from .config import DEFAULTS, MarrelConfig
from .design import maximin_lhs
from .campbell2d import campbell2d_map, campbell2d_maps_batch, make_spatial_grid
from .wavelet import dwt2_flatten, idwt2_unflatten
from .coeff_models import fit_coefficient_models, predict_coefficients
from .functional_metamodel import predict_map_from_models
from .sobol_exact import saltelli_exact_sobol_maps
from .sobol_metamodel import nested_mc_sobol_maps
from .diagnostics import mse_q2, rmae

from .frk_basis import FRKBasis, build_multires_bisquare_basis
from .frk import frk_decompose_training, frk_reconstruct_maps
from .functional_frk_metamodel import predict_map_from_frk_models

from .plotting import plot_s2_st2_s6_st6_comparison
