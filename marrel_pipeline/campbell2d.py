from __future__ import annotations
import numpy as np

def make_spatial_grid(n1: int = 64, n2: int = 64, z_low: float = -90.0, z_high: float = 90.0):
    """Create spatial grid (z1,z2) in [z_low,z_high]^2 with shape (n1,n2)."""
    z1 = np.linspace(z_low, z_high, n1)
    z2 = np.linspace(z_low, z_high, n2)
    Z1, Z2 = np.meshgrid(z1, z2, indexing="ij")
    return Z1, Z2

def campbell2d_map(X: np.ndarray, Z1: np.ndarray, Z2: np.ndarray) -> np.ndarray:
    """Campbell2D function (Marrel et al., 2010, Eq. (6)).

    Inputs:
      X: shape (8,), each Xi ~ Unif[-1,5]
      Z1, Z2: spatial grids, same shape (n1,n2), in degrees in [-90,90]
    Output:
      Y: shape (n1,n2)
    """
    x1,x2,x3,x4,x5,x6,x7,x8 = X
    term1 = x1 * np.exp(-((0.8*Z1 + 0.2*Z2 - 10*x2)**2) / (60*(x2**2)*1.0))
    term2 = (x2 + x4) * np.exp(((0.5*Z1 + 0.5*Z2) * x1) / 500.0)
    term3 = x5*(x3 - 2.0) * np.exp(-((0.4*Z1 + 0.6*Z2 - 20*x6)**2) / (40*(x5**2)*1.0))
    term4 = (x6 + x8) * np.exp(((0.3*Z1 + 0.7*Z2) * x7) / 250.0)
    return term1 + term2 + term3 + term4

def campbell2d_maps_batch(Xs: np.ndarray, Z1: np.ndarray, Z2: np.ndarray, *,
                          dtype=np.float32) -> np.ndarray:
    """Batch evaluation returning (n, n1, n2)."""
    n = Xs.shape[0]
    out = np.empty((n, Z1.shape[0], Z1.shape[1]), dtype=dtype)
    for i in range(n):
        out[i] = campbell2d_map(Xs[i], Z1, Z2).astype(dtype, copy=False)
    return out
