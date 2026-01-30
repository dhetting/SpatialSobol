from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from .frk_basis import FRKBasis, build_multires_bisquare_basis

def frk_decompose_training(Y_train: np.ndarray,
                           Z1: np.ndarray, Z2: np.ndarray,
                           *,
                           levels: int = 3,
                           knots_per_level: tuple[int,...] = (10, 15, 20),
                           radius_factor: float = 2.5,
                           lambda_pen: float = 1e-2,
                           add_intercept: bool = False):
    """Decompose training maps into FRK basis coefficients via penalized least squares.

    We estimate mu(z) as the sample mean over runs (as in Marrel et al. wavelet step),
    and for each run solve:
        eta_hat = argmin || y - mu - B eta ||^2 + lambda * eta^T Q eta
    giving:
        (B^T B + lambda Q) eta_hat = B^T (y - mu)

    Returns
    -------
    mu_map : (n1,n2)
    eta    : (n,r)
    basis  : FRKBasis (contains B, Q, etc.)
    """
    mu_map = Y_train.mean(axis=0).astype(np.float32)
    n, n1, n2 = Y_train.shape
    P = n1*n2

    basis = build_multires_bisquare_basis(Z1, Z2,
                                          levels=levels,
                                          knots_per_level=knots_per_level,
                                          radius_factor=radius_factor,
                                          add_intercept=add_intercept)
    B = basis.B  # (P,r)
    Q = basis.Q.tocsr()
    r = B.shape[1]

    BtB = B.T @ B  # dense (r,r) where r ~ 700
    A = sparse.csr_matrix(BtB) + (lambda_pen * Q)
    lu = splu(A.tocsc())

    eta = np.empty((n, r), dtype=np.float64)
    for i in range(n):
        y = (Y_train[i] - mu_map).reshape(-1).astype(np.float64)
        rhs = B.T @ y
        eta[i] = lu.solve(rhs)

    return mu_map, eta, basis

def frk_reconstruct_maps(mu_map: np.ndarray, eta: np.ndarray, basis: FRKBasis) -> np.ndarray:
    """Reconstruct maps given coefficients eta and a basis."""
    n = eta.shape[0]
    P, r = basis.B.shape
    Yc = (basis.B @ eta.T).T  # (n,P)
    Y = Yc.reshape(n, *basis.grid_shape) + mu_map[None, :, :]
    return Y.astype(np.float32, copy=False)
