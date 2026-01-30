from __future__ import annotations
import numpy as np
from .campbell2d import campbell2d_maps_batch

def _saltelli_matrices(N: int, d: int, low: float, high: float, seed: int):
    rng = np.random.default_rng(seed)
    A = low + (high - low) * rng.random((N, d))
    B = low + (high - low) * rng.random((N, d))
    AB = np.empty((d, N, d), dtype=np.float64)
    for i in range(d):
        AB[i] = A
        AB[i][:, i] = B[:, i]
    return A, B, AB

def saltelli_exact_sobol_maps(Z1: np.ndarray, Z2: np.ndarray, *,
                             N: int = 100_000,
                             low: float = -1.0, high: float = 5.0,
                             seed: int = 2020,
                             chunk: int = 2000):
    """Compute first-order and total-effect Sobol maps by Saltelli estimators on the true Campbell2D simulator.

    This is used as the "exact" reference in our reproducible pipeline (paper used analytic first-order and Saltelli MC for totals).

    Returns:
      S: (d,n1,n2) first-order
      ST: (d,n1,n2) total-effect
    """
    d = 8
    A, B, AB = _saltelli_matrices(N, d, low, high, seed)
    n1, n2 = Z1.shape
    P = n1*n2

    # Online accumulation of mean and variance for f(A), and covariances with f(ABi)
    # We use Saltelli 2010-style estimators:
    # S_i = (1/N) sum f(B) * (f(ABi) - f(A)) / Var(Y)
    # ST_i = (1/(2N)) sum (f(A) - f(ABi))^2 / Var(Y)
    # Here Var(Y) estimated from f(A) and f(B) pooled.
    sum_fA = np.zeros(P, dtype=np.float64)
    sum_fB = np.zeros(P, dtype=np.float64)
    sum_fA2 = np.zeros(P, dtype=np.float64)
    sum_fB2 = np.zeros(P, dtype=np.float64)

    sum_S_num = np.zeros((d, P), dtype=np.float64)
    sum_ST_num = np.zeros((d, P), dtype=np.float64)

    for start in range(0, N, chunk):
        end = min(N, start+chunk)
        A_c = A[start:end]
        B_c = B[start:end]
        fA = campbell2d_maps_batch(A_c, Z1, Z2, dtype=np.float64).reshape(-1, P)
        fB = campbell2d_maps_batch(B_c, Z1, Z2, dtype=np.float64).reshape(-1, P)
        sum_fA += fA.sum(axis=0)
        sum_fB += fB.sum(axis=0)
        sum_fA2 += (fA**2).sum(axis=0)
        sum_fB2 += (fB**2).sum(axis=0)

        for i in range(d):
            AB_c = AB[i, start:end]
            fAB = campbell2d_maps_batch(AB_c, Z1, Z2, dtype=np.float64).reshape(-1, P)
            sum_S_num[i] += (fB * (fAB - fA)).sum(axis=0)
            sum_ST_num[i] += ((fA - fAB)**2).sum(axis=0)

    N_eff = N
    mean = (sum_fA + sum_fB) / (2*N_eff)
    var = (sum_fA2 + sum_fB2) / (2*N_eff) - mean**2
    var = np.maximum(var, 1e-12)

    S = (sum_S_num / N_eff) / var
    ST = (sum_ST_num / (2*N_eff)) / var

    return S.reshape(d, n1, n2), ST.reshape(d, n1, n2)
