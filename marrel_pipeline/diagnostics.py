from __future__ import annotations
import numpy as np

def mse_q2(Y_true: np.ndarray, Y_pred: np.ndarray) -> tuple[float, float]:
    """Compute MSE over space and samples, and Q2 as in Marrel et al. (2010), Eq. (12)."""
    # Y_true/pred shape (m,n1,n2)
    err = (Y_true - Y_pred).astype(np.float64)
    mse = float(np.mean(err**2))
    # Ez VarX[Y(X,z)] estimated from Y_true sample
    var_over_X = np.var(Y_true.astype(np.float64), axis=0, ddof=1)
    denom = float(np.mean(var_over_X))
    q2 = 1.0 - mse / denom if denom > 0 else float("nan")
    return mse, q2

def rmae(S_est: np.ndarray, S_true: np.ndarray) -> float:
    """Relative mean absolute error (rMAE) from Marrel et al. (2010), Eq. (13)."""
    num = float(np.mean(np.abs(S_est - S_true)))
    den = float(np.mean(S_true))
    if den == 0:
        return float("nan")
    return num / den
