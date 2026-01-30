from __future__ import annotations
import numpy as np

def lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube on [0,1]^d.

    For each dimension j, sample one point uniformly within each of the n strata
    and then randomly permute the strata assignment across rows.
    """
    H = np.empty((n, d), dtype=float)
    for j in range(d):
        u = rng.random(n)
        perm = rng.permutation(n)
        H[:, j] = (perm + u) / n
    return H

def pairwise_min_dist(X: np.ndarray) -> float:
    """Minimum pairwise Euclidean distance (maximin criterion)."""
    n = X.shape[0]
    min_d = np.inf
    for i in range(n):
        di = np.sqrt(((X[i+1:] - X[i])**2).sum(axis=1))
        if di.size:
            md = di.min()
            if md < min_d:
                min_d = md
    return float(min_d)

def maximin_lhs(n: int, d: int, low: float, high: float, *,
               iters: int = 200, seed: int = 12345) -> np.ndarray:
    """Maximin Latin hypercube on [low, high]^d.

    Random search over LHS candidates to maximize the minimum pairwise distance.
    """
    rng = np.random.default_rng(seed)
    best_X = None
    best_score = -np.inf
    for _ in range(iters):
        H = lhs(n, d, rng)
        score = pairwise_min_dist(H)
        if score > best_score:
            best_score = score
            best_X = H
    return low + (high - low) * best_X
