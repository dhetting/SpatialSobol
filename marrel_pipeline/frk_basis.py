from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import sparse

@dataclass
class FRKBasis:
    """Fixed-rank basis for FRK-like surrogate on a regular grid.

    This is a Python approximation of the basis-expansion component of FRK/FRK v2:
        Y(s) = mu(s) + B(s) eta + xi(s)

    We use compactly-supported bisquare basis functions at multiple resolutions.
    """
    centers: np.ndarray          # (r,2) centers in spatial coords
    radii: np.ndarray            # (r,) support radii
    B: np.ndarray                # (P,r) basis evaluated on grid (flattened)
    Q: sparse.spmatrix           # (r,r) sparse precision / penalty matrix
    grid_shape: tuple[int,int]   # (n1,n2)

def _bisquare(d: np.ndarray, r: float) -> np.ndarray:
    # d: distances
    u = d / r
    out = np.zeros_like(u)
    mask = u < 1.0
    out[mask] = (1.0 - u[mask]**2)**2
    return out

def build_multires_bisquare_basis(Z1: np.ndarray, Z2: np.ndarray,
                                  *,
                                  levels: int = 3,
                                  knots_per_level: tuple[int,...] = (10, 15, 20),
                                  radius_factor: float = 2.5,
                                  add_intercept: bool = False) -> FRKBasis:
    """Construct a multi-resolution bisquare basis on the grid.

    Parameters
    ----------
    Z1, Z2 : (n1,n2) grid coordinate arrays
    levels : number of resolutions (must match len(knots_per_level))
    knots_per_level : number of knot locations per axis at each level; total r is sum(m_l^2)
    radius_factor : radius = radius_factor * knot_spacing (per level)
    add_intercept : if True, prepend a constant basis

    Returns
    -------
    FRKBasis with B evaluated on the flattened grid, and a sparse penalty Q.
    """
    assert levels == len(knots_per_level), "levels must equal len(knots_per_level)"
    n1, n2 = Z1.shape
    P = n1*n2
    pts = np.column_stack([Z1.reshape(-1), Z2.reshape(-1)])

    centers = []
    radii = []

    for m in knots_per_level:
        c1 = np.linspace(Z1.min(), Z1.max(), m)
        c2 = np.linspace(Z2.min(), Z2.max(), m)
        C1, C2 = np.meshgrid(c1, c2, indexing="ij")
        C = np.column_stack([C1.reshape(-1), C2.reshape(-1)])
        centers.append(C)

        # approximate spacing (uniform)
        spacing = float((Z1.max() - Z1.min()) / (m - 1)) if m > 1 else float(Z1.max() - Z1.min())
        r = radius_factor * spacing
        radii.append(np.full(C.shape[0], r, dtype=float))

    centers = np.vstack(centers).astype(float)
    radii = np.concatenate(radii).astype(float)

    if add_intercept:
        centers = np.vstack([np.array([[np.nan, np.nan]]), centers])
        radii = np.concatenate([np.array([np.inf]), radii])

    r = centers.shape[0]
    B = np.zeros((P, r), dtype=np.float64)

    # Evaluate basis
    for j in range(r):
        if add_intercept and j == 0:
            B[:, 0] = 1.0
            continue
        c = centers[j]
        rr = radii[j]
        d = np.sqrt(((pts - c)**2).sum(axis=1))
        B[:, j] = _bisquare(d, rr)

    # Build a sparse penalty Q using a graph Laplacian over knot centers within each level.
    Q = build_block_laplacian(centers[1:] if add_intercept else centers,
                              blocks=[m*m for m in knots_per_level])
    if add_intercept:
        Q = sparse.block_diag([sparse.csr_matrix((1,1)), Q], format="csr")

    return FRKBasis(centers=centers, radii=radii, B=B, Q=Q, grid_shape=(n1,n2))

def build_block_laplacian(centers: np.ndarray, *, blocks: list[int], k: int = 4) -> sparse.spmatrix:
    """Sparse block-diagonal Laplacian penalty for multi-resolution knot blocks.

    For each block (corresponding to one resolution), connect each knot to its k nearest neighbors
    (within-block) and build an unweighted graph Laplacian L = D - W.

    This is a lightweight analog of the sparse-precision idea used in FRK v2.
    """
    Q_blocks = []
    start = 0
    for b in blocks:
        C = centers[start:start+b]
        start += b
        # pairwise distances (b can be up to 400; OK)
        D2 = ((C[:, None, :] - C[None, :, :])**2).sum(axis=2)
        np.fill_diagonal(D2, np.inf)
        nn = np.argsort(D2, axis=1)[:, :k]
        rows = np.repeat(np.arange(b), k)
        cols = nn.reshape(-1)
        data = np.ones(rows.size, dtype=float)
        W = sparse.csr_matrix((data, (rows, cols)), shape=(b, b))
        # symmetrize
        W = ((W + W.T) > 0).astype(float).tocsr()
        deg = np.array(W.sum(axis=1)).reshape(-1)
        L = sparse.diags(deg, format="csr") - W
        # add a small ridge for numerical stability
        L = L + 1e-6 * sparse.identity(b, format="csr")
        Q_blocks.append(L)
    return sparse.block_diag(Q_blocks, format="csr")
