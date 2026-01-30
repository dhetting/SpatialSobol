from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_map(ax, M: np.ndarray, title: str = "", vmin=None, vmax=None):
    im = ax.imshow(M, origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    return im

def plot_index_grid(maps: np.ndarray, titles, *, suptitle: str, ncols: int = 4, vmin=None, vmax=None):
    d = maps.shape[0]
    nrows = int(np.ceil(d / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2*ncols, 3.0*nrows), constrained_layout=True)
    axes = np.array(axes).reshape(nrows, ncols)
    for i in range(nrows*ncols):
        ax = axes.flat[i]
        if i < d:
            im = plot_map(ax, maps[i], titles[i], vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        else:
            ax.axis("off")
    fig.suptitle(suptitle, y=1.02)
    return fig


def plot_s2_st2_s6_st6_comparison(
    *,
    S_exact: np.ndarray,
    ST_exact: np.ndarray,
    S_wavelet: np.ndarray,
    ST_wavelet: np.ndarray,
    S_frk: np.ndarray,
    ST_frk: np.ndarray,
    inputs: tuple[int, int] = (2, 6),
    figsize: tuple[float, float] = (14, 10),
):
    """Plot S_2, ST_2, S_6, ST_6 indices in a 4x4 grid.

    Columns:  S_i, ST_i for i in inputs[0], inputs[1]
    Rows:
      1) Exact
      2) Wavelet
      3) FRK
      4) Wavelet - FRK

    Notes
    -----
    - Indices are plotted with vmin=0, vmax=1.
    - Differences are plotted with symmetric limits based on max absolute difference.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    iA = inputs[0] - 1
    iB = inputs[1] - 1

    panels = [
        (S_exact[iA],   f"Exact S_{inputs[0]}"),
        (ST_exact[iA],  f"Exact ST_{inputs[0]}"),
        (S_exact[iB],   f"Exact S_{inputs[1]}"),
        (ST_exact[iB],  f"Exact ST_{inputs[1]}"),
        (S_wavelet[iA], f"Wavelet S_{inputs[0]}"),
        (ST_wavelet[iA],f"Wavelet ST_{inputs[0]}"),
        (S_wavelet[iB], f"Wavelet S_{inputs[1]}"),
        (ST_wavelet[iB],f"Wavelet ST_{inputs[1]}"),
        (S_frk[iA],     f"FRK S_{inputs[0]}"),
        (ST_frk[iA],    f"FRK ST_{inputs[0]}"),
        (S_frk[iB],     f"FRK S_{inputs[1]}"),
        (ST_frk[iB],    f"FRK ST_{inputs[1]}"),
        (S_wavelet[iA]-S_frk[iA],     f"Wavelet − FRK S_{inputs[0]}"),
        (ST_wavelet[iA]-ST_frk[iA],   f"Wavelet − FRK ST_{inputs[0]}"),
        (S_wavelet[iB]-S_frk[iB],     f"Wavelet − FRK S_{inputs[1]}"),
        (ST_wavelet[iB]-ST_frk[iB],   f"Wavelet − FRK ST_{inputs[1]}"),
    ]

    diff_stack = np.stack([panels[j][0] for j in range(12, 16)], axis=0)
    diff_lim = float(np.nanmax(np.abs(diff_stack)))
    diff_lim = max(diff_lim, 1e-6)

    fig, axes = plt.subplots(4, 4, figsize=figsize, constrained_layout=True)
    for r in range(4):
        for c in range(4):
            ax = axes[r, c]
            M, title = panels[r*4 + c]
            if r < 3:
                im = ax.imshow(M, origin="lower", aspect="equal", vmin=0, vmax=1)
            else:
                im = ax.imshow(M, origin="lower", aspect="equal", vmin=-diff_lim, vmax=diff_lim)
            ax.set_title(title)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(f"S/ST comparison for X{inputs[0]} and X{inputs[1]}", y=1.02)
    return fig
