from __future__ import annotations
import numpy as np
import pywt

def dwt2_flatten(Y: np.ndarray, wavelet: str = "db4", level: int = 6):
    """2D discrete wavelet transform (DWT) and flatten coefficients to a vector.

    Returns:
      coeff_vec: shape (K,)
      coeff_slices: pywt bookkeeping (for unflatten)
      coeff_shapes: additional shapes for consistent reconstruction
      Y_mean: pixelwise mean of Y (for centering)
    """
    # Center by spatial mean mu(z) estimated from sample (handled outside); here just decompose Y directly.
    coeffs = pywt.wavedec2(Y, wavelet=wavelet, level=level, mode="periodization")
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    return coeff_arr.reshape(-1), coeff_slices, coeff_arr.shape

def idwt2_unflatten(coeff_vec: np.ndarray, coeff_slices, coeff_arr_shape, wavelet: str = "db4"):
    """Inverse 2D DWT from flattened coefficient vector."""
    coeff_arr = coeff_vec.reshape(coeff_arr_shape)
    coeffs = pywt.array_to_coeffs(coeff_arr, coeff_slices, output_format="wavedec2")
    Y = pywt.waverec2(coeffs, wavelet=wavelet, mode="periodization")
    return Y
