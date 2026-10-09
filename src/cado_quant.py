from __future__ import annotations

"""Low-precision helpers used to emulate int8 quantization in the CADO prototype."""

import numpy as np


def quantize_int8(values: np.ndarray, scale: float | None = None) -> tuple[np.ndarray, float]:
    """Quantize a float array to signed int8 using a scalar scale.

    Args:
        values: Floating-point tensor to quantize.
        scale: Optional explicit scale factor. If omitted, it is inferred from the
            maximum absolute value in ``values``.

    Returns:
        A tuple ``(quantized_values, scale)``. The first element contains the int8
        representation; the second is the float scale used to map back to the original
        magnitude.
    """
    values = np.asarray(values, dtype=np.float32)
    max_abs = float(np.max(np.abs(values))) if values.size else 1.0
    scale = scale if scale is not None else max(max_abs / 127.0, 1e-12)
    quantized = np.clip(np.round(values / scale), -127, 127).astype(np.int8)
    return quantized, float(scale)


def dequantize_int8(values: np.ndarray, scale: float) -> np.ndarray:
    """Convert an int8 array back to floating point using the given scale.

    Args:
        values: Integer array representing the quantized values.
        scale: The scale used during quantization.

    Returns:
        The reconstructed floating-point array.
    """
    values = np.asarray(values, dtype=np.int32)
    return values.astype(np.float32) * float(scale)


def int8_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Return the relative error after int8 quantization and reconstruction.

    This is a compact helper for measuring how much accuracy is lost during the
    low-precision approximation.

    Args:
        original: The original floating-point values.
        reconstructed: The values after quantization and dequantization.

    Returns:
        A scalar relative L2 error between the original and reconstructed arrays.
    """
    original = np.asarray(original, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    denom = np.linalg.norm(original)
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(original - reconstructed) / denom)
