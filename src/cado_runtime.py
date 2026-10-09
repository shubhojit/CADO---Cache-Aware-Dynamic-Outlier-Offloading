from __future__ import annotations

"""A small runtime-style implementation of the CADO prototype."""

import numpy as np

from src.cado_quant import dequantize_int8, quantize_int8
from src.cado_reference import outlier_mask


def cado_runtime(x: np.ndarray, weights: np.ndarray, threshold: float = 6.0) -> np.ndarray:
    """Run a minimal runtime-style CADO approximation.

    This version intentionally mirrors the logic used in the research prototype:
    - outlier features are processed in full precision
    - non-outlier features are quantized and dequantized with int8 scales

    Args:
        x: Activation matrix with shape (batch, features).
        weights: Weight matrix with shape (features, output_dim).
        threshold: Magnitude threshold used to detect outliers.

    Returns:
        The approximated output matrix of shape (batch, output_dim).
    """
    x = np.asarray(x, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    mask = outlier_mask(x, threshold=threshold)
    result = np.zeros((x.shape[0], weights.shape[1]), dtype=np.float32)

    # Keep the high-impact outlier dimensions in a dense, exact computation.
    if np.any(mask):
        result += x[:, mask] @ weights[mask, :]

    # For the remaining columns, apply int8 quantization to each matrix and reconstruct it.
    normal_mask = ~mask
    if np.any(normal_mask):
        x_normal = x[:, normal_mask]
        w_normal = weights[normal_mask, :]
        q_x, scale_x = quantize_int8(x_normal)
        q_w, scale_w = quantize_int8(w_normal)

        deq_x = dequantize_int8(q_x.astype(np.int8), scale_x)
        deq_w = dequantize_int8(q_w.astype(np.int8), scale_w)
        result += deq_x @ deq_w

    return result
