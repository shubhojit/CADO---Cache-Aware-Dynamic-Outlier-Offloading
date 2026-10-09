from __future__ import annotations

"""Reference implementations for the CADO prototype.

These functions intentionally keep the math explicit so the research idea is easy to
inspect and validate before optimizing for performance.
"""

import numpy as np


def fp32_gemm(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute the reference full-precision matrix multiply.

    Args:
        x: Input activations with shape (batch, features).
        weights: Weight matrix with shape (features, output_dim).

    Returns:
        The dense FP32 output matrix with shape (batch, output_dim).
    """
    x = np.asarray(x, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if x.ndim != 2 or weights.ndim != 2:
        raise ValueError("Expected 2D arrays for x and weights.")
    if x.shape[1] != weights.shape[0]:
        raise ValueError("x.shape[1] must match weights.shape[0].")
    return x @ weights


def outlier_mask(x: np.ndarray, threshold: float = 6.0) -> np.ndarray:
    """Return a boolean mask marking activation columns that pass the outlier test.

    A feature column is treated as an outlier if its maximum absolute magnitude is at
    least ``threshold``. This is a simple proxy for the high-impact activations that we
    want to route through a higher-precision path.

    Args:
        x: Activation matrix of shape (batch, features).
        threshold: Magnitude threshold used to classify a feature as an outlier.

    Returns:
        A boolean array shaped like (features,), where True means the column is an
        outlier.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("Expected a 2D activation matrix.")
    return np.max(np.abs(x), axis=0) >= threshold


def cado_reference(x: np.ndarray, weights: np.ndarray, threshold: float = 6.0) -> np.ndarray:
    """Run a simple CADO decomposition and return the approximate output.

    The idea is to separate outlier columns from normal columns:
    - outlier columns are computed exactly in FP32
    - normal columns are quantized to int8 and reconstructed with a scale factor

    This function is intentionally easy to read and validate, not optimized for speed.

    Args:
        x: Activation matrix with shape (batch, features).
        weights: Weight matrix with shape (features, output_dim).
        threshold: Magnitude cutoff used to detect outlier features.

    Returns:
        The approximate output with shape (batch, output_dim).
    """
    x = np.asarray(x, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if x.ndim != 2 or weights.ndim != 2:
        raise ValueError("Expected 2D arrays for x and weights.")
    if x.shape[1] != weights.shape[0]:
        raise ValueError("x.shape[1] must match weights.shape[0].")

    mask = outlier_mask(x, threshold=threshold)
    result = np.zeros((x.shape[0], weights.shape[1]), dtype=np.float32)

    # Compute the high-impact outlier columns explicitly in full precision.
    if np.any(mask):
        result += x[:, mask] @ weights[mask, :]

    # For the remaining columns, quantize the activations and weights to int8.
    normal_mask = ~mask
    if np.any(normal_mask):
        x_normal = x[:, normal_mask]
        w_normal = weights[normal_mask, :]

        # Use a per-matrix scale so the quantization fits in the signed int8 range.
        scale_x = max(float(np.max(np.abs(x_normal))) / 127.0, 1e-12)
        scale_w = max(float(np.max(np.abs(w_normal))) / 127.0, 1e-12)

        x_int8 = np.clip(np.round(x_normal / scale_x), -127, 127).astype(np.int8)
        w_int8 = np.clip(np.round(w_normal / scale_w), -127, 127).astype(np.int8)

        result += (x_int8.astype(np.int32) @ w_int8.astype(np.int32)) * (scale_x * scale_w)

    return result


def relative_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Return the normalized L2 error between a reference and an estimate.

    The result is a scalar in the same broad range as a relative error metric, where
    lower values mean the approximation is closer to the reference output.

    Args:
        reference: The full-precision target array.
        estimate: The approximation we want to compare against the target.

    Returns:
        The relative L2 error, computed as ||ref - est|| / ||ref||.
    """
    ref = np.asarray(reference, dtype=np.float32)
    est = np.asarray(estimate, dtype=np.float32)
    denom = np.linalg.norm(ref)
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(ref - est) / denom)
