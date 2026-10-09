from __future__ import annotations

"""Utilities for deciding which activation columns count as outliers."""

import numpy as np


def route_outlier_mask(
    activations: np.ndarray,
    threshold: float = 6.0,
    max_density: float | None = None,
) -> np.ndarray:
    """Return the selected outlier mask for a 2D activation matrix.

    The router marks a feature column as an outlier when its largest absolute magnitude
    exceeds the configured threshold. If ``max_density`` is provided, the number of
    outliers is capped to that fraction of the feature dimension.

    Args:
        activations: Matrix of activations with shape (batch, features).
        threshold: Minimum magnitude needed to treat a feature as an outlier.
        max_density: Optional upper bound on the outlier fraction, such as 0.10.

    Returns:
        A boolean mask with shape (features,), where True marks outlier columns.
    """
    activations = np.asarray(activations, dtype=np.float32)
    if activations.ndim != 2:
        raise ValueError("Expected a 2D activation matrix.")

    # A feature is considered extreme if any element in that column is large.
    mask = np.max(np.abs(activations), axis=0) >= threshold

    if max_density is not None:
        total_cols = activations.shape[1]
        allowed = max(1, int(round(total_cols * max_density)))
        if int(np.count_nonzero(mask)) > allowed:
            scores = np.max(np.abs(activations), axis=0)
            keep_idx = np.argsort(scores)[-allowed:]
            mask = np.zeros_like(mask, dtype=bool)
            mask[keep_idx] = True
    return mask


def route_summary(activations: np.ndarray, threshold: float = 6.0) -> dict[str, float | int]:
    """Summarize the outlier routing decision for a matrix of activations.

    Args:
        activations: Matrix of activations with shape (batch, features).
        threshold: Magnitude threshold used by ``route_outlier_mask``.

    Returns:
        A dictionary containing the number of outliers, the fraction of outlier columns,
        and the threshold used.
    """
    mask = route_outlier_mask(activations, threshold=threshold)
    outlier_frac = float(np.count_nonzero(mask) / activations.shape[1])
    return {
        "outlier_count": int(np.count_nonzero(mask)),
        "outlier_fraction": outlier_frac,
        "threshold": float(threshold),
    }
