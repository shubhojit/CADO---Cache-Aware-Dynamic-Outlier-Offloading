from __future__ import annotations

"""Sweep tile size and local budget as a cache-aware proxy for CADO routing.

This experiment is designed to close the remaining gap in the proposal: instead of
only varying the global threshold, it models a processor that works on tiles of
features and must decide how many columns in each tile can stay in a higher-precision
path while the rest proceed through the cheaper int8 path.

This is still a toy numerical proxy, but it mimics the core cache-pressure tradeoff:
if the fast-memory budget is small, the router has to choose carefully which columns
inside each tile deserve the expensive representation.
"""

import csv
from pathlib import Path

import numpy as np

from src.cado_quant import dequantize_int8, quantize_int8
from src.cado_reference import cado_reference_with_mask, fp32_gemm, relative_error


def make_outlier_matrix(
    rows: int = 128,
    cols: int = 256,
    outlier_fraction: float = 0.05,
    rng: np.random.Generator | None = None,
    outlier_scale: float = 20.0,
) -> np.ndarray:
    """Create a synthetic activation matrix with a sparse set of large columns.

    The matrix is close to Gaussian in the ordinary case, but a few columns are
    amplified to mimic outlier activations that should be routed through a more
    expensive path when the cache budget is limited.
    """
    rng = rng or np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(rows, cols)).astype(np.float32)

    if outlier_fraction > 0.0:
        n_outliers = max(1, int(round(cols * outlier_fraction)))
        target_cols = rng.choice(cols, size=n_outliers, replace=False)
        x[:, target_cols] = x[:, target_cols] * outlier_scale + rng.normal(
            0.0, 0.5, size=(rows, n_outliers)
        )
    return x.astype(np.float32)


def tile_dynamic_mask(
    x: np.ndarray,
    tile_size: int,
    budget_fraction: float,
) -> np.ndarray:
    """Select the top columns inside each tile using the current activation values.

    Args:
        x: Activation matrix with shape (batch, features).
        tile_size: Number of feature columns processed together inside one tile.
        budget_fraction: Fraction of columns within each tile to keep in the
            higher-precision path.

    Returns:
        A boolean mask with one entry per feature; True means the column is treated as
        a high-precision outlier within its tile.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("Expected a 2D activation matrix.")

    mask = np.zeros(x.shape[1], dtype=bool)
    for start in range(0, x.shape[1], tile_size):
        end = min(start + tile_size, x.shape[1])
        tile = x[:, start:end]
        scores = np.max(np.abs(tile), axis=0)

        budget = max(0, min(tile.shape[1], int(round(tile.shape[1] * budget_fraction))))
        if budget <= 0:
            continue
        if budget >= tile.shape[1]:
            mask[start:end] = True
            continue

        keep_idx = np.argsort(scores)[-budget:]
        local_mask = np.zeros(tile.shape[1], dtype=bool)
        local_mask[keep_idx] = True
        mask[start:end] = local_mask
    return mask


def tile_static_mask(
    calibration: np.ndarray,
    tile_size: int,
    budget_fraction: float,
) -> np.ndarray:
    """Select columns per tile using a calibration batch instead of the test batch.

    This mirrors the static mixed-precision baseline under the same per-tile budget:
    the routing decision is fixed once from the calibration data and then reused during
    evaluation.
    """
    return tile_dynamic_mask(calibration, tile_size=tile_size, budget_fraction=budget_fraction)


def uniform_int8_baseline(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return a naive all-quantized baseline without any outlier-aware routing."""
    x_q, x_scale = quantize_int8(x)
    w_q, w_scale = quantize_int8(weights)
    x_hat = dequantize_int8(x_q.astype(np.int8), x_scale)
    w_hat = dequantize_int8(w_q.astype(np.int8), w_scale)
    return x_hat @ w_hat


def run_tile_sweep() -> list[dict[str, float | int | str]]:
    """Sweep tile sizes and per-tile high-precision fractions.

    The experiment intentionally uses a very small proxy for cache pressure: a larger
    tile means a larger working set, while a smaller budget_fraction means fewer
    columns within each tile can be kept in the expensive path.
    """
    rng = np.random.default_rng(123)
    weights = rng.normal(0.0, 1.0, size=(256, 64)).astype(np.float32)
    outlier_fraction = 0.05

    rows: list[dict[str, float | int | str]] = []
    tile_sizes = [8, 16, 32, 64, 128, 256]
    budget_fractions = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]

    for tile_size in tile_sizes:
        # Use one calibration batch and one evaluation batch with the same outlier pattern
        # so the static policy is evaluated under a fair comparison.
        calibration = make_outlier_matrix(
            rows=128,
            cols=256,
            outlier_fraction=outlier_fraction,
            rng=rng,
        )
        eval_x = make_outlier_matrix(
            rows=128,
            cols=256,
            outlier_fraction=outlier_fraction,
            rng=rng,
        )

        reference = fp32_gemm(eval_x, weights)
        uniform_err = relative_error(reference, uniform_int8_baseline(eval_x, weights))

        for budget_fraction in budget_fractions:
            dynamic_mask = tile_dynamic_mask(eval_x, tile_size=tile_size, budget_fraction=budget_fraction)
            static_mask = tile_static_mask(calibration, tile_size=tile_size, budget_fraction=budget_fraction)

            dynamic_est = cado_reference_with_mask(eval_x, weights, dynamic_mask)
            static_est = cado_reference_with_mask(eval_x, weights, static_mask)
            dynamic_err = relative_error(reference, dynamic_est)
            static_err = relative_error(reference, static_est)

            rows.append(
                {
                    "tile_size": int(tile_size),
                    "budget_fraction": float(budget_fraction),
                    "dynamic_relative_error": float(dynamic_err),
                    "static_relative_error": float(static_err),
                    "uniform_int8_relative_error": float(uniform_err),
                    "dynamic_high_precision_cols": int(np.count_nonzero(dynamic_mask)),
                    "static_high_precision_cols": int(np.count_nonzero(static_mask)),
                }
            )
    return rows


def main() -> None:
    """Write a CSV containing the cache-aware tile sweep results."""
    rows = run_tile_sweep()
    output_path = Path(__file__).resolve().parents[1] / "results" / "cache_tile_sweep.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as csv_file:
        fieldnames = [
            "tile_size",
            "budget_fraction",
            "dynamic_relative_error",
            "static_relative_error",
            "uniform_int8_relative_error",
            "dynamic_high_precision_cols",
            "static_high_precision_cols",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved_csv={output_path}")
    for tile_size in [8, 16, 32, 64, 128, 256]:
        subset = [r for r in rows if r["tile_size"] == tile_size and r["budget_fraction"] in {0.0, 0.05, 0.10, 0.20, 0.30}]
        print(f"\ntile_size={tile_size}")
        for row in subset:
            print(
                f"budget={row['budget_fraction']:.2f} "
                f"dyn={row['dynamic_relative_error']:.6f} "
                f"static={row['static_relative_error']:.6f} "
                f"uniform={row['uniform_int8_relative_error']:.6f}"
            )


if __name__ == "__main__":
    main()
