from __future__ import annotations

"""Compare the CADO approximation against a uniform int8 baseline."""

import csv
from pathlib import Path

import numpy as np

from src.cado_quant import dequantize_int8, quantize_int8
from src.cado_reference import cado_reference, fp32_gemm, relative_error


def uniform_int8_baseline(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the output of a naive full-matrix int8 approximation.

    This baseline quantizes both the activations and weights globally, then reconstructs
    them before multiplying. It is deliberately simple and intentionally worse than the
    outlier-aware CADO route in structured outlier scenarios.

    Args:
        x: Activation matrix.
        weights: Weight matrix.

    Returns:
        The full int8 approximation of the matrix product.
    """
    x_q, x_scale = quantize_int8(x)
    w_q, w_scale = quantize_int8(weights)
    x_hat = dequantize_int8(x_q.astype(np.int8), x_scale)
    w_hat = dequantize_int8(w_q.astype(np.int8), w_scale)
    return x_hat @ w_hat


def make_outlier_matrix(
    rows: int = 128,
    cols: int = 256,
    outlier_fraction: float = 0.05,
    rng: np.random.Generator | None = None,
    outlier_scale: float = 20.0,
) -> np.ndarray:
    """Construct a synthetic activation matrix with a few extreme outlier columns.

    Args:
        rows: Number of batch rows.
        cols: Number of activation columns.
        outlier_fraction: Fraction of columns to inflate as outliers.
        rng: Optional RNG for deterministic behavior.
        outlier_scale: Multiplier applied to the selected outlier columns.

    Returns:
        A float32 activation matrix with a sparse but impactful outlier structure.
    """
    rng = rng or np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(rows, cols)).astype(np.float32)
    if outlier_fraction > 0:
        n_outliers = max(1, int(round(cols * outlier_fraction)))
        target_cols = rng.choice(cols, size=n_outliers, replace=False)
        x[:, target_cols] = x[:, target_cols] * outlier_scale + rng.normal(
            0.0, 0.5, size=(rows, n_outliers)
        )
    return x.astype(np.float32)


def main() -> None:
    """Compare the full-precision reference, CADO, and a naive int8 baseline.

    The script prints the relative error for each method and saves the numerical results
    to a CSV file so they can be inspected or plotted later.
    """
    rng = np.random.default_rng(11)
    x = make_outlier_matrix(rows=128, cols=256, outlier_fraction=0.05, rng=rng)
    weights = rng.normal(0.0, 1.0, size=(256, 64)).astype(np.float32)

    full_ref = fp32_gemm(x, weights)
    cado_est = cado_reference(x, weights, threshold=6.0)
    uniform_est = uniform_int8_baseline(x, weights)

    cado_error = relative_error(full_ref, cado_est)
    uniform_error = relative_error(full_ref, uniform_est)

    print(f"full_fp32_error={relative_error(full_ref, full_ref):.6f}")
    print(f"cado_relative_error={cado_error:.6f}")
    print(f"uniform_int8_relative_error={uniform_error:.6f}")
    print(f"cado_shape={cado_est.shape}")
    print(f"uniform_shape={uniform_est.shape}")

    output_path = Path(__file__).resolve().parents[1] / "results" / "baseline_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["method", "relative_error", "shape"],
        )
        writer.writeheader()
        writer.writerow({"method": "fp32_reference", "relative_error": 0.0, "shape": str(full_ref.shape)})
        writer.writerow({"method": "cado", "relative_error": float(cado_error), "shape": str(cado_est.shape)})
        writer.writerow({"method": "uniform_int8", "relative_error": float(uniform_error), "shape": str(uniform_est.shape)})

    print(f"saved_csv={output_path}")


if __name__ == "__main__":
    main()
