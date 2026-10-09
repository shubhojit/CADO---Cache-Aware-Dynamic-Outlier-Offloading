from __future__ import annotations

"""Sweep the routing threshold and track the error-vs-cost tradeoff."""

import csv
from pathlib import Path

import numpy as np

from src.cado_quant import dequantize_int8, quantize_int8
from src.cado_reference import cado_reference, fp32_gemm, relative_error
from src.cado_router import route_summary


def make_outlier_matrix(
    rows: int = 128,
    cols: int = 256,
    outlier_fraction: float = 0.05,
    rng: np.random.Generator | None = None,
    outlier_scale: float = 20.0,
) -> np.ndarray:
    """Create a synthetic matrix with a controlled number of outlier columns.

    In this simple setup, the activation matrix is mostly near-Gaussian, with a few
    columns amplified to represent high-impact values that should be isolated by the
    outlier router.
    """
    rng = rng or np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(rows, cols)).astype(np.float32)

    if outlier_fraction <= 0.0:
        return x

    n_outliers = max(1, int(round(cols * outlier_fraction)))
    target_cols = rng.choice(cols, size=n_outliers, replace=False)
    x[:, target_cols] = x[:, target_cols] * outlier_scale + rng.normal(
        0.0, 0.5, size=(rows, n_outliers)
    )
    return x.astype(np.float32)


def uniform_int8_baseline(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return a simple all-quantized baseline for comparison.

    This differs from CADO because it never isolates outlier features. It is a useful
    fallback baseline when evaluating whether the outlier-aware route adds value.
    """
    x_q, x_scale = quantize_int8(x)
    w_q, w_scale = quantize_int8(weights)
    x_hat = dequantize_int8(x_q.astype(np.int8), x_scale)
    w_hat = dequantize_int8(w_q.astype(np.int8), w_scale)
    return x_hat @ w_hat


def main() -> None:
    """Sweep the threshold and summarize both accuracy and cost proxy data.

    The cost proxy here is the fraction of activation columns routed through the
    high-precision path. This gives a lightweight measure of how much work is being
    treated as exceptional instead of low-precision.
    """
    rng = np.random.default_rng(42)
    weights = rng.normal(0.0, 1.0, size=(256, 64)).astype(np.float32)
    outlier_fraction = 0.05

    rows: list[dict[str, float | int]] = []
    thresholds = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    for threshold in thresholds:
        x = make_outlier_matrix(
            rows=128,
            cols=256,
            outlier_fraction=outlier_fraction,
            rng=rng,
        )
        reference = fp32_gemm(x, weights)
        approx = cado_reference(x, weights, threshold=threshold)
        err = relative_error(reference, approx)
        summary = route_summary(x, threshold=threshold)
        baseline = uniform_int8_baseline(x, weights)
        baseline_err = relative_error(reference, baseline)

        row = {
            "threshold": float(threshold),
            "actual_outlier_fraction": float(summary["outlier_fraction"]),
            "outlier_count": int(summary["outlier_count"]),
            "cado_relative_error": float(err),
            "uniform_int8_relative_error": float(baseline_err),
        }
        rows.append(row)

        print(
            f"threshold={threshold:.1f} "
            f"outlier_frac={summary['outlier_fraction']:.4f} "
            f"cado_err={err:.6f} "
            f"baseline_err={baseline_err:.6f}"
        )

    output_path = Path(__file__).resolve().parents[1] / "results" / "threshold_sweep.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        fieldnames = [
            "threshold",
            "actual_outlier_fraction",
            "outlier_count",
            "cado_relative_error",
            "uniform_int8_relative_error",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved_csv={output_path}")


if __name__ == "__main__":
    main()
