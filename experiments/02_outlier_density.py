from __future__ import annotations

"""Sweep the fraction of outlier columns and record the resulting approximation error."""

import csv
from pathlib import Path

import numpy as np

from src.cado_reference import cado_reference, fp32_gemm, relative_error
from src.cado_router import route_summary


def make_outlier_matrix(
    rows: int = 128,
    cols: int = 256,
    outlier_fraction: float = 0.05,
    rng: np.random.Generator | None = None,
    outlier_scale: float = 20.0,
) -> np.ndarray:
    """Create a synthetic activation matrix with a controlled number of outlier columns.

    Args:
        rows: Number of batch rows.
        cols: Number of feature columns.
        outlier_fraction: Fraction of columns to amplify as outliers.
        rng: Optional random generator for reproducibility.
        outlier_scale: Multiplicative scale applied to selected outlier columns.

    Returns:
        An activation matrix with a controlled proportion of large-magnitude columns.
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


def main() -> None:
    """Run a density sweep and save the measurements to a CSV file.

    Each row reports the target density, the actual outlier fraction observed in the
    sampled matrix, the number of outlier columns, and the relative error of the CADO
    approximation compared with the full-precision reference.
    """
    rng = np.random.default_rng(7)
    weights = rng.normal(0.0, 1.0, size=(256, 64)).astype(np.float32)
    rows: list[dict[str, float | int]] = []

    for fraction in [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        x = make_outlier_matrix(
            rows=128,
            cols=256,
            outlier_fraction=fraction,
            rng=rng,
        )

        reference = fp32_gemm(x, weights)
        approx = cado_reference(x, weights, threshold=6.0)
        err = relative_error(reference, approx)
        summary = route_summary(x, threshold=6.0)
        row = {
            "density": float(fraction),
            "actual_outlier_fraction": float(summary["outlier_fraction"]),
            "outlier_count": int(summary["outlier_count"]),
            "relative_error": float(err),
        }
        rows.append(row)

        print(
            f"density={fraction:.2f} "
            f"actual_outlier_fraction={summary['outlier_fraction']:.4f} "
            f"err={err:.6f} "
            f"outlier_count={summary['outlier_count']}"
        )

    output_path = Path(__file__).resolve().parents[1] / "results" / "outlier_density.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        fieldnames = ["density", "actual_outlier_fraction", "outlier_count", "relative_error"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved_csv={output_path}")


if __name__ == "__main__":
    main()
