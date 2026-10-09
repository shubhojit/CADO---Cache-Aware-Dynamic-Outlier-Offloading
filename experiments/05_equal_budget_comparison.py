from __future__ import annotations

"""Compare dynamic and static outlier routing under the same precision budget.

This experiment is designed to test the blog’s central claim more directly than the
threshold sweep: for a fixed number of feature columns to route through the
higher-precision path, does adapting the selection to the current activation batch
improve quality relative to a static selection learned from a calibration batch?
"""

import csv
from pathlib import Path

import numpy as np

from src.cado_quant import dequantize_int8, quantize_int8
from src.cado_reference import fp32_gemm, relative_error


def make_outlier_matrix(
    rows: int = 128,
    cols: int = 256,
    outlier_fraction: float = 0.05,
    rng: np.random.Generator | None = None,
    outlier_scale: float = 20.0,
    target_cols: np.ndarray | None = None,
) -> np.ndarray:
    """Create a synthetic activation matrix with a controlled set of outlier columns.

    Args:
        rows: Number of batch rows.
        cols: Number of columns to generate.
        outlier_fraction: Fraction of columns to amplify if ``target_cols`` is not given.
        rng: Optional random generator used for reproducibility.
        outlier_scale: Multiplicative scale applied to the selected outlier columns.
        target_cols: Optional explicitly chosen columns to amplify.

    Returns:
        A float32 activation matrix whose outlier pattern is controlled by either the
        provided target columns or the requested outlier fraction.
    """
    rng = rng or np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(rows, cols)).astype(np.float32)

    if target_cols is not None:
        selected = np.asarray(target_cols, dtype=int)
        x[:, selected] = x[:, selected] * outlier_scale + rng.normal(
            0.0, 0.5, size=(rows, len(selected))
        )
        return x.astype(np.float32)

    if outlier_fraction > 0.0:
        n_outliers = max(1, int(round(cols * outlier_fraction)))
        selected = rng.choice(cols, size=n_outliers, replace=False)
        x[:, selected] = x[:, selected] * outlier_scale + rng.normal(
            0.0, 0.5, size=(rows, n_outliers)
        )
    return x.astype(np.float32)


def top_k_mask(scores: np.ndarray, budget_cols: int) -> np.ndarray:
    """Return a mask selecting the top-``budget_cols`` columns by score.

    The score is the maximum absolute value of each feature column. This matches the
    blog’s prototype idea: a column is considered important when it contains large
    activations, and the precision budget is spent on those columns.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if budget_cols <= 0:
        return np.zeros(scores.shape[0], dtype=bool)
    if budget_cols >= scores.shape[0]:
        return np.ones(scores.shape[0], dtype=bool)

    idx = np.argsort(scores)[-budget_cols:]
    mask = np.zeros(scores.shape[0], dtype=bool)
    mask[idx] = True
    return mask


def dynamic_mask(x: np.ndarray, budget_cols: int) -> np.ndarray:
    """Pick the high-precision columns from the current batch itself."""
    return top_k_mask(np.max(np.abs(x), axis=0), budget_cols)


def static_mask(calibration: np.ndarray, budget_cols: int) -> np.ndarray:
    """Pick the high-precision columns from a separate calibration batch."""
    return top_k_mask(np.max(np.abs(calibration), axis=0), budget_cols)


def cado_with_mask(x: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply CADO using a precomputed outlier mask with a fixed precision budget."""
    x = np.asarray(x, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    result = np.zeros((x.shape[0], weights.shape[1]), dtype=np.float32)

    if np.any(mask):
        result += x[:, mask] @ weights[mask, :]

    normal_mask = ~mask
    if np.any(normal_mask):
        x_normal = x[:, normal_mask]
        w_normal = weights[normal_mask, :]

        x_q, scale_x = quantize_int8(x_normal)
        w_q, scale_w = quantize_int8(w_normal)
        x_hat = dequantize_int8(x_q.astype(np.int8), scale_x)
        w_hat = dequantize_int8(w_q.astype(np.int8), scale_w)
        result += x_hat @ w_hat

    return result


def uniform_int8_baseline(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return a full-matrix int8 baseline without any outlier routing."""
    x_q, x_scale = quantize_int8(x)
    w_q, w_scale = quantize_int8(weights)
    x_hat = dequantize_int8(x_q.astype(np.int8), x_scale)
    w_hat = dequantize_int8(w_q.astype(np.int8), w_scale)
    return x_hat @ w_hat


def run_case(
    mode: str,
    rng: np.random.Generator,
    budget_fractions: list[float],
    outlier_fraction: float = 0.05,
    rows: int = 128,
    cols: int = 256,
) -> list[dict[str, float | int | str]]:
    """Run one equal-budget experiment under either a persistent or moving outlier regime.

    The important comparison is that both policies are evaluated with the same
    high-precision budget and the same evaluation batch.
    """
    weights = rng.normal(0.0, 1.0, size=(cols, 64)).astype(np.float32)
    rows_out: list[dict[str, float | int | str]] = []

    if mode == "persistent":
        target_cols = rng.choice(cols, size=max(1, int(round(cols * outlier_fraction))), replace=False)
        calibration_x = make_outlier_matrix(
            rows=rows,
            cols=cols,
            rng=rng,
            outlier_fraction=0.0,
            target_cols=target_cols,
        )
        eval_x = make_outlier_matrix(
            rows=rows,
            cols=cols,
            rng=rng,
            outlier_fraction=0.0,
            target_cols=target_cols,
        )
    elif mode == "moving":
        calibration_target_cols = rng.choice(
            cols,
            size=max(1, int(round(cols * outlier_fraction))),
            replace=False,
        )
        calibration_x = make_outlier_matrix(
            rows=rows,
            cols=cols,
            rng=rng,
            outlier_fraction=0.0,
            target_cols=calibration_target_cols,
        )
        eval_target_cols = rng.choice(
            cols,
            size=max(1, int(round(cols * outlier_fraction))),
            replace=False,
        )
        eval_x = make_outlier_matrix(
            rows=rows,
            cols=cols,
            rng=rng,
            outlier_fraction=0.0,
            target_cols=eval_target_cols,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode!r}")

    ref = fp32_gemm(eval_x, weights)
    uniform_est = uniform_int8_baseline(eval_x, weights)
    uniform_error = relative_error(ref, uniform_est)

    for frac in budget_fractions:
        budget_cols = max(0, min(cols, int(round(cols * frac))))
        dynamic_mask_value = dynamic_mask(eval_x, budget_cols)
        static_mask_value = static_mask(calibration_x, budget_cols)

        dynamic_est = cado_with_mask(eval_x, weights, dynamic_mask_value)
        static_est = cado_with_mask(eval_x, weights, static_mask_value)

        dynamic_err = relative_error(ref, dynamic_est)
        static_err = relative_error(ref, static_est)

        rows_out.append(
            {
                "mode": mode,
                "budget_fraction": float(frac),
                "budget_cols": int(budget_cols),
                "dynamic_relative_error": float(dynamic_err),
                "static_relative_error": float(static_err),
                "uniform_int8_relative_error": float(uniform_error),
            }
        )

    return rows_out


def main() -> None:
    """Run the equal-budget dynamic-vs-static comparison for multiple budget fractions."""
    rng = np.random.default_rng(21)
    budget_fractions = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    rows: list[dict[str, float | int | str]] = []

    for mode in ["persistent", "moving"]:
        rows.extend(run_case(mode=mode, rng=rng, budget_fractions=budget_fractions))

    output_path = Path(__file__).resolve().parents[1] / "results" / "equal_budget_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        fieldnames = [
            "mode",
            "budget_fraction",
            "budget_cols",
            "dynamic_relative_error",
            "static_relative_error",
            "uniform_int8_relative_error",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("saved_csv=", output_path)
    for mode in ["persistent", "moving"]:
        subset = [r for r in rows if r["mode"] == mode]
        print(f"\nmode={mode}")
        for row in subset:
            print(
                f"budget={row['budget_fraction']:.2f} "
                f"dyn={row['dynamic_relative_error']:.6f} "
                f"static={row['static_relative_error']:.6f} "
                f"uniform={row['uniform_int8_relative_error']:.6f}"
            )


if __name__ == "__main__":
    main()
