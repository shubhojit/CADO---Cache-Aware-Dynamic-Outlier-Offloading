from __future__ import annotations

"""First experiment: sanity-check that the CADO decomposition stays close to FP32."""

import numpy as np

from src.cado_reference import cado_reference, fp32_gemm, relative_error


def main() -> None:
    """Generate a synthetic batch of activations with a few extreme outlier columns.

    The synthetic setup creates a matrix whose general distribution is near-Gaussian but
    includes a sparse set of unusually large activations. These values are intended to
    mimic the outlier columns that a routing-based mixed-precision algorithm should
    isolate.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(128, 256)).astype(np.float32)
    weights = rng.normal(0.0, 1.0, size=(256, 64)).astype(np.float32)

    # Inject a sparse set of larger values to simulate a small number of extreme
    # activation columns. These are the columns that the outlier mask should detect.
    x[0, :8] *= 20.0
    x[1, 10:18] *= 25.0

    ref = fp32_gemm(x, weights)
    approx = cado_reference(x, weights, threshold=6.0)
    err = relative_error(ref, approx)

    print(f"reference_shape={ref.shape}")
    print(f"approx_shape={approx.shape}")
    print(f"relative_error={err:.6f}")


if __name__ == "__main__":
    main()
