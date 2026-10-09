# CADO

CADO stands for Cache-Aware Dynamic Outlier Offloading.

This repository is a small, focused research project studying whether a sparse set of high-impact activation values can be routed through a higher-precision path while the bulk of the work stays in a cheaper low-precision representation.

## Goal

Test a narrow hypothesis: under a fixed fast-memory or cache budget, dynamic outlier routing can improve the quality–latency tradeoff relative to uniform low-precision execution or a static mixed-precision baseline.

## Repository layout

```text
cado/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── cado_reference.py
│   ├── cado_router.py
│   ├── cado_quant.py
│   └── cado_runtime.py
├── experiments/
│   ├── 01_correctness.py
│   ├── 02_outlier_density.py
│   ├── 03_baseline_comparison.py
│   └── 04_threshold_sweep.py
├── analysis/
│   └── first_results.ipynb
├── results/
│   ├── baseline_comparison.csv
│   ├── outlier_density.csv
│   └── threshold_sweep.csv
```

## Project status

This is a minimal research scaffold. It is intentionally small and designed to support early experiments before broadening the codebase.

## Local setup

```bash
cd /path/to/cado
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Experiments

### 1. Correctness check

```bash
python experiments/01_correctness.py
```

This validates that the decomposition stays close to the full-precision reference on a synthetic matrix with a small set of injected outliers.

### 2. Outlier-density sweep

```bash
python experiments/02_outlier_density.py
```

This sweeps the fraction of activation columns treated as outliers and reports the resulting relative error. In the current toy setup, the error decreases as outlier density increases from 0% to 30%.

### 3. Baseline comparison

```bash
python experiments/03_baseline_comparison.py
```

This compares the full-precision reference against:

- the CADO approximation
- a uniform int8 baseline

This helps isolate whether outlier-aware routing adds value beyond naive low-precision execution.

### 4. Threshold sweep

```bash
python experiments/04_threshold_sweep.py
```

This sweeps the outlier threshold and records CADO and uniform-int8 relative error
along with the fraction of columns routed to the high-precision path. The routing
fraction is a simple cost proxy, not a measured latency or cache-footprint result.

## Current observed result

On the synthetic experiment currently checked in, the prototype performs as follows:

- CADO relative error: 0.003129
- uniform int8 relative error: 0.037619

This indicates that, in this toy setting, CADO materially reduces approximation error relative to a naive uniform low-precision path.

## Notes

This project is intentionally scoped to the early research questions:

- correctness of the decomposition,
- effect of outlier density,
- threshold sensitivity,
- simple baseline comparisons and tradeoff analysis.

It is not yet a general-purpose large-model inference framework.
