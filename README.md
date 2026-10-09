# CADO

CADO stands for Cache-Aware Dynamic Outlier Offloading.

This project studies whether a small, activation-aware subset of values can be routed through a higher-precision path while the majority of the computation remains in a cheaper low-precision representation. The goal is to improve the quality–latency tradeoff under a fixed fast-memory or cache budget.

This repo is set up for a blog post I wrote on CADO: [Precision on a Budget](https://shubhojit.github.io/writing/precision-on-a-budget/).

## Motivation

Uniform low-precision inference is efficient, but it is not always optimal. In transformer-style workloads, activations often contain a small fraction of very large values, or outliers, that disproportionately affect numerical error. A promising direction is to exploit this structure more deliberately: preserve the values that matter most, while keeping the bulk of the computation cheap.

## Core idea

Under a fixed fast-memory budget, we test whether a dynamic routing policy can:

- identify a sparse set of high-impact activations,
- send those values through a higher-precision path,
- keep the rest on a low-precision path,
- and improve the quality–latency frontier relative to static quantization strategies.

## Research question

Can a limited high-precision budget be spent more effectively by routing activation outliers dynamically rather than applying a uniform low-precision policy across the full computation?

## Scope

This project is intentionally narrow and reproducible. It focuses on:

- controlled GEMM and matmul-style experiments,
- outlier density sweeps,
- threshold sweeps,
- cache-aware tile and budget analysis,
- comparison against uniform low-precision and static mixed-precision baselines.

This repository is intended as a focused research codebase for a controlled study of cache-aware dynamic outlier routing. It is not a general-purpose production inference framework.

## Repository structure

```text
cado/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── cado_reference.py
│   ├── cado_router.py
│   ├── cado_quant.py
│   ├── cado_runtime.py
│   └── __init__.py
├── experiments/
│   ├── 01_correctness.py
│   ├── 02_outlier_density.py
│   ├── 03_baseline_comparison.py
│   ├── 04_threshold_sweep.py
│   └── 05_equal_budget_comparison.py
├── analysis/
│   └── first_results.ipynb
├── results/
│   ├── baseline_comparison.csv
│   ├── equal_budget_comparison.csv
│   ├── outlier_density.csv
│   └── threshold_sweep.csv
└── docs/
    └── proposal.md
```

## Planned experiments

1. Correctness: compare FP32 reference against CADO decomposition.
2. Outlier density sweeps: understand how sensitive the method is to the frequency of outliers.
3. Threshold sweeps: determine how routing policy affects accuracy and runtime.
4. Cache-aware budget analysis: study how fast-memory constraints affect the tradeoff.
5. Baseline comparison: compare against uniform INT8 and static mixed-precision baselines.
6. Dynamic routing analysis: check whether the method meaningfully improves the quality–latency frontier.

## Expected outcome

A convincing result would show that, under a fixed fast-memory budget, CADO yields a better quality–latency tradeoff than a uniform low-precision strategy for a meaningful regime of outlier density and matrix shapes.

## Local setup

```bash
cd /path/to/cado
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Current status

The repo currently includes a small set of numerical experiments validating the prototype idea and a fixed-budget comparison. The work remains a focused research scaffold rather than a full end-to-end acceleration stack.
