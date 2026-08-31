# Supervised diagonal metric toy

This research-only toy compares current raw OI, a bounded supervised diagonal
metric used consistently by OI, and a cross-fitted linear probe. It uses paired
synthetic backbone panels and independent downstream-reference banks.

The metric is fitted on each selector training fold only. No public
`OverlapIndex` behavior or API is changed.

Run focused verification:

```bash
PYTHONPATH=. pytest -q tests/test_diagonal_metric_toy.py
```

Run the frozen toy:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
PYTHONPATH=. python -m experiments.diagonal_metric_toy.experiment \
  --output artifacts/diagonal_metric_toy/run_v1
```

This is a development smoke screen. It cannot establish a product or
real-backbone runtime claim.
