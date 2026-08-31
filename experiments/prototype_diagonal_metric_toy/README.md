# Prototype-margin diagonal metric toy

This follow-up keeps the completed class-mean diagonal toy unchanged. It uses
a disjoint internal split within each selector training fold to fit provisional
class-owned prototypes and estimate correct-versus-impostor feature margins.

Run:

```bash
PYTHONPATH=. pytest -q tests/test_prototype_diagonal_metric_toy.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
PYTHONPATH=. python -m experiments.prototype_diagonal_metric_toy.experiment \
  --output artifacts/prototype_diagonal_metric_toy/run_v1
```

The result is development-only and cannot establish a real-backbone runtime or
product claim.
