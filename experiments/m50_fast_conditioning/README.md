# M50 fast-conditioning probe

This is a post-outcome, outcome-neutral performance follow-up. It does not
modify the frozen `m50_backbone_ranking` package, candidate definitions, or
artifacts.

The probe compares:

1. canonical stable-sort MAD with dense diagonal state;
2. exact partition MAD with the same dense state; and
3. exact partition MAD with vector-only diagonal state.

Run the small semantic tests:

```bash
PYTHONPATH=. pytest -q tests/test_m50_fast_conditioning.py
```

Run one actual maximum-budget Food fold under the frozen thread controls:

```bash
VECLIB_MAXIMUM_THREADS=1 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
BLIS_NUM_THREADS=1 \
LOKY_MAX_CPU_COUNT=1 \
PYTHONPATH=. python -m experiments.m50_fast_conditioning.benchmark \
  --model resnet50 \
  --arm baseline \
  --budget 640 \
  --folds 1 \
  --repeats 2 \
  --output artifacts/m50_fast_conditioning/resnet50_b640.json
```

The runtime projection reports point-estimate wall ratios only. A promising
projection still requires a fresh counterbalanced benchmark with paired
bootstrap intervals and fresh-process peak memory before it can satisfy the
frozen resource gates.
