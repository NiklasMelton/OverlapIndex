# Continuous Fisher ranking experiment

This frozen exploratory experiment compares the opt-in continuous
FusedKMeans/shared-Fisher stack against legacy Continuous OI and isolated
metric ablations. Each method ranks the same synthetic representation panels.
The reference rank is the held-out best-head envelope across tuned Ridge,
quadratic Ridge, kNN, and RBF-SVR regressors.

Run from the repository root:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
LOKY_MAX_CPU_COUNT=1 python \
  experiments/continuous_fisher_ranking/run.py \
  --output artifacts/continuous_fisher_ranking
```

The runner writes one atomic checkpoint per seed/family panel, followed by
`raw_results.json` and `summary.json`. Existing terminal outputs are never
overwritten.
