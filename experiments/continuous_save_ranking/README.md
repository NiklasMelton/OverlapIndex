# Continuous SAVE ranking experiment

This frozen factorial tests a single covariance-aware target geometry before
considering score-level hybrids. Every target-derived transform is fitted
inside each cross-fitting training fold.

The target-moment implementations are experiment-local. Existing public OI
behavior is unchanged. The compared views are observed geometry, residual
correction, full Fisher, mean-only SDR, SAVE covariance SDR, fixed equal-weight
mean+SAVE SDR, and a privileged quadratic-lift ceiling.

The 3-seed × 6-family × 4-quality × 4-nuisance grid independently varies
predictive signal and nuisance. Overall interpretation requires the frozen
reference heads to clear per-panel performance floors and nonlinear-head win
counts.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
CONTINUOUS_OI_COMMIT=$(git rev-parse --short HEAD) PYTHONPATH=. \
python experiments/continuous_save_ranking/run.py \
  --output artifacts/continuous_save_ranking
```

The runner writes atomic panel checkpoints and refuses to overwrite terminal
artifacts.
