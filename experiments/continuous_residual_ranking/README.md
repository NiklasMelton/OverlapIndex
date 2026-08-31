# Continuous residual-only ranking experiment

This frozen exploratory factorial follows the first continuous Fisher ranking
experiment. It tests the specific hypothesis that diagonal within-target-cell
scaling and residual nuisance deflation are useful, while the final Fisher
target-cell mean-contrast projection destroys nonlinear target geometry.

The full grid contains 3 seeds, 8 relationship families, 8 candidate
representations, 12 continuous-OI configurations, and the tuned linear-probe
baseline. Every panel uses one shared latent/noise bank across all candidates.
All target-derived geometry is learned inside cross-fitting folds.

The best-head envelope contains tuned linear Ridge, quadratic Ridge, kNN, and
RBF-SVR reference heads. Overall metrics are interpretable only if the frozen
head-diversity gate passes; this prevents a mostly linear envelope from being
reported as broad nonlinear evidence.

Run with one-thread controls:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
CONTINUOUS_OI_COMMIT=$(git rev-parse --short HEAD) \
PYTHONPATH=. python experiments/continuous_residual_ranking/run.py \
  --output artifacts/continuous_residual_ranking
```

The runner writes atomic per-panel checkpoints and refuses to overwrite a
terminal artifact. Numeric artifacts are intentionally ignored by git.
