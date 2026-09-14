# Continuous OI mechanism diagnostic v1

This experiment isolates a question that real backbone panels cannot answer
cleanly: does continuous OI reward information that improves the conditional
mean, or can it reward target-dependent distributions that do not improve
regression prediction?

Each family uses the same three noisy measurements of a latent scalar with
quality `q = 0.95, 0.65, 0.35`, plus seven irrelevant coordinates. The fixed
independent evaluation draw supplies an analytic conditional-mean reference:

- `linear_mean`: `Y = U + epsilon`;
- `quadratic_mean`: `Y = U^2 - 1 + epsilon`;
- `periodic_mean`: `Y = sin(U) + epsilon`;
- `variance_only`: `Y = (0.4 + 0.8 |U|) epsilon`, so `E[Y|X] = 0` while the conditional variance changes;
- `null_noise`: target and representation are independent.

The comparison is frozen before outcomes. `observed_oi` is current observed
continuous OI, `save_oi` is the existing SAVE target-moment OI diagnostic, and
`broad_rff` is the locked prototype-free null-adjusted multibandwidth RFF
kernel-alignment score. Every method sees identical rows and fixed folds.

Algorithmic variance changes only selector seeds on one training sample. Sampling
variance changes the training sample while holding the evaluation draw, folds,
and selector seed fixed. Systematic misalignment is tested against the analytic
conditional-mean R2 and against the equal-utility variance/null controls.

Run with the archived Python 3.9 environment:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
PYTHONPATH=. python -m experiments.continuous_mechanism_diagnostic_v1.run \
  --output artifacts/continuous_mechanism_diagnostic_v1/results
```

The protocol is frozen before outcome execution. The result note is written
after validation and reports each failure class independently; no method or
parameter is promoted from this diagnostic.
