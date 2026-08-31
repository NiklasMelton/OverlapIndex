# Continuous kernel-mean ranking experiment

This follow-up tests the direct use of a characteristic random-feature map:
target-cell conditional means are estimated in a deterministic 64-dimensional
RBF sketch and projected with the existing regularized target-moment machinery.
A fixed trace-normalized mean+SAVE variant tests whether conditional covariance
adds useful information without tuning a mixture weight.

The exact balanced SAVE panel, reference heads, selector backend, and RFF map
are reused. All state is fitted inside each cross-fit training fold. This is an
experiment-local implementation and does not alter public OI behavior.

Focused verification:

```bash
PYTHONPATH=. pytest -q tests/test_continuous_rff_mean_ranking.py
```

After committing the frozen implementation:

```bash
PYTHONPATH=. CONTINUOUS_OI_COMMIT=$(git rev-parse --short HEAD) \
  python -m experiments.continuous_rff_mean_ranking.run \
  --output artifacts/continuous_rff_mean_ranking
```
