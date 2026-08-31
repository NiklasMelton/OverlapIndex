# Continuous RFF-SAVE ranking experiment

This experiment tests one nonlinear continuous-OI geometry rather than a
post-hoc score fusion. It applies a deterministic median-bandwidth RBF random
Fourier sketch inside each cross-fit training fold, then fits the same
regularized rank-8 SAVE transform used by the preceding covariance experiment.

The exact balanced six-family SAVE panel and its four reference heads are
reused. The frozen controls distinguish the effect of the random feature map
alone from RFF followed by SAVE at 64 and 128 features. No public OI behavior
is changed.

Run the focused checks with:

```bash
PYTHONPATH=. pytest -q tests/test_continuous_rff_save_ranking.py
```

After committing the frozen implementation, execute into a new directory:

```bash
PYTHONPATH=. CONTINUOUS_OI_COMMIT=$(git rev-parse --short HEAD) \
  python -m experiments.continuous_rff_save_ranking.run \
  --output artifacts/continuous_rff_save_ranking
```
