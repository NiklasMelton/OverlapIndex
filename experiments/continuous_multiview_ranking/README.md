# Continuous block-preserving multi-view experiment

This experiment tests whether complementary geometries can coexist in one OI
metric without score fusion. Raw-space SAVE, RFF kernel-mean, and optionally
observed or residual-corrected local geometry are fitted independently within
each cross-fit training fold. Each block receives equal row norm before fixed
concatenation, followed by one FusedKMeans fit and one continuous OI score.

No block weights are learned or tuned, and no public OI behavior is changed.

Focused verification:

```bash
PYTHONPATH=. pytest -q tests/test_continuous_multiview_ranking.py
```

After committing the frozen implementation:

```bash
PYTHONPATH=. CONTINUOUS_OI_COMMIT=$(git rev-parse --short HEAD) \
  python -m experiments.continuous_multiview_ranking.run \
  --output artifacts/continuous_multiview_ranking
```
