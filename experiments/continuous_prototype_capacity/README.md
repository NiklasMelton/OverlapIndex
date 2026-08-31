# Continuous prototype-capacity experiment

This experiment changes only the number of FusedKMeans prototypes per target
cell, pairing `k=4` and `k=8` for SAVE, RFF-mean, SAVE+RFF-mean, and the
residual three-block metric. It tests the frozen hypothesis that four
prototypes cannot represent the triangle wave's approximately six separated
inverse branches per target value.

Every fold records resolved/occupied prototypes, empty prototypes, support,
and runtime. Geometry, folds, target cells, random seeds, RFF bandwidth,
aggregation, and reference heads are unchanged. No public OI behavior changes.

Focused verification:

```bash
PYTHONPATH=. pytest -q tests/test_continuous_prototype_capacity.py
```

After committing the frozen implementation:

```bash
PYTHONPATH=. CONTINUOUS_OI_COMMIT=$(git rev-parse --short HEAD) \
  python -m experiments.continuous_prototype_capacity.run \
  --output artifacts/continuous_prototype_capacity
```
