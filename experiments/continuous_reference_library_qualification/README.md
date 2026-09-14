# Continuous reference-library qualification

This development-only stage asks whether a stronger, scale-aware four-head
library is adequate to define the downstream backbone-ranking target. It does
not evaluate OverlapIndex and it does not open reserved backbones or data.

The experiment reuses the corrected V2 evaluator's unique identities,
fold-local row normalization/PCA-32/scaling, and fixed 200-row evaluation
cohorts. It changes only the head recipes:

- linear and degree-two Ridge use a dimensionless mean-loss penalty;
- kNN searches both weighting rules and a wider neighbor range;
- RBF-SVR standardizes the target within every training fold and expresses its
  bandwidth relative to the training-only median pair distance.

The grid was frozen before outcomes. Endpoint selections are acceptable only
when their advantage over the best interior setting is below the declared
material threshold; otherwise the reference grid remains unqualified.

Run with one numerical thread:

```text
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
python -m experiments.continuous_reference_library_qualification.run
```

