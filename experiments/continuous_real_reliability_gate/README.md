# Continuous-OI reliability gate

This developmental experiment tests whether observable OI diagnostics can decide when the broad ranking should be left alone and when the fixed 50/50 broad/SAVE reranking should be used inside the broad top-10 pool.

The gate has four closed features: broad/SAVE rank concordance, complete-top-five Jaccard overlap, the normalized broad top-five boundary margin, and mean broad score. It is a fixed ridge model trained on the other dataset group. Head outcomes are used only as training labels and held-out evaluation metrics; they are never inference features.

An oracle per-panel gate is reported only as a diagnostic upper bound.

```bash
PYTHONPATH=. LOKY_MAX_CPU_COUNT=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/niklasmelton/Library/Caches/pypoetry/virtualenvs/vertebrae-XeUus8FY-py3.9/bin/python \
  -m experiments.continuous_real_reliability_gate.run
```
