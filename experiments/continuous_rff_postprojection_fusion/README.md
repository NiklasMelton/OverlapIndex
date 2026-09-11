# Scalable post-projection RFF fusion

The low-rank bagging experiment showed that averaging eight independently
fitted RFF64 selector scores was accurate and stable, while pooling 4x64 raw
random features before the supervised projection failed. The likely useful
unit is therefore the independently regularized target projection—not the raw
random-feature block.

This experiment tests a scalable approximation:

1. fit each RFF64 map and its rank-8 target-moment projection independently;
2. combine the resulting low-dimensional blocks with equal energy;
3. balance the combined nonlinear view 50/50 with one unchanged SAVE view; and
4. fit one fused prototype selector.

Four-map and eight-map versions are measured. The frozen primary comparison is
the eight-map single-fit SAVE score against a directly remeasured eight-model
SAVE score bag. Their execution order is counterbalanced per candidate, and
the recomputed score bag must exactly match the prior hash-bound scores.

Run with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
PYTHONPYCACHEPREFIX=/tmp/oi_structured_pycache \
python -m experiments.continuous_rff_postprojection_fusion.run
```

This remains a development-only scalability diagnostic and cannot promote a
method.
