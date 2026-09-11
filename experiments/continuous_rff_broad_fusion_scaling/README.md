# Prototype-free broad RFF fusion scaling

The post-projection fusion experiment found that eight independently projected
RFF64 blocks retained useful ranking under the broad kernel score, even though
one shared prototype partition damaged the SAVE score. The broad score did not
use the fitted prototypes, but its measured path still paid for SAVE geometry,
target-cell construction, and a fused prototype fit.

This experiment implements the actual broad-only path. It must reproduce all
45 previously observed broad fused scores at absolute tolerance `1e-12`. It
then measures the broad-only and full fused paths in counterbalanced order and
runs a descriptive synthetic row-scaling benchmark at 200, 500, 1,000, and
2,000 samples against the same full fused path and a fixed linear Ridge CV.

Run with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
PYTHONPYCACHEPREFIX=/tmp/oi_structured_pycache \
python -m experiments.continuous_rff_broad_fusion_scaling.run
```

This is a development-only implementation and scaling diagnostic. It cannot
promote a method.
