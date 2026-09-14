# Continuous intermediate-layer and pooling audit V1

The cached terminal embeddings contained much more predictive information than
PCA-32 retained, but even exact smooth nonlinear heads preserved the linear
probe's backbone ordering.  This development-only audit tests the next plausible
bottleneck: global terminal pooling.

For DINOv2-Small, ResNet-50, and ConvNeXt-Tiny, it extracts three predeclared
relative depths in one model pass.  Each depth produces a global mean and a
mean-plus-standard-deviation summary over patch or spatial units.  The exact
previously cached terminal vector is a seventh aligned control.  The frozen
600-row draw and 1,000-row evaluation cohort are reused for JHU Crowd and SUN
RGB-D.

Linear Ridge and exact precomputed-kernel RBF-SVR are tuned only within training
folds in the full representation space.  A stable ranking conflict requires
replication across targets or across more than one nonterminal view; a single
weak reversal is not sufficient.  No OverlapIndex method or reserved backbone is
evaluated, and this diagnostic cannot promote a method.

Run extraction and outcomes separately from a clean committed tree:

```bash
PYTHONPATH=. python -m experiments.continuous_intermediate_pooling_audit_v1.extract
PYTHONPATH=. python -m experiments.continuous_intermediate_pooling_audit_v1.run
```

