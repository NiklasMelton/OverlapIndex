# Continuous cached-embedding information audit V1

This development-only experiment asks whether the prior real-source screen hid a
nonlinear backbone-ranking opportunity through one of three evaluator choices:
PCA-32 compression, removal of embedding magnitude by row normalization, or a
600-row training budget.

It reuses the immutable JHU Crowd and SUN RGB-D embedding caches.  DINOv2-Small,
ResNet-50, and ConvNeXt-Tiny provide three predeclared architecture families.  The
archived 600-row, row-L2, PCA-32 result remains the reference.  Eight new views
cross 600 versus 2,000 training rows, PCA-32 versus PCA-128 versus no PCA, and—at
the larger budget—row normalization versus preserved magnitude.

Every transformation and every head choice is fitted only on an inner training
fold.  The audit uses linear Ridge, exact k-nearest-neighbor regression, and a
small Extra Trees grid.  Extra Trees is intentionally a tractable nonlinear
diagnostic at the original embedding dimension; it is not a proposed production
head or a replacement for the already-qualified four-head library.

This experiment evaluates no OverlapIndex method, no reserved backbone, and no
distribution shift.  It cannot promote a method.  A material dimension, sample,
or magnitude effect nominates a repeated five-backbone/multiple-draw study.  If
none is material, the next diagnostic is to extract intermediate layers and
alternative pooling views, because the terminal global vectors—not the head
library—then become the more plausible bottleneck.

Run from a clean committed tree with the project environment:

```bash
PYTHONPATH=. python -m experiments.continuous_cached_embedding_information_audit_v1.run
```

Resume a partial immutable checkpoint with `--resume`; validate a completed
artifact with `--validate-only`.

