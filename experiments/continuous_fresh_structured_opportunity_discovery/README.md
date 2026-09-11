# Fresh structured continuous opportunity discovery

This is a prospective, development-only screen for real image tasks where the
best held-out head family changes backbone rankings relative to a linear probe.
It does **not** evaluate or select an OverlapIndex variant.

The protocol was frozen before predictive outcomes. It uses two new datasets:

- CUB-200-2011: beak-to-tail orientation and normalized span, using only images
  with both parts visible and the official train/test split.
- Leeds Sports Pose (LSP): torso orientation and bilateral knee extension,
  using target-specific visible-joint cohorts and the standard first/last
  1,000-image split.

Both are discovery datasets. CUB explicitly warns that some images overlap
ImageNet, and pretraining overlap cannot be excluded for LSP. Neither dataset
may be used as untouched confirmation evidence.

## Stages

1. Acquire the exact raw sources and write a content-addressed input manifest.
2. Prepare deterministic, disjoint training replicates and fixed evaluation
   cohorts without fitting embeddings or heads.
3. Extract the five frozen scout backbones.
4. Evaluate the frozen linear, quadratic, kNN, and RBF head grids.
5. Admit a dataset only if at least one target satisfies every opportunity
   gate in `protocol.json`. Do not inspect the reserved backbones in this stage.

No threshold, target, cohort, head grid, or backbone may be changed after head
outcomes are observed. A later method comparison requires a new frozen protocol
and at least two admitted datasets.
