# Results: COCO pose opportunity replication

Status: completed prospectively under protocol SHA-256
`88868969b723d5baf16f46b018d62b283ee54482b61318c8100330adeb1199e6`.
The COCO dataset gate and combined LSP-plus-COCO gate both failed. Reserved
backbones and OverlapIndex methods were not evaluated.

## Result

- `torso_axis_sine`, the direct replication of the admitted LSP target, was
  unrecoverable at the frozen budget: median best-head R2 0.0638. Its median
  envelope span was 0.0666, but material nonlinear prevalence was only 0.133
  and LP selected an envelope-optimal backbone in every replicate.
- `torso_axis_cosine` was also unrecoverable (median best-head R2 -0.0071) and
  had insufficient envelope spread (0.0441).
- `bilateral_knee_extension_cosine` barely cleared recoverability (median
  best-head R2 0.1021) and had useful envelope spread (0.0936), but it was an
  LP-dominated control: linear ridge uniquely won 14/15 candidate rows, with
  zero LP winner changes or regret.

## Interpretation

The LSP torso opportunity does not replicate merely by transferring the same
geometric target to independently sourced person-pose images. The current
evidence therefore contains one new real opportunity dataset, not a reusable
two-dataset stratum. Continuing to inspect alternate COCO targets or relaxing
the gates would be post-outcome dataset fishing. The frozen decision is to stop
before reserved-backbone or ranker-method evaluation and revisit how candidate
tasks are defined or acquire a genuinely predeclared benchmark collection.

Canonical artifacts are in
`artifacts/continuous_coco_pose_opportunity_replication/results/`.
