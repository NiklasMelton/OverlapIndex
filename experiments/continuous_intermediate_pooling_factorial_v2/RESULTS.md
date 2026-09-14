# Intermediate-layer × pooling factorial V2 result

## Decision

The frozen reference-only factorial completed and nominated exactly one view:
`early_mean`.  No OI outcome was run on these data.  The next authorized stage
is a fresh OI benchmark using the nominated view; same-outcome OI evaluation
remains prohibited.

The nomination rule required at least three of five material linear-to-envelope
rank conflicts and at least three of five material nonlinear-head gains on
**both** targets.  `early_mean` passed that rule on both targets:

| Target | View | Conflicts | Gains | Mean LP rho | Mean LP top-1 regret | Mean max nonlinear gain |
|---|---|---:|---:|---:|---:|---:|
| JHU Crowd log-count | early mean | 3/5 | 5/5 | 0.780 | 0.0079 | 0.0626 |
| SUNRGBD log-depth | early mean | 3/5 | 5/5 | 0.620 | 0.0170 | 0.0625 |

`early_mean_std` passed on SUNRGBD but not JHU, where it produced only one
conflicting draw.  None of the middle or late views produced a rank-conflict
draw on either target, even though most retained material nonlinear gains.

## What the factorial teaches us

The earlier instruction to test intermediate layers together with pooling was
not obsolete.  The original audit covered only three backbones and one draw;
this experiment expanded the comparison to five backbones and five frozen
draws per target.  It confirms that the useful benchmark condition is specific
to early representations rather than an incidental property of one sample.

Depth is the dominant factor.  At middle and late depth, the linear-probe
ranking was nearly identical to the best-head-envelope ranking: mean Spearman
correlations were 0.96--1.00 and mean top-one regret was exactly zero for every
pooling choice.  These representations can still benefit from a nonlinear
head, but the improvement does not reorder the backbone candidates.  They are
therefore poor tests of whether a continuous transferability score can improve
on linear-probe ranking.

Adding per-channel spatial standard deviation is not a universal improvement.
On JHU it reduced the early-view conflict count from 3/5 to 1/5 and reduced the
mean nonlinear gain by 0.0082 R2.  On SUN it improved mean linear-to-envelope
rank correlation from 0.62 to 0.88 and reduced mean top-one regret, while the
conflict count remained 3/5.  This makes `mean_std` less useful for the intended
stress test, not a better general representation choice.  The most defensible
frozen view is therefore simple global mean pooling at the early layer.

Together with the post-outcome `early_mean` stability decomposition, the result
supports a precise interpretation: candidate-dependent nonlinear uplift is
mostly stable, but top-rank reversal is visible only before later network
layers make candidate orderings linearly recoverable.  The benchmark should
target that early regime rather than merely search for large nonlinear gains.

## Execution and validation

- Two real-image regression targets, five backbones, three depths, two pooling
  choices, and five frozen draws produced 60 matrices, 300 fitted candidate
  cells, and 60 ranking panels.
- Extraction completed for all ten target/backbone bundles (30,000 image
  passes).  Archived overlapping parent rows were reused exactly where
  available; newly extracted early-mean rows matched the recent full cache to
  within approximately `5.72e-06` maximum absolute error.
- All artifacts validate through the frozen validator, and no model warning
  types were recorded.
- Public OverlapIndex and Vertebrae behavior were not modified.

## Artifact identity

- Implementation commits: `a572896` and parity correction `1f56f97`
- Protocol SHA-256:
  `4c773cf461230ad7c46b3e7ccc3f8df72c7692cb5ee3245764aea3d0a29bd84f`
- Embedding manifest SHA-256:
  `dc39f737f65125f7c7d08861cc6b7ab0f95abcb47763b113d1ff10d2435451e9`
- Raw-results SHA-256:
  `d7ef71fb60f242cf40f224b1ad3d9923d53d0fe5259d825c9a1223c11ee1451f`
- Summary SHA-256:
  `d0a37415919164a4c92aa21b903f79248caff431283fa44f139b1f657d507e47`
- Manifest SHA-256:
  `0ae7687cfd3fc7ea9480543f45999146d38158cad261cb4260505afe7bd0ccdd`
- Terminal SHA-256:
  `b8f28a873231989bb78fe19b60853cbded012918e33ef23e46abd60931a19fe2`

## Recommendation

Freeze `early_mean` as the qualifying representation recipe for the next
continuous-OI benchmark.  Do not tune on these completed outcomes and do not
run OI post hoc on these matrices.  Instead, obtain a fresh target or untouched
evaluation split with the same five backbones and early global-mean features,
then compare the predeclared continuous-OI candidate against the linear probe
using best-head-envelope rank correlation and top-one regret.  Retain
`early_mean_std`, middle, and late views as negative controls.
