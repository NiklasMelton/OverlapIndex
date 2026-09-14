# Replicated early-mean benchmark result

## Decision

The reference gate **failed**, so the frozen continuous-OI comparison did not
run.  The OI runner was invoked only to verify the authorization boundary and
failed closed before creating outcomes.

The original three-backbone, one-draw `early_mean` conflict therefore should
not be treated as a sufficiently stable development benchmark.  Expanding to
five backbones and five frozen 600-row draws preserved strong nonlinear head
gains, but it did not make the linear-probe ranking fail consistently on both
targets.

| Target | Conflicting draws | Material nonlinear-gain draws | Mean LP rho | Mean LP top-1 regret | Gate |
|---|---:|---:|---:|---:|---|
| JHU Crowd log-count | 2/5 | 5/5 | 0.880 | 0.0066 | fail |
| SUNRGBD log-depth | 3/5 | 5/5 | 0.700 | 0.0143 | pass |

The rule required at least three conflicting draws and three material
nonlinear-gain draws on **each** target.  JHU missed by one conflicting draw.
Its individual top-one regrets were 0, 0.0135, 0.0093, 0, and 0.0100 R2.
SUNRGBD had three clear reversals, with top-one regrets 0.0319, 0.0175, and
0.0219 R2 on those draws.

## What the result means

Nonlinearity is present but usually does not change the backbone order.  RBF
SVR was a held-out envelope winner in 47 of the 50 candidate/draw cells, and
every one of the ten panels contained a candidate with at least 0.02 R2
nonlinear gain.  Yet the linear probe still had high rank correlation and
often selected the same best backbone.

This separates two conditions that earlier work had sometimes conflated:

1. a nonlinear head improves prediction on a backbone; and
2. the nonlinear improvement is sufficiently candidate-dependent to reverse
   the backbone ranking.

Only the second condition creates the ranking opportunity needed to test
whether continuous OI beats a linear-probe ranker.  `early_mean` reliably
creates the first condition but only intermittently creates the second.

The added backbones explain part of the change.  DeiT is the mean-envelope JHU
winner and the mean-envelope SUNRGBD winner, while it is also often strong
under the linear probe.  Adding it turns several former three-backbone rank
conflicts into agreement.  This is not evidence that the nonlinear signal
vanished; it is evidence that a strong backbone can be strong under both head
families.

## Extraction and validation

- 10 matrices, 30,000 image passes, five backbones, and two targets completed.
- Existing DINOv2, ResNet50, and ConvNeXt overlap rows reproduced the prior
  cache with maximum absolute deltas from `7.45e-08` to `5.72e-06`, below the
  frozen `5e-05` limit.
- All 50 reference cells and ten ranking panels completed with no model warning
  types.
- The focused implementation and adjacent available tests passed: 6 passed.
- Two separately attempted bounded-evaluator tests remain unavailable in this
  recovered checkout because their archived parent result files are absent;
  they were not used by this experiment.

## Artifact identity

- Implementation commit: `9e0de3b9e35e7f24f1aa2d899684c470bce26342`
- Protocol SHA-256:
  `f2eadae5098b7107b74f5c878be102f98a061b216e1059ad22c7ff2a2c1b69b7`
- Embedding manifest SHA-256:
  `8d0a94f23752c1955cb3508ea4e6946a6727cc616a4c1e536b6483f719efffac`
- Reference raw-results SHA-256:
  `f8ba0da6d652175ce6a2884171e097a41786269e79fd55651a6a88f9866ec6f5`
- Reference summary SHA-256:
  `402874dc11ce665957559abf3524e480c114941d2e3f40e6f50222dfccf8a80c`
- Reference manifest SHA-256:
  `5fd149e60a6b38658145f7088247519efc480d090096002559cf6ad35ca1330f`
- Reference terminal SHA-256:
  `39c932bb30ff9869ad51d9555e1e52f1a90096d1c3dbe24a5e5a18968c75ae7f`

## Recommendation

Do not weaken this gate and do not run OI post hoc on only SUNRGBD.  Preserve
these matrices as a regression resource, but change the next benchmark search
criterion: qualify datasets or representation panels by **stable,
candidate-dependent nonlinear uplift and repeated backbone-rank reversal**, not
by nonlinear head gains alone.

The most efficient next diagnostic is to use the already extracted matrices
to quantify the variance of each candidate's nonlinear uplift across draws and
identify whether reversals arise from stable representation differences or
sampling noise.  Then construct or source a panel with at least two independent
targets for which the same candidate ordering reversal repeats.  Only that
panel should unlock the already implemented full-vs-PCA32 OI comparison.

## Follow-up uplift-stability diagnostic

The recommended decomposition was subsequently completed from the frozen raw
artifact without fitting any new head or OI model.  Nonlinear uplift was
defined per cell as held-out envelope R2 minus held-out linear-Ridge R2.  A
two-way backbone-plus-draw variance decomposition gave:

| Target | Backbone fraction | Draw fraction | Residual interaction |
|---|---:|---:|---:|
| JHU Crowd log-count | 0.680 | 0.161 | 0.158 |
| SUNRGBD log-depth | 0.729 | 0.171 | 0.100 |

Thus the uplift itself is mostly a stable representation property, not noise.
For JHU, mean uplift ranged from 0.0312 R2 for DINOv2 to 0.0661 for ConvNeXt;
each backbone exceeded 0.02 on all five draws.  For SUNRGBD, ConvNeXt again had
the largest and most stable uplift (mean 0.0618, material on 5/5 draws), while
ResNet50 had essentially no material uplift (mean 0.0089, material on 0/5).

What is unstable is whether these differences reorder the narrowly separated
leading backbones.  No JHU candidate pair reversed on more than three of five
draws, and the leading candidate varied among DeiT, DINOv2, and ResNet50.  SUN
showed the same pattern around DeiT, DINOv2, and ResNet50 even though the
ConvNeXt nonlinear uplift was highly stable.  The original gate failure is
therefore best described as *unstable top-rank reversal over stable nonlinear
uplift*, not disappearance of nonlinear information.
