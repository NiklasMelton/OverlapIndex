# Early-mean LSP fresh-evaluation result

## Decision

The locked primary `broad_rff_full` method **failed** the frozen gate.  It passed
the runtime check but failed both ranking noninferiority checks and both
nonlinear-advantage checks.  No secondary method replaces it, and no method is
promoted.

The benchmark itself was informative.  `early_mean` passed the frozen
opportunity gate with material nonlinear gain on 5/5 draws and a diagnostic
linear-to-envelope rank conflict on 4/5 draws.  Thus the failure is not caused
by accidentally choosing another linearly trivial representation.

| Early-mean method | Mean rho | Mean top-1 regret | Exact best | Material-pair accuracy | Relative time |
|---|---:|---:|---:|---:|---:|
| Linear CV | 0.660 | 0.0022 | 0.800 | 0.869 | 1.00x |
| Four-head CV maximum | 0.600 | 0.0046 | 0.600 | 0.811 | 12.78x |
| Observed continuous OI | 0.120 | 0.0076 | 0.400 | 0.547 | 1.31x |
| Locked broad RFF | 0.360 | 0.0113 | 0.200 | 0.694 | 0.42x |
| SAVE+RFF | -0.400 | 0.0198 | 0.000 | 0.306 | 7.50x |

## What happened

The fresh evaluation confirms a distinction that the layer-selection screen
could not establish.  A representation may expose strong nonlinear downstream
gain without making the best-backbone ordering identifiable by a particular
unsupervised or weakly supervised geometry score.

RBF-SVR was the held-out winner in 21 of 25 early-mean backbone/draw cells and
quadratic kernel Ridge won the other four.  Nevertheless, linear CV selected
the held-out envelope winner on four of five draws.  The nonlinear uplift was
large for several weaker or near-leading candidates, but usually did not make
the linear probe choose a materially bad backbone.

The five-draw mean envelope scores were 0.2465 (DINOv2), 0.2413 (DeiT), 0.2312
(MobileNetV3), 0.2552 (ResNet-50), and 0.2556 (ConvNeXt).  The top two candidates
therefore differed by only 0.0004 R2 after averaging.  Per-draw top gaps ranged
from 0.0022 to 0.0124 R2, and pairwise Spearman agreement among the five
held-out envelope rankings ranged from -0.4 to 0.9.  This is a real nonlinear
task, but its top ordering is narrow and training-draw dependent.

Broad RFF's failure is mainly per-draw stability rather than complete loss of
the average candidate signal.  Averaging its five candidate scores after the
fact gives Spearman 0.90 against the five-draw mean envelope, the same aggregate
correlation as linear CV.  That observation is diagnostic only: repeated-draw
averaging was not the locked deployment rule, would change the estimator and
its runtime, and cannot be promoted from this holdout.

## Layer and pooling controls

The negative controls show why reference-only view selection is insufficient.
`early_mean` maximized the opportunity to disagree with a linear probe, but it
was not the view on which an OI-style score ranked best in this target.  For
example, observed OI on the predeclared `middle_mean_std` control reached 0.0057
mean top-one regret and 0.80 exact-best rate, versus 0.0227 and 0.20 for linear
CV on that same view.  This is not eligible for selection after outcome review,
but it demonstrates that two properties must be qualified separately:

1. the view contains a stable linear-to-best-head ranking opportunity; and
2. the proposed ranking score is aligned with candidate utility in that view.

The previous factorial established the first property for `early_mean`; this
holdout rejects the assumption that it automatically supplies the second.

## Runtime

The speed-focused prototype-free implementation worked.  Its broad score
matched the shared reference path exactly in every cell and required only 0.42
times the aggregate early-mean linear-CV runtime.  Speed is therefore no longer
the limiting issue for this candidate at 300 rows.  Observed OI was 1.31x the
linear probe, while SAVE+RFF was 7.50x and became much slower on wide late-layer
controls.

## Integrity and validation

- The 300 evaluation images have zero overlap with the 200 LSP wrist-span
  evaluation images used previously.  They were deterministically selected
  from 384 unused valid official-test images.
- Five 300-row draws came from the frozen 590-row training pool.
- Five backbones and six views produced 30 matrices, 150 outcome cells, 750
  method scores, 150 ranking rows, and 1,200 prototype-based OI fold fits.
- The best-head envelope included linear Ridge, quadratic kernel Ridge,
  distance-weighted kNN, and exact RBF-SVR.
- All outcome artifacts terminal-validate and recorded no model warning types.
- Public OverlapIndex and Vertebrae behavior were not modified.

## Artifact identity

- Implementation commit: `9f7ccad097d4104f1a227e6d2362302b866fd6c7`
- Pre-outcome seed correction: `8d46dba`
- Protocol SHA-256:
  `2ca43b266f7e68a93ef635bdb7cbd595257cfb1a66afae5071a2c4b8dca2abb0`
- Prepared cohort SHA-256:
  `37ad4d05a78f4c03a7550571e037ff8c9d26506a29892a490916a7baf4063dcc`
- Prepared input manifest SHA-256:
  `7a5f8a2a070bbdf8a36b3988a4453a22a4e2deb83ad67bd66eb9adb2d22a7183`
- Embedding manifest SHA-256:
  `a7fe9295d1409f3e79440b8c990e428ea16432f7e202844165dbb8ac274fb378`
- Raw-results SHA-256:
  `59858a7d1f04853230d7e921a844791a796aa3825cea6dbcd4968f7f7e24615a`
- Summary SHA-256:
  `cbadd527f1ebd19ce75be8a3c53e8023e15ca727baefdd311414047c02339641`
- Manifest SHA-256:
  `3b67343210445d04842376dcbe022bdc363efccab1f75e165773649c71fe344c`
- Terminal SHA-256:
  `7aae93e07268f151f03110a447d2e01292bd39e9ceebfa5c15681833e9f8f30a`

## Recommendation

Do not tune `broad_rff_full` or switch to `middle_mean_std` on this holdout.
The next experiment should separate reference stability from score alignment
prospectively.  On development data only, test a fixed stability intervention
for the broad score—such as a predeclared multi-map or fold ensemble—and require
it to improve per-draw repeatability while retaining the now-satisfactory
runtime.  Then freeze both the representation and scoring rule before another
untouched evaluation.  A wholly new dataset remains necessary before any
integration decision.
