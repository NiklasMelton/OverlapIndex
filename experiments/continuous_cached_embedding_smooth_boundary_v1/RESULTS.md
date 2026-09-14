# Cached-embedding information and smooth-head audit results

## Status

Completed and artifact-validated on 2026-09-14.  These experiments are
reference-only development diagnostics.  They evaluate no OverlapIndex method,
no reserved backbone, and promote no method.

## What changed relative to the archived screen

The archived real-source screen used 600 training rows, row-L2 normalization,
PCA-32, and StandardScaler.  The information audit evaluated JHU Crowd and SUN
RGB-D on DINOv2-Small, ResNet-50, and ConvNeXt-Tiny over eight frozen views:
600 versus 2,000 training rows, PCA-32 versus PCA-128 versus no PCA, and (at the
larger budget) row normalization versus preserved magnitude.

PCA-32 was a material predictive bottleneck.  Moving from PCA-32 to the full
embedding improved the best score by as much as 0.0678 R² on JHU and 0.2166 R²
on SUN.  It also changed the leading backbone from ResNet-50 to DINOv2 on both
targets.  Increasing the training budget produced additional gains; preserving
raw embedding magnitude did not have a material effect.

The randomized PCA-32 calibration missed the archived full-solver linear score
by at most 0.01181 R², entirely on SUN/ConvNeXt.  A targeted decomposition showed
that the Ridge solver was numerically equivalent and the discrepancy came from
the approximate PCA subspace (retained variance 0.61291 versus 0.61362).  The
winner did not change.  All dimension and sample contrasts inside the audit use
the same deterministic randomized solver and remain internally controlled.

## Nonlinear heads

kNN and Extra Trees did not beat Ridge except for one nonmaterial 0.0041 R² gain.
Because the archived smooth RBF family had been stronger, a calibrated
precomputed-kernel RBF-SVR follow-up evaluated PCA-128 and full embeddings at 600
and 2,000 rows.  All six direct-versus-precomputed PCA-32 replays agreed to
machine precision.  RBF-SVR delivered at least 0.02 R² gain in five of eight
target/view panels, reaching 0.0301 on JHU and 0.0285 on SUN.

Those gains did not produce a robust backbone-ranking conflict.  Seven of eight
panels had linear-to-envelope Spearman 1.0 and zero top-1 regret.  The compact
grid produced one weak JHU/PCA-128/600-row reversal (Spearman 0.5, top-1 regret
0.0042), but every candidate touched a grid boundary.  Repeating those three
cells with the complete qualified grid plus C=30 removed the reversal:

| Candidate | Linear R² | RBF R² | Envelope R² | Nonlinear gain |
|---|---:|---:|---:|---:|
| ConvNeXt-Tiny | 0.6782 | 0.6703 | 0.6782 | 0.0000 |
| DINOv2-Small | 0.6837 | 0.6937 | 0.6937 | 0.0100 |
| ResNet-50 | 0.6919 | 0.7059 | 0.7059 | 0.0141 |

The resolved ordering has Spearman 1.0 and linear top-1 regret 0.  C=30 remains
selected, but its strongest endpoint advantage occurs for ResNet-50 (0.0172
inner-CV R²), which reinforces rather than threatens the resolved ordering.  No
model warnings occurred.

## Interpretation

The first problem was evaluator compression, not a lack of information in the
terminal embeddings.  Full embeddings are substantially more predictive than
PCA-32 and should be the default reference space in the next development
benchmark.  However, better smooth nonlinear predictions still improve the same
backbones that linear Ridge already ranks highly.  These two real targets
therefore remain poor tests of the specific failure mode needed for continuous
OI development: candidate-dependent nonlinear recoverability that changes the
backbone order.

The next experiment should inspect intermediate layers and pooling summaries,
while retaining the full terminal embedding as a control.  The useful candidate
views are matched relative-depth representations with global mean and
mean-plus-standard-deviation pooling.  If those views still produce no stable
linear-versus-envelope ranking conflict, the project should stop searching these
two sources and use a separately designed mechanism-controlled synthetic panel
for method development, with real datasets reserved for regression and external
validity.

## Runtime

Recorded model-stage wall time was 183.8 seconds for the 48-cell information
audit, 39.2 seconds for the 24-cell compact smooth-head audit, and 6.9 seconds for
the three-cell expanded-grid resolution.  Source extraction time is excluded
because the embedding cache already existed.

