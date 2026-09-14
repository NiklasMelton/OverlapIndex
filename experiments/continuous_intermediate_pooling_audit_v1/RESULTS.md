# Continuous intermediate-layer and pooling audit V1 results

## Status

Completed and artifact-validated on 2026-09-14.  The extraction produced all 36
predeclared intermediate matrices for two targets and three backbones.  All six
terminal-vector parity checks passed; the largest absolute deviation from the
archived cache was 0.0000267 against a frozen 0.00005 tolerance.  The outcome
stage completed all 42 candidate cells with no model warnings and no
OverlapIndex fits.

## Main result

Early global-mean features expose the real backbone-ranking failure mode that
terminal embeddings did not.  The conflict replicates on both independent
targets:

| Target | Linear winner | Best-head-envelope winner | Spearman | Linear top-1 regret |
|---|---|---|---:|---:|
| JHU Crowd | DINOv2-Small | ResNet-50 | 0.5 | 0.0123 R² |
| SUN RGB-D | ResNet-50 | DINOv2-Small | -0.5 | 0.0190 R² |

On JHU early-mean features, the nonlinear gains over Ridge were 0.0574 for
ConvNeXt-Tiny, 0.0220 for DINOv2-Small, and 0.0394 for ResNet-50.  On SUN they
were 0.0495, 0.0209, and 0.0000 respectively.  These candidate-dependent gains
are exactly what was missing from the terminal-vector screens.

The full early-mean scores were:

| Target | Candidate | Linear R² | RBF R² | Envelope R² |
|---|---|---:|---:|---:|
| JHU | ConvNeXt-Tiny | 0.5705 | 0.6279 | 0.6279 |
| JHU | DINOv2-Small | 0.6111 | 0.6331 | 0.6331 |
| JHU | ResNet-50 | 0.6059 | 0.6453 | 0.6453 |
| SUN | ConvNeXt-Tiny | 0.2348 | 0.2843 | 0.2843 |
| SUN | DINOv2-Small | 0.2735 | 0.2944 | 0.2944 |
| SUN | ResNet-50 | 0.2755 | 0.2750 | 0.2755 |

The early-mean RBF selections did not have material search-boundary advantages,
and no warnings occurred.  This makes the replicated conflict more credible
than the earlier compact-grid PCA-128 reversal, which disappeared after boundary
resolution.

## What the other views say

Adding channel dispersion (`mean_std`) removes the early-view conflict on both
targets: the linear and envelope rankings return to Spearman 1.0 with zero
top-1 regret.  The extra statistics therefore make relevant information linearly
accessible rather than creating a stronger nonlinear challenge.

Middle and late views mostly agree with the linear ranking.  JHU
`middle_mean_std` has Spearman 0.5 but zero top-1 regret; it is a shortlist-order
diagnostic, not a winner-selection failure.  All late and cached-final views have
Spearman 1.0 and zero linear top-1 regret.  This supports a concrete mechanism:
later representation stages and richer pooling progressively linearize or expose
the target information, whereas an early global mean retains candidate-specific
nonlinear recoverability.

Six of fourteen aligned target/view panels have at least 0.02 R² nonlinear gain,
but only early global mean gives a replicated top-1 ranking conflict.  The
predeclared stable-conflict gate therefore passes.

## Interpretation and next step

The real datasets were not inherently too easy.  The terminal representation
choice made them look too linear.  Intermediate features provide a valid real
development setting in which the best-head envelope and linear probe disagree.

Freeze `early_mean` as the implicated development view.  Before making a broad
generalization claim, extend only this inexpensive view to the two remaining
backbones and the complete 2,000-row training pools, then repeat several frozen
600-row draws.  If the conflict persists, evaluate continuous OI variants against
the already-frozen best-head envelope on this panel.  Keep `early_mean_std`,
`late_mean`, and `cached_final` as negative controls.  Synthetic mechanism panels
remain useful for diagnosis, but they are no longer the only viable development
benchmark.

## Artifact identities

- Embedding manifest: `5becc217680ca123f70c51550e226fd2e7788706451ef859b62e4cf4a9f5f94f`
- Raw results: `d569e904dc4d8627280bac0b10abb947f7f82eed137a3a7ea26dbfbecab19dee`
- Summary: `687290fde4769b4dc1557356e9a901e72e95a0b3ed8589b466d2024a933111f7`
- Manifest: `7e66374acee65d6e912cfe6a73b240bc1f1a0c7923a265b727f5cf1c93f5fc05`
- Terminal: `6d7e7973b65db1b9d69e7fd0bd1bfdf5c490667c04b710e9fe5c1c2419482901`

