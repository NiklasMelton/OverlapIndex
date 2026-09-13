# Corrected evaluator pilot results

## Identity

- Executed commit: `2bc47f985ee9b6a2a6880233107ea10d342eb2f8`
- Protocol SHA-256: `f355726c6713c33c005f1484d99c14b30652792917eddce53995287851684f0b`
- Terminal manifest SHA-256: `0fa03659bc529067b922da590342484d2ed98c014bd6c147a450f65d8e94970e`
- Terminal summary SHA-256: `04c1d7dcf50478ed97fa08487487a356901404e67527cde401bab0c210ad9a34`
- Environment: Python 3.9, one numerical thread
- Focused verification before execution: 8 tests passed
- Terminal validation after execution: passed
- Executed grid: 3 real targets, 105 real candidate draws, 15 synthetic panels,
  75 synthetic candidate draws, 0 OverlapIndex fits, and 0 reserved backbones

The run used the frozen protocol without tuning or post-outcome changes. The
manifest records a clean source tree at execution.

## Decision

The predeclared pilot gate **failed** because the corrected evaluator did not
replicate the earlier LSP wrist backbone reversal. The other two components of
the gate passed:

- all linear, quadratic, and smooth-periodic synthetic controls passed;
- both CUB linear controls were preserved;
- the LSP wrist nonlinear reversal was not replicated.

This is an evaluator-only result and promotes no continuous OverlapIndex
method.

## Corrected real references

| Target | Envelope best | Linear best | Rank rho | Linear top-1 regret | Maximum nonlinear gain | Stable |
|---|---|---|---:|---:|---:|---:|
| CUB leg span | DINOv2-small | DINOv2-small | 1.00 | 0.000 | 0.023 | yes |
| CUB wing axis | DINOv2-small | DINOv2-small | 1.00 | 0.000 | 0.000 | yes |
| LSP wrist span | ConvNeXt-tiny | ConvNeXt-tiny | 0.70 | 0.000 | 0.236 | yes |

LSP remains nonlinear in head selection: RBF-SVR is the best head for all five
backbones, and the largest candidate-level nonlinear gain is 0.236 R2. It is
not, however, a nonlinear *backbone-ranking reversal* under the corrected
reference. ConvNeXt-tiny is best under both linear Ridge (0.149 R2) and the
four-head envelope (0.386 R2). DINOv2-small is second under the envelope at
0.348 R2.

The LSP top candidate is not overwhelming: its paired evaluation-row bootstrap
top-1 probability is 0.7835, and the ConvNeXt-minus-DINO point difference is
0.0374 R2 with sign probability 0.784. This is above the frozen stability
threshold but should not be described as a large separation. LSP has only one
unique 590-row training draw, so its uncertainty does not include training-draw
variation.

## Why V1 changed

V1's nominal 590-row LSP bootstrap repeats contained only 378 or 379 distinct
sample identities. Because inner folds were split by row rather than identity,
the same sample could occur in both training and validation. Preprocessing was
also fitted outside the inner folds. Those defects made inner validation
optimistic and hyperparameter selection unstable.

The clearest example is ConvNeXt-tiny RBF-SVR. V1 selected aggressive
configurations (`C=10` twice and `C=100` once) after inner R2 values of 0.590,
0.473, and 0.772. Their true held-out R2 values were 0.308, 0.044, and -2.079,
for a mean of -0.576. The corrected evaluator uses all 590 identities exactly
once, has zero identity overlap in every fold, fits preprocessing within each
fold, selects `C=1`, `gamma=0.03`, `epsilon=0.1`, and obtains held-out R2 of
0.386. DINO's RBF result also improves, from 0.269 to 0.348, but ConvNeXt
improves enough to become the envelope winner.

Because the pilot changes unique-sample selection, fold-local preprocessing,
fold count, and numerical solver together, it does not separately estimate the
causal contribution of each correction. The duplicated-identity leakage and
the V1 inner/held-out discrepancy nevertheless provide direct evidence that the
old LSP reversal was not a trustworthy reference result.

## Synthetic controls

| Mechanism | Clean-latent top probability | Mean linear regret | Mean linear-to-envelope rho | Passed |
|---|---:|---:|---:|---:|
| Linear | 1.00 | 0.000 | 1.00 | yes |
| Quadratic | 1.00 | 0.126 | -0.30 | yes |
| Smooth periodic | 1.00 | 0.215 | 0.12 | yes |

Every one of the 15 panels ranked the clean nonlinear representation first;
every panel had three material adjacent candidate gaps. The corrected evaluator
therefore detects the intended nonlinear representation advantage when it is
known to exist, while preserving the linear control.

## Reference-library limitation exposed by the pilot

The pilot deliberately retained the historical V1 grids to isolate evaluator
corrections. Those grids should not be mistaken for a finished best-head
envelope:

- all 105 real quadratic fits selected the maximum available Ridge alpha (10),
  indicating an inadequate regularization range for the 560-feature quadratic
  expansion of PCA-32 embeddings;
- 103 of 105 real RBF fits selected the minimum available `C` (1), indicating
  that the RBF range is also boundary-limited;
- real quadratic held-out scores are often severely negative, whereas the
  two-dimensional quadratic synthetic control is nearly perfectly learnable.

The stronger benchmark must expand these grids using the same fold-safe nested
procedure and freeze them before evaluating any ranking method. This is a
reference-definition correction, not tuning of an OI candidate.

## Updated interpretation and next action

The corrected evaluator is suitable as the foundation for a larger benchmark:
it passes the designed nonlinear controls, preserves the real linear controls,
uses unique samples, and provides reproducible fold and artifact identity. The
previous claim that LSP wrist demonstrates a reliable real nonlinear backbone
reversal is retired.

The next benchmark should therefore do two things before comparing continuous
OI variants:

1. strengthen and freeze the best-head reference library, including wider
   quadratic regularization and lower-complexity RBF settings; and
2. find a broader, independently qualified real nonlinear stratum with more
   datasets, targets, backbones, and repeated unique training draws.

Only targets whose corrected reference is learnable, materially separated, and
stable should enter method comparison. The reserved-backbone confirmation panel
must remain untouched until a method and all gates are frozen.
