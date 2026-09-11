# Recoverable-target continuous ranker comparison result

## Outcome

The prospectively frozen comparison completed all 24 panels, 120 scout-backbone
candidates, and 480 Continuous OI outer fits. No reserved backbone was
evaluated. Neither fixed continuous ranker passed the combined opportunity,
control-preservation, shortlist, and runtime gate.

| Group | Method | Spearman | Top-1 regret R2 | Exact best | Top-3 opportunity regret R2 | Top-3 utility regret R2 | Top-3 recall |
|---|---|---:|---:|---:|---:|---:|---:|
| All | Linear CV | 0.3792 | 0.04119 | 0.375 | 0.00900 | 0.02458 | 0.708 |
| All | Broad RFF | 0.2042 | 0.06968 | 0.250 | **0.00648** | **0.02228** | 0.694 |
| All | SAVE+RFF | 0.1472 | 0.07535 | 0.375 | 0.03083 | 0.04031 | 0.625 |
| All | Raw four-head CV max | **0.4792** | **0.03673** | **0.500** | 0.01072 | **0.01601** | **0.778** |
| Nonlinear opportunity | Linear CV | 0.3500 | 0.04191 | 0.333 | 0.01154 | 0.02619 | 0.750 |
| Nonlinear opportunity | Broad RFF | 0.3417 | 0.07655 | 0.167 | **0.00504** | 0.01682 | **0.806** |
| Nonlinear opportunity | SAVE+RFF | 0.1778 | 0.06364 | 0.417 | 0.03266 | 0.04158 | 0.639 |
| Nonlinear opportunity | Raw four-head CV max | **0.4833** | **0.03107** | **0.583** | 0.01372 | **0.01572** | 0.778 |
| LP-dominated control | Linear CV | 0.4083 | **0.04047** | **0.417** | **0.00646** | 0.02297 | 0.667 |
| LP-dominated control | Broad RFF | 0.0667 | 0.06281 | 0.333 | 0.00792 | 0.02775 | 0.583 |
| LP-dominated control | SAVE+RFF | 0.1167 | 0.08707 | 0.333 | 0.02899 | 0.03903 | 0.611 |
| LP-dominated control | Raw four-head CV max | **0.4750** | 0.04238 | **0.417** | 0.00772 | **0.01630** | **0.778** |

## Interpretation

Broad RFF reproduced the distinction seen in earlier experiments. It is useful
as a nonlinear *shortlisting* channel: on the four opportunity targets it cut
mean top-three opportunity regret from 0.01154 to 0.00504 R2, improved top-three
recall from 0.750 to 0.806, and reduced top-three utility regret from 0.02619 to
0.01682. It did not identify the best backbone reliably. Its opportunity
Spearman was essentially tied with Linear CV (0.3417 versus 0.3500), while its
top-one regret was substantially worse (0.07655 versus 0.04191).

That shortlist benefit did not generalize uniformly. Broad RFF improved the LSP
ankle and Pascal area rankings and strongly improved the COCO wrist control,
but it damaged LSP wrist, CUB leg, and Pet compactness. Across all controls its
Spearman fell by 0.342 and top-one regret increased by 0.02234 R2 relative to
Linear CV. This target-level sign switching is the central obstacle to treating
the score as a universal replacement ranker.

SAVE+RFF does not merit further scaling in its current form. It was worse than
Linear CV on opportunity Spearman, opportunity top-one regret, control
preservation, and overall shortlisting, while costing 9.31 times Linear CV.
The discrete-inspired prototype surface therefore remains a poor universal
continuous analog on this broader real benchmark.

The raw four-head training-CV maximum is the most informative diagnostic. It
had the best overall Spearman, exact-best rate, material-pair accuracy, and
top-three recall, and improved opportunity top-one regret. This supports the
best-head envelope as a learnable ranking target. It is not a solution: it cost
7.99 times Linear CV and did not dominate Linear CV on every regret measure.
The next method experiment should therefore test a prospectively frozen,
low-cost approximation to the multi-head CV envelope rather than another
prototype or aggregation adjustment.

## Runtime

| Method | Aggregate seconds | Median seconds/candidate | Relative to Linear CV |
|---|---:|---:|---:|
| Linear CV | 2.205 | 0.0184 | 1.00x |
| Broad RFF | 6.671 | 0.0553 | 3.02x |
| SAVE+RFF | 20.538 | 0.1704 | 9.31x |
| Raw four-head CV max | 17.617 | 0.1481 | 7.99x |

Broad RFF missed its three-times runtime threshold by only 0.8%, but this is
not the decisive failure: it also missed four substantive ranking gates.
Runtime optimization alone would not make it viable as the universal ranker.

## Frozen gate contrasts

- Broad RFF opportunity Spearman gain: -0.0083 (required at least +0.05).
- Broad RFF opportunity top-one regret reduction: -0.03463 R2 (required at
  least +0.005).
- Broad RFF control Spearman drop: 0.3417 (allowed at most 0.05).
- Broad RFF control top-one regret increase: 0.02234 R2 (allowed at most 0.005).
- Broad RFF did preserve overall top-three opportunity regret.
- SAVE+RFF failed every frozen check.

## Verification and identity

- Protocol SHA-256:
  `ee7a4864e8b229ce12d883fac9ae80718bd52514efde725fa6a98a547e5df79c`
- Runner SHA-256:
  `a4d363cb22556e9570ae793c6c0c76d90a0753a5f62453cdb4779dc8ba7ff029`
- Raw results SHA-256:
  `7b497d8e6746c42c0e1bcb88afa403c65882abebaad3a2792aeaa8875142488f`
- Summary SHA-256:
  `ead4d39d3b35bf6e7b368a7a129fdad2ccd799a978e8785c3d269601d12c26a4`
- Report SHA-256:
  `e9acd48d979efbfce0ce0198120340ccaadad142058a7054a0aba6c654d66b5e`
- Manifest SHA-256:
  `a3ee721249f10b6ab154c3d72f1eabfa84fcc91ffefc89faacb24ec72ffc8234`
- Focused/source/evaluator tests: 29 passed.
- All 24 panel checkpoints are present and the terminal artifact records 120
  candidates, 480 Continuous OI outer fits, and zero reserved backbones.

The initial runner correctly failed before scoring when it derived positional
PCA seeds from a canonically sorted artifact mapping. The accepted run derives
seeds from the original hash-validated protocol file; a regression test locks
that behavior.
