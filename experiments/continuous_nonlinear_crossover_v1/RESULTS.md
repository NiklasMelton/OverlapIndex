# Continuous nonlinear crossover V1 result

## Decision

The calibrated reference was valid in all 24 panels, but none of the four
geometry-based ranking methods passed the frozen crossover gate. The only
survivor was the training-only maximum of the screened linear and screened-pair
head scores.

This is a mechanism result, not a promotion result. It shows that the desired
ranking objective is feasible on the controlled problem, while the current
continuous-OI formulations do not reliably estimate it.

| Ranking method | Direction accuracy | Mean regret (R2) | Spearman | Exact best | Mean seconds per candidate | Survives |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Linear CV | 50.0% | 0.1423 | 0.217 | 50.0% | 0.0636 | No |
| Observed continuous OI | 54.2% | 0.1304 | 0.550 | 54.2% | 0.9379 | No |
| Broad RFF | 45.8% | 0.1767 | 0.550 | 45.8% | 0.1539 | No |
| SAVE OI | 70.8% | 0.1271 | 0.442 | 70.8% | 0.9345 | No |
| Screened head CV maximum | **100.0%** | **0.0000** | **0.950** | **100.0%** | 0.0716 | **Yes** |

The timing column is descriptive rather than a counterbalanced runtime result.
It includes the complete selector call for one candidate. On this panel the
screened two-head score was about 1.13 times the dense linear-CV time on
average, whereas observed OI and SAVE were about 14.8 times as slow.

## Crossover behavior

The reference winner had a held-out R2 advantage of 0.269 to 0.299 in every
panel, comfortably above the frozen 0.10 validity threshold.

When the axis-aligned representation was stronger, linear CV and observed OI
both chose it in all 12 panels. When the signed-pair representation was
stronger, linear CV chose it in 0/12 panels and observed OI in 1/12. Broad RFF
chose it in 0/12.

SAVE contained real nonlinear sensitivity, choosing the signed-pair winner in
8/12 nonlinear panels. That success did not scale reliably with nuisance
dimension:

| Dimension | Rows | SAVE nonlinear wins |
| ---: | ---: | ---: |
| 16 | 300 | 3/3 |
| 16 | 1,200 | 3/3 |
| 512 | 300 | 1/3 |
| 512 | 1,200 | 1/3 |

The screened-head maximum chose the correct winner in every linear and
nonlinear cell. Its per-scenario Spearman correlations were 0.983 and 0.917.

## What was learned

1. The repaired reference rules out evaluator failure: both competing signals
   were accessible to the frozen head envelope in every panel.
2. Continuous OI is not currently a universal proxy for best attainable head
   performance. Its score is biased toward particular marginal and local
   geometries even when a simple nonlinear head can recover the target.
3. SAVE is the strongest OI-family nonlinear detector tested here, but wide
   nuisance dimensions erase too much of its advantage.
4. A small cross-validated head envelope can rank both regimes accurately at a
   cost close to a linear probe. This is better evidence for a separate
   `potential` ranking objective than for another prototype or normalization
   revision to continuous OI.

## Recommendation

Keep continuous OI as an interpretable overlap statistic. For Vertebrae's
best-head-potential ranking, test a scalable two-channel proxy:

1. one regularized linear channel;
2. one fixed-capacity nonlinear channel that targets sparse interactions;
3. a training-only cross-validated maximum with no outcome-dependent channel
   selection.

The present screened-pair implementation is an exact diagnostic, not a
production candidate: enumerating all feature pairs is quadratic in embedding
dimension and it repeats work across regularization settings. The next frozen
experiment should compare it with blocked, sketched, or low-rank interaction
screening at 384, 768, and 1,536 dimensions, with a counterbalanced runtime gate
against the linear probe. Only a variant that preserves the crossover result
and remains within the runtime budget should advance to unused real data.

## Provenance

- Implementation commit: `f26a0c8`
- Protocol SHA-256: `a10b44ba85ccf7f3c4b39e85fc0422804863689d6a32cd8bcc78d0e187ca459c`
- Adequacy manifest SHA-256: `7084e1b2b6734d045a6eb4c10ccc93845354b3445f49c5ef48aeef6a06ba71c1`
- Runner SHA-256: `2cbb0420ffcae7887002af8a4f38f17abb18e7062b64daf4224266fd7e7f4f8e`
- Raw result SHA-256: `d36eb0fb82daad1d821243890750b539bcee7a5603bedebbc2bd0920d22fe283`
- Summary SHA-256: `0c736294a2a3e9ec53cde0658680e943d552f3eaf48075e2b022599ce877b163`
- Terminal manifest SHA-256: `62ec39bba7368f9d69179dafe8697a79836e5b09c0989b8fa30f44ccbe56d53a`
- Terminal counts: 24 panels, 1,056 selector rows, 192 reference rows
