# Continuous matched-geometry diagnostic v1 results

## Status

Complete. The immutable terminal artifact contains 12 panels, 540 selector rows,
and 20 finite-head reference rows. A separately hashed post-outcome analyzer
computed the declared best-head-envelope endpoint without changing the raw
artifact.

This is a synthetic mechanism diagnostic. It is not evidence for promotion on
real data.

## Follow-up correction: reference-head adequacy

A subsequent pre-frozen calibration showed that the finite head library used
in this original analysis was underpowered for sparse nonlinear structure. A
training-only screened-pair quadratic head recovered the signed interaction in
all required dimension/sample-size cells and came within 0.0107 R2 of the
analytic reference.

The analytic geometry-sensitivity findings below remain valid. The conclusion
that the signed interaction itself was inaccessible to a practical finite head
does not. Treat the calibrated
[reference-head adequacy result](../continuous_reference_head_adequacy_v1/RESULTS.md)
and the subsequent
[linear/nonlinear crossover](../continuous_nonlinear_crossover_v1/RESULTS.md)
as authoritative for finite-head accessibility and ranking behavior.

## Analytic-information result

Four candidates were constructed to carry the same analytic target information
through direct, rotated, anisotropic, and signed-interaction encodings. A fifth
candidate carried materially less information.

| Method | Mean matched score range | Geometry-biased panels | Weaker selected | Mean analytic regret |
| --- | ---: | ---: | ---: | ---: |
| Observed continuous OI | 0.271 | 12/12 | 0/12 | 0.000 |
| Broad RFF | 0.048 | 6/12 | 1/12 | 0.031 |
| SAVE OI | **0.032** | **3/12** | 1/12 | 0.031 |

No method passed every predeclared safeguard. Observed OI never selected the
weaker candidate, but strongly preferred particular encodings of equal analytic
information. Broad RFF and SAVE reduced that representation bias, yet each
selected the weaker candidate once and had material-pair accuracy below the
required 0.90.

## Finite best-head-envelope result

The operational truth for backbone ranking is the best held-out performance
attainable by the frozen practical head library, not analytic conditional
information. On the four reference panels:

| Ranking method | Mean top-1 regret (R2) | Spearman | Exact best | Within 0.01 |
| --- | ---: | ---: | ---: | ---: |
| Linear CV | **0.00602** | **0.878** | **75%** | 75% |
| Observed continuous OI | 0.02980 | 0.473 | 0% | 50% |
| Broad RFF | 0.08643 | 0.380 | 50% | 75% |
| SAVE OI | 0.00604 | 0.573 | 50% | 75% |

SAVE matched linear-CV top-1 regret numerically on this small panel, but did not
match its overall ordering and has no real-data generalization evidence. Broad
RFF suffered 0.346 regret in the wide, low-sample panel after selecting the
weaker signed candidate.

## Mechanism learned

Analytically equivalent candidates were not equivalent for the practical head
library. The spread between their finite best-head scores was approximately:

| Dimension | Rows | Best-head spread (R2) |
| ---: | ---: | ---: |
| 16 | 300 | 0.255 |
| 16 | 1,200 | 0.068 |
| 512 | 300 | 0.346 |
| 512 | 1,200 | 0.552 |

In particular, the signed-interaction encoding preserves target information in
principle but becomes difficult for the finite, regularized head library to
access in wide nuisance dimensions. This is not merely noise in the OI score:
it shows that representation invariance is not the desired ranking invariant.
A backbone can preserve information yet require an impractical head to recover
it.

## Decision

Do not promote another geometry normalization, prototype rule, or scalar
continuous-OI variant from this result. The general continuous ranking problem
should pivot toward a cheap approximation of the attainable head envelope:

1. retain the linear-CV channel;
2. add one fixed-capacity nonlinear channel that can expose quadratic or smooth
   nonlinear accessibility;
3. combine their cross-validated predictions as an explicit `potential`
   ranking mode rather than redefining the existing continuous OI score.

First validate that the nonlinear channel can recover the signed interaction at
both dimensions without selecting the weaker candidate and within the linear
probe runtime budget. Then apply the frozen practical-value gate on unused real
datasets with genuine linear-probe ranking headroom. SAVE is the strongest OI
baseline for that comparison, but is not itself a promoted solution.

## Provenance

- Protocol SHA-256: `d14bf65d14d5580580e717ab4fe72300b07dbef2d01e48ba5d962fa4e506c0d9`
- Runner SHA-256: `530ebe3e341973089aaf8c8f2427f6fbb052a3f99cbb6bd2e085338a65e32270`
- Raw result SHA-256: `2dc5c7c09ea3db3e3c4f5ad0146175824d266f0e4864588604f198fb13dcb4ac`
- Summary SHA-256: `147124d64ff262cb4c6473766de80326992b7aca4e4032e4f05fc1920bc44e10`
- Analysis SHA-256: `dcba2659a8bea3f835d9af18d2e261a653b5826833d953d4b270701a3fd0c63d`
