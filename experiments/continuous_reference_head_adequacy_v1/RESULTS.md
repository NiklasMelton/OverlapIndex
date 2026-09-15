# Continuous reference-head adequacy V1 result

## Decision

The calibrated reference passed all 12 required checks. The previous
matched-geometry best-head envelope was underpowered for sparse nonlinear
structure in wide embeddings.

The screened heads selected the known axis feature or signed pair in every
required dimension/sample-size cell. Their held-out R2 differed from the
analytic conditional-mean R2 by at most 0.0107, below the frozen 0.05 limit.

| Required signal | 16d, 300 | 16d, 1,200 | 512d, 300 | 512d, 1,200 |
| --- | ---: | ---: | ---: | ---: |
| Direct q=.85 absolute R2 gap | 0.0023 | 0.0069 | 0.0041 | 0.0106 |
| Signed q=.85 absolute R2 gap | 0.0046 | 0.0041 | 0.0005 | 0.0100 |
| Signed q=.55 absolute R2 gap | 0.0038 | 0.0082 | 0.0024 | 0.0091 |

## What changed in the interpretation

The original quadratic SVR used a fixed, narrow regularization grid and
dimension-dependent kernel scaling. In 512 dimensions it did not recover the
single informative interaction. After adding a training-only screened-pair
quadratic head, signed q=.85 improved from approximately zero held-out R2 to
0.644 at 300 rows and 0.654 at 1,200 rows.

This means the former finite-head envelope did not reliably represent the best
attainable head performance. The synthetic data did retain the nonlinear
signal; the reference learner failed to access it.

Reanalyzing the existing selectors against the calibrated envelope makes the
geometry problem clearer. Observed continuous OI avoided top-one regret in all
four reference panels but retained modest rank correlation. Broad RFF selected
the weak signed candidate at 512 dimensions and 300 rows, incurring 0.381 R2
regret. SAVE selected the rotated, less-accessible representation in that cell,
incurring 0.324 R2 regret. The selectors therefore remain sensitive to
geometry even after the reference is repaired.

## Next gate

The passing calibration authorizes the fresh linear-versus-nonlinear crossover.
That diagnostic reverses which geometry carries the stronger signal and tests
whether each ranking method follows representation quality rather than a fixed
geometric preference.

## Provenance

- Implementation commit: `672dec1`
- Protocol SHA-256: `fcb51b16f3bd70b4515f77e8db16c732a8aced6c1b0d2f5453f34b0773b60a6e`
- Source raw SHA-256: `2dc5c7c09ea3db3e3c4f5ad0146175824d266f0e4864588604f198fb13dcb4ac`
- Raw result SHA-256: `4bff453a7d9a6a0b5621bed8bf5d4008b9d093db265ea96e83c62fd9290dda20`
- Summary SHA-256: `f97012dcf233fff73bdc079d92a4b906ffdb6bb40f685c34fcb6aaa8b617f02a`
- Terminal manifest SHA-256: `7084e1b2b6734d045a6eb4c10ccc93845354b3445f49c5ef48aeef6a06ba71c1`
