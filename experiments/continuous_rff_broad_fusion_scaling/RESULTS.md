# Prototype-free broad fusion scaling results

The frozen experiment completed all 45 real candidate rows and all 12
synthetic scaling blocks. The prototype-free implementation exactly reproduced
the predeclared fused broad diagnostic.

## Exactness

- All 45 aggregate scores exactly matched the prior fused8 broad scores.
- All 45 recomputed full-path scores exactly matched the prior scores.
- All 180 prototype-free fold scores exactly matched the corresponding
  recomputed full-path fold scores.
- All 18 ranking rows therefore reproduced the prior rankings exactly.

The maximum observed aggregate and fold-level absolute difference was zero.
This proves that SAVE geometry, target-cell construction, and prototype fitting
were computationally unused by this particular broad-score estimand.

## Real-panel runtime

| Path | Median seconds/candidate | Relative to broad-only |
|---|---:|---:|
| Prototype-free broad fused8 | 0.1142 | 1.0x |
| Full post-projection fused8 | 1.0123 | 8.9x |
| Archived eight-selector score bag | 7.4777 | 65.5x |

The broad-only path was faster than the recomputed full path on every one of
the 45 candidate rows. The paired broad/full execution order was balanced
22/23 versus 23/22.

By task, prototype-free medians were 0.1146 seconds on the 500-row SmallNORB
panels and 0.0706 seconds on the 200-row GTSRB panels.

## Synthetic row scaling

| Rows | Broad-only seconds | Full fused seconds | Linear Ridge CV seconds | Broad/linear |
|---:|---:|---:|---:|---:|
| 200 | 0.0721 | 0.4509 | 0.0246 | 2.92x |
| 500 | 0.1216 | 1.0650 | 0.0272 | 4.47x |
| 1,000 | 0.2054 | 1.8236 | 0.0323 | 6.36x |
| 2,000 | 0.4809 | 3.2345 | 0.0424 | 11.33x |

The measured 200-to-2,000-row growth exponent was 0.82 for broad-only, 0.86
for the full fused path, and 0.24 for linear Ridge CV over this small range.
These finite-range exponents should not be extrapolated: the broad score still
constructs held-out Gram matrices and has an eventual quadratic row-cost and
memory term.

## Interpretation

Removing the prototype path is an unambiguous implementation improvement for
the broad score. It preserves the previously observed nonlinear and linear
ranking behavior while reducing the measured cost by nearly an order of
magnitude versus the fused selector and by roughly two orders of magnitude
versus selector score bagging.

It is not, however, a prototype-based overlap index. The remaining estimator
is a null-adjusted kernel-alignment ranker built on supervised low-rank RFF
views. It also remains slower than the deliberately lean synthetic linear
Ridge comparator, increasingly so as rows grow, despite small absolute times.

The next useful approximation is therefore at the Gram/alignment layer:
compare exact broad alignment with a fixed-budget landmark, sampled-pair, or
second random-feature estimator. The goal should be to preserve the 45-row
ranking exactly or within a frozen tolerance while replacing the eventual
quadratic row cost. If continuous-OI prototype semantics are mandatory, the
alternative path is a vectorized multi-partition fitter that retains one
prototype partition per RFF map; single-partition concatenation should not be
revisited.

## Artifact identity

- protocol SHA-256: `aa296354e92d10bc1b198d1f35cd44c0f69181e906f7d3623952912fc01c64c2`
- raw results SHA-256: `fbf81cbf3cdd9c8accf5d81399489770710f6a30fe1d12166ae192fa4e92af50`
- summary SHA-256: `2c571c95e7591df3cacb16ce14e7b15b2ffe2fe7498284a09437bcd5b15019aa`
- report SHA-256: `ac53c26eefdec740a88a0f97fe0b1a58598d1b485598aa97975b453d86dcf030`
- manifest SHA-256: `fe98fd43cf5c070450c9c10e60b651f03425b0b0c59e0cf43b2ffe1c0a273b5d`

All 57 checkpoints matched the terminal raw artifact, all timing repeats were
score-deterministic, and the complete related suite passed before execution.
