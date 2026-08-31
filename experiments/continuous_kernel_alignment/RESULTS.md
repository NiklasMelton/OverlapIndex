# Continuous characteristic-kernel alignment result

The frozen new-seed characteristic-kernel experiment completed and failed its
universal full-ranking gate. It nevertheless produced the strongest
best-candidate signal observed in the continuous series: a predeclared broad
RFF feature-kernel diagnostic selected the exact best candidate in 16 of 18
panels with mean regret .0055. Because that bandwidth was diagnostic-only and
the full-panel ordering remained weak, it is a new validation hypothesis, not
an eligible result or integration candidate.

## Identity and validation

- Frozen pre-outcome implementation commit: `6c943b5`
- Protocol SHA-256:
  `333a014dc75a5bc4029caee9ba03d605f63adc059783a9ead27dd3e9835f739c`
- Runner SHA-256:
  `309ac8b390cf5c41858921b6a7c0caf17258a15909dbeed06620ac07b936d1ab`
- Motivation raw-results SHA-256:
  `ca10227cc0c4dcc861e4d176758326de1c6a29778c279c4f62b3a6d9436f2860`
- Raw-results SHA-256:
  `e1d99f1659ba0ba620249039116919d964a486a8f2e490e8b91ea10ceb3a5012`
- Summary SHA-256:
  `07560be5b55cd3b1c623773a0f28f916bf38275eb8170c97256812e1aacc7eee`
- Manifest SHA-256:
  `930a4f1dc0507d7f4027639e920b43c196bdd5444dc3fa13144b246c5157d508`
- New panel seeds: 1601, 1709, and 1811.
- Complete grid: 18 panels, 288 candidates, 576 selectors, 2,304 fitted
  fold models, 294,912 kernel-null alignments, 73,728 matched current-OI null
  losses, and 18 checkpoints.

Every source and terminal hash validated. All panel/candidate/head/fold
identities, training-fit bandwidths, raw and null alignments, individual
bandwidths, null-adjusted scores, null banks, and checkpoints were checked.
Rebuilding the full summary from raw panels matched exactly. The complete
project suite passed with 391 tests, 14 expected skips, and 34 existing
under-prototyping warnings.

## Frozen primary result

Neither eligible mixed-kernel score passed.

| Geometry / estimator | Mean rho | Mean regret | Exact best | Within .01 |
| --- | ---: | ---: | ---: | ---: |
| Linear probe | .446 | .255 | .444 | .444 |
| RFF mean / current OI | .421 | .114 | .611 | .611 |
| RFF mean / mixed-kernel alignment | .400 | **.018** | **.778** | **.778** |
| SAVE + RFF / current OI | **.610** | .105 | .722 | .722 |
| SAVE + RFF / mixed-kernel alignment | .481 | .120 | .611 | .611 |

The RFF mixed-kernel score was an excellent best-candidate selector, passing
absolute regret, exact-best, and within-.01 gates. It failed overall rho,
periodic gain, and linear preservation. SAVE + RFF kernel alignment weakened
both full ordering and selection relative to matched current OI.

The RFF mixed-kernel family results show the split directly:

| Family | Mean rho | Mean regret | Exact-best rate |
| --- | ---: | ---: | ---: |
| Linear | .525 | .000 | 1.000 |
| Quadratic product | .404 | .000 | 1.000 |
| Radial quadratic | .256 | .061 | .667 |
| Absolute value | .385 | .000 | 1.000 |
| Sinusoid | .878 | .049 | .000 |
| Triangle wave | -.047 | .000 | 1.000 |

It often identified the winner even when it ordered the remaining 15
candidates poorly. High top-selection fidelity and high full-panel Spearman
are empirically different objectives.

## Broad-bandwidth diagnostic

The three feature-kernel bandwidths were frozen and reported diagnostically,
but only their equal mixture was eligible. Within RFF geometry, increasing
the feature bandwidth monotonically improved the observed top-selection
metrics:

| Feature bandwidth multiplier | Mean rho | Mean regret | Exact best | Within .01 |
| --- | ---: | ---: | ---: | ---: |
| 0.25 | .340 | .121 | .611 | .611 |
| 1.0 | .422 | .046 | .778 | .778 |
| 4.0 | **.473** | **.0055** | **.889** | **.889** |

The broad multiplier selected the exact best candidate in every linear,
quadratic-product, radial-quadratic, absolute-value, and triangle-wave panel.
Its only selection failures were two sinusoid panels; even there, mean family
regret was .033. Family rho was .585 linear, .406 quadratic product, .310
radial quadratic, .402 absolute value, .879 sinusoid, and .254 triangle wave.

This does not pass the full-ranking target: overall rho .473 remains well
below .60. Nor may multiplier 4.0 be promoted from this experiment, because
the protocol explicitly made individual bandwidths diagnostic and choosing
the best one now would be post-outcome selection.

## What this changes

Characteristic-kernel dependence avoided the coarse-neighbor failure of the
multiscale graph and produced extremely low selection regret. It did not
produce a faithful total order. Together with the prior row-edge experiment,
this is now a repeated pattern on independent seed blocks:

- local or characteristic dependence can identify a representation with
  nearly maximal downstream performance;
- current prototype OI or SAVE + RFF geometry better orders the middle of the
  candidate panel;
- optimizing one behavior does not automatically optimize the other.

The broad-bandwidth result suggests that winner selection depends on smooth,
large-scale kernel dependence rather than only fine neighborhoods. But a
dependence statistic naturally compresses all kinds of target information
into a magnitude; it has no reason to reproduce small performance differences
among mediocre representations. That is why selection can be nearly perfect
while Spearman remains modest.

## Recommendation

The original universal *full-ranking* objective is not achieved. Do not add
the mixed kernel or multiplier 4.0 to public continuous OI yet.

There are now two defensible paths:

1. If the practical objective is choosing the best or a near-best backbone,
   freeze multiplier 4.0 as the sole primary estimator and validate it on a
   genuinely broader panel—preferably real continuous tasks rather than a
   fourth seed block from the same synthetic generator. Include runtime and
   compare selection regret directly with linear probing.
2. If a faithful complete ranking remains mandatory, stop searching for a
   nominally universal scalar. The repeated evidence supports an explicit
   two-channel method: one channel for global panel order and one for
   top-candidate evidence, with a frozen combination rule validated outside
   these synthetic outcomes.

The broad RFF kernel is a promising unified *selector*. It is not a universal
ranker.
