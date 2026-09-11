# Scalable post-projection fusion results

This development-only experiment completed the frozen 6-panel, 45-candidate
grid. It compared an eight-selector RFF64 score bag with a scalable geometry
that independently fits the same eight rank-8 target projections, fuses their
projected coordinates, and performs one final prototype fit.

## Frozen primary result

The primary `save_rff/postprojection_fusion8` candidate failed the frozen
ranking-retention rule despite a large speedup.

| Task | Ranker | Mean Spearman | Mean top-1 regret R2 | Exact-best rate |
|---|---|---:|---:|---:|
| SmallNORB elevation | Linear CV | 0.8465 | 0.01902 | 0.333 |
| SmallNORB elevation | SAVE score-bag8 | 0.8909 | 0.00397 | 0.667 |
| SmallNORB elevation | SAVE fused4 | 0.7535 | 0.01294 | 0.333 |
| SmallNORB elevation | SAVE fused8 | 0.6768 | 0.05876 | 0.333 |
| GTSRB box area | Linear CV | 1.0000 | 0.00000 | 1.000 |
| GTSRB box area | SAVE score-bag8 | 0.9000 | 0.00000 | 1.000 |
| GTSRB box area | SAVE fused8 | 0.8667 | 0.00000 | 1.000 |

The frozen decision was
`postprojection_fusion_does_not_retain_bagged_ranking_and_speed`. Five of ten
criteria failed: nonlinear regret versus Linear CV, nonlinear rank correlation
versus Linear CV, both nonlinear retention checks versus score-bag8, and the
linear-control rank-correlation floor.

## Runtime result

The computational approximation worked as intended:

- median score-bag8 time: 7.4777 seconds/candidate;
- median fused8 time: 1.0118 seconds/candidate;
- median paired ratio: 0.1358, or a 7.36x speedup;
- fused8 was faster on all 45 paired candidate rows; and
- primary execution order was counterbalanced 23/22 versus 22/23.

The similar fused4 and fused8 medians (0.9633 versus 1.0118 seconds) and the
large gap to score bagging show that repeated prototype fitting, rather than
the additional projected maps, dominates this experiment's runtime.

## Predeclared broad-score diagnostic

The broad score on the same post-projection fused geometry did retain the
useful ranking behavior:

| Task | Ranker | Mean Spearman | Mean top-1 regret R2 | Exact-best rate |
|---|---|---:|---:|---:|
| SmallNORB elevation | Broad score-bag8 | 0.8747 | 0.00397 | 0.667 |
| SmallNORB elevation | Broad fused4 | 0.8263 | 0.00397 | 0.667 |
| SmallNORB elevation | Broad fused8 | 0.8545 | 0.00397 | 0.667 |
| GTSRB box area | Broad score-bag8 | 0.9000 | 0.01870 | 0.667 |
| GTSRB box area | Broad fused8 | 0.9667 | 0.00000 | 1.000 |

Broad fused8 would satisfy the analogous frozen ranking and timing thresholds,
but it was diagnostic-only and therefore is not promoted by this experiment.
Its candidate-level rank agreement with broad score-bag8 was 0.964, 0.988,
and 0.976 across the three nonlinear panels.

## Interpretation

The failure is specific to forcing all projected views through one shared
prototype partition, not to post-projection fusion itself. Score bagging lets
each randomized low-rank view form its own prototypes and averages only the
final evidence. A single fused SAVE selector instead asks one fixed collection
of prototypes to represent a 64-dimensional union of eight independently
oriented views. The worsening from fused4 to fused8 on the nonlinear task,
while the smooth broad score improves, is direct evidence of this partition
conflict.

The next scalable test should therefore preserve map-specific prototype
partitions while sharing/vectorizing their computation, or independently
validate the broad fused score with a broad-only runtime path that does not pay
for an unused prototype fit. Simply concatenating more projected maps before
one ordinary prototype fit is not recommended.

## Artifact identity

- protocol SHA-256: `30324be51ac48cd6c20277a53ca0b7527b4ae5f50c421d63e23bf89a017489be`
- raw results SHA-256: `de57e26631fed506e4626c1966bc7aa6989b616cbf4bf105747afce2ca0ad651`
- summary SHA-256: `8be1a596af026d3384a4324d3650230744e1ce67c2f98f63df7a40f9fd28f2f8`
- report SHA-256: `0c42f5911fb1ee5ff44329f2aad7dc3ad134acdfb704794fb314906d4c3efb51`
- manifest SHA-256: `5aae88b963231f276b6c658d99c85596f8d751256837643f09ad09336c9fc60b`

All 45 checkpoints matched the terminal raw rows, all 360 recomputed map
scores matched the hash-bound source evidence at absolute tolerance 1e-12,
and an independent recomputation matched all 42 ranking rows.
