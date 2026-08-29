# FUSED3 full Food-101 replay results

Status: `pass_full_retrospective` for the screen-locked `FUSED3` candidate.
This is complete retrospective development evidence under the archived,
class-balanced Food-101 protocol. It is not untouched confirmation.

## Frozen identity

- Execution commit: `de14fb0e35f1543ed60966d382efe557a141b166`
- `develop` merge-base: `165fa344653a72dd2abc04a20387d90b891b9acc`
- Full protocol SHA-256:
  `58943c711ac19f2bc4e1ef8a117e625b7d1c3730f328fd5b1736e8b1d3289135`
- Code identity SHA-256:
  `f2449c8528e0b719086069c4422b8897e03727def037d5c405943f2027c6b26c`
- Run identity SHA-256:
  `6a7bdbf553f49e416f8d183de83fe68427aa5de5381ebf5f2e43ad856eda1666`
- Raw results SHA-256:
  `079929a9dffd9a5b961e8023377f090647cafc9c0bc45de3e1b8b90d95a8b686`
- Run manifest SHA-256:
  `957a1a1394e60098d90a6075dbc681642e2d3f14d07860e14c765d6fc8474889`
- Analysis summary SHA-256:
  `2d310be5cb15dad1b8cd61eb449299c1887168061ff0e2e1e19695ef283ad9bb`
- Analysis decision SHA-256:
  `90ff8eaafa6b680ff7d08d9720e9afa904d286f0073bc0a11147e2d38fe14eb5`
- Analysis report SHA-256:
  `8b8e498a48781bd6b0dfdaeb0b6b25c71b28baa99f73f851d2309cd7de7566ac`
- Analysis manifest SHA-256:
  `277472b5e7f4d235bb757aa4c34e115c9733fade2962556f935f093284ec47c8`

The terminal artifact contains the exact frozen grid: 600 panels, 1,800
fresh selector rows, 600 archived refined-OI comparator rows, 600 downstream
reference rows, 600 fresh-probe parity rows, 30 screen-overlap parity rows,
and 600 complete atomic checkpoints. All three fresh methods repeated exactly
with clocks excluded. Fresh `LP-FULL` matched all archived probe scores,
fresh refined OI matched all archived refined scores, and the 30 overlapping
FUSED3 scores matched the locked screen.

## Ranking result

Mean selection regret in accuracy percentage points:

| Regime / downstream head | FUSED3 | Refined OI | Full linear probe |
|---|---:|---:|---:|
| Baseline / linear | 0.000 | 0.023 | 0.120 |
| Nuisance / linear | 0.000 | 3.312 | 0.000 |
| Nonlinear / quadratic | 0.000 | 0.243 | 3.250 |
| Nonlinear / kNN | 0.000 | 0.526 | 4.495 |
| Nonlinear / RBF | 0.000 | 0.255 | 3.070 |

FUSED3 selected the downstream-best backbone on every one of the 20 frozen
budget-by-replicate panels in each primary regime/head. It selected
`openclip-vit-b-32` throughout baseline and nonlinear geometry and
`dinov2-small` throughout nuisance geometry.

The paired FUSED3-minus-probe nonlinear regret upper 95% bounds were
strictly below zero for quadratic, kNN, and RBF, establishing the frozen
nonlinear-superiority claim. FUSED3 also had lower regret than fresh refined
OI for all three nonlinear heads, with upper 95% bounds of `-0.058`, `-0.142`,
and `-0.065` percentage points, respectively.

Middle-order ranking is more nuanced than top selection. FUSED3-minus-refined
normalized log-budget rank-AUC deltas were:

| Head | Estimate | Paired 95% interval |
|---|---:|---:|
| Quadratic | -0.0366 | [-0.0452, -0.0284] |
| kNN | +0.0120 | [-0.0039, +0.0279] |
| RBF | +0.0028 | [-0.0284, +0.0340] |

Thus FUSED3 slightly rearranged the quadratic middle ranks but stayed well
inside the predeclared `-0.10` retention margin. It retained or modestly
improved the kNN/RBF rank ordering while improving actual top-model regret.

## Runtime result

Median per-call FUSED3 wall time was `0.235 s` on baseline, `0.238 s` on
nonlinearity, and `0.239 s` on nuisance geometry. Corresponding median ratios:

| Arm | FUSED3 / full probe | FUSED3 / refined OI |
|---|---:|---:|
| Baseline | 0.200 | 0.218 |
| Nonlinearity | 0.336 | 0.192 |
| Nuisance | 0.260 | 0.107 |

The summed ten-backbone panel ratios were `0.192/0.207`, `0.327/0.185`, and
`0.256/0.105` against the probe/refined OI for baseline, nonlinearity, and
nuisance, respectively. Every per-call and full-panel upper 95% bound was
below `1.0`, so FUSED3 was demonstrably faster than both comparators in all
three arms, not merely noninferior.

## Decision and interpretation

All 33 required ranking, rank-retention, runtime, parity, and determinism gates
passed. No candidate was reselected after the screen.

For this frozen Food-101 panel, fused class-batched three-step prototype
fitting solves the practical failure that motivated the experiment: it matches
the full linear probe's zero nuisance-linear regret, retains OI's nonlinear
advantage, improves top-selection regret relative to refined OI, and costs
only about 11--22% of refined OI or 19--34% of the probe depending on the
regime/estimand.

This supports FUSED3 as the current development winner for Vertebrae's
class-balanced backbone-ranking workflow. A separately frozen, genuinely
untouched backbone panel is still required before treating this as a confirmed
general product result.

## Reproduction

```bash
VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 PYTHONPATH=. \
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  -m experiments.scalable_relevance_kmeans.fused_food_full \
  --output artifacts/scalable_relevance_kmeans/fused_food_full_v1

VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 PYTHONPATH=. \
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  -m experiments.scalable_relevance_kmeans.fused_food_full_analysis \
  --input artifacts/scalable_relevance_kmeans/fused_food_full_v1 \
  --output artifacts/scalable_relevance_kmeans/fused_food_full_analysis_v1
```
