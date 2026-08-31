# Residual-only continuous ranking result

This note records the completed frozen exploratory experiment. Numeric outputs
remain ignored under `artifacts/continuous_residual_ranking`.

## Identity and validation

- Implementation commit recorded by the runner: `68b76e1`
- Protocol SHA-256:
  `1a669c8742bcf86279d58a02974b6b1a5b324eef1235ef899d3f4aecb883803c`
- Runner SHA-256:
  `45e962a2e1f65cc65a5a927f2e312870b55d63166caaa829ed2b5a152ad4b84c`
- Common summary-runner SHA-256:
  `90d6c3e5b1effb83eb8bec65ce30200288fd198314f09305124032feccdd6ce4`
- Raw-results SHA-256:
  `7a873b88e8cfd1b9ddf7fc4772e22cf92e6a4ae3f221cf46c40c3c4ecec0ac76`
- Summary SHA-256:
  `04ce615fd73b67aaac4b2326fed40aa98215b0c22db18173f3a85f55778059bc`
- Manifest SHA-256:
  `bb426aa7c9c29400411fea05bd2dfc902176e46f917ebef9739f6e4d56afe394`
- Complete grid: 24 panels, 192 candidate rows, and 24 matching atomic
  checkpoints.

All source and terminal hashes were independently verified. Rebuilding the
summary from the checkpoint panels produced an exact match.

The complete repository suite passes: 305 tests passed and 14 were skipped.
The 34 warnings are existing expected under-prototyped/multilabel diagnostics,
not failures from this experiment.

## Head-diversity gate

The predeclared gate failed narrowly:

| Head | Unique wins | Required | Status |
| --- | ---: | ---: | --- |
| Linear Ridge | 135 | 16 | pass |
| Quadratic Ridge | 28 | 16 | pass |
| kNN | 6 | 8 | **fail** |
| RBF SVR | 23 | 8 | pass |

Therefore the overall aggregate is descriptive and cannot support a general
best-head-envelope claim. Family-level predeclared contrasts remain useful.
The gate correctly exposed that the sinusoid and sawtooth generators still did
not produce stable local-head wins under the frozen high-dimensional nuisance.

## Descriptive aggregate

| Ranker | Mean Spearman | Mean regret (R2) | Exact best | Within 0.01 | Median candidate time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear probe | 0.6339 | 0.1495 | 0.667 | 0.708 | 0.0191 s |
| Full Fisher + relevance | 0.4623 | 0.0761 | 0.458 | 0.625 | 0.1209 s |
| Residual rank 8 + relevance | 0.0794 | 0.0548 | 0.208 | 0.333 | 0.1273 s |
| Residual rank 0 + relevance | 0.0492 | 0.0664 | 0.042 | 0.208 | 0.1221 s |
| Legacy continuous OI | -0.1470 | 0.1327 | 0.000 | 0.125 | 0.1299 s |

Residual rank 8, rank 16, and full residual correction produced exactly the
same OI scores on all 192 rows. Direct inspection showed only four energetic
residual directions; rank 8 already captured every direction with a nonzero
deflation gain. This is useful scalability evidence, not evidence that rank is
irrelevant in general.

## Primary mechanism result

Residual-only correction preserved substantially more nonlinear geometry:

| Family group | Linear probe regret | Full Fisher + relevance | Residual-8 + relevance |
| --- | ---: | ---: | ---: |
| Linear controls | 0.0000 | 0.0000 | 0.0231 |
| Quadratic | 0.5928 | 0.2737 | **0.1286** |
| Local/kernel | 0.0051 | 0.0136 | 0.0102 |
| Nuisance controls | 0.0000 | **0.0172** | 0.0575 |

This confirms the motivating diagnosis. The Fisher target-cell mean-contrast
projection discards useful second-order structure. Stopping before that
projection more than halved quadratic regret relative to full Fisher and cut it
by almost four fifths relative to the linear probe.

The residual transform is not a complete replacement. It damaged linear and
correlated-nuisance ranking, where the discriminant projection is valuable.
Most importantly, its low aggregate regret did not correspond to good ranking:
mean Spearman was only 0.0794 and exact-best rate 0.208. It often selected a
reasonably good candidate but did not order the full backbone panel faithfully.

Target-distance relevance remained secondary. For residual rank 8 it improved
mean regret by 0.0036 and exact-best rate by 0.125, while mean Spearman declined
by 0.0179. For full Fisher it improved Spearman and near-best selection but did
not improve mean regret. Relevance cannot substitute for choosing an
appropriate feature view.

## Conclusion and next experiment

The hypothesis was partly confirmed: residual nuisance correction preserves
quadratic structure that Fisher projection destroys. But neither view alone
meets the ranking objective across linear and nonlinear relationships.

The next experiment should treat them as complementary frozen views:

1. Fit the full-Fisher view and the rank-8 residual view independently inside
   every cross-fitting fold.
2. Produce one OI score from each view.
3. Compare a predeclared 50/50 fractional-midrank fusion and a fixed two-view
   normalized score average against each individual view. Do not tune fusion
   weights from outcomes.
4. Use improved nonlinear panels where the diversity gate must pass before any
   aggregate interpretation. In particular, lower irrelevant dimension for
   the local/kernel families or increase their sample size so kNN/RBF can
   express the intended relationship.
5. Require the fused view to retain full-Fisher performance on linear/nuisance
   controls and residual-view performance on the quadratic group.

This is preferable to forcing a single preprocessing transform to solve two
geometrically different problems.
