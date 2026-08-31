# Continuous Fisher ranking experiments 1 and 2

This note records the completed, frozen exploratory synthetic experiment. The
numeric artifacts remain ignored under `artifacts/`; their identities are
recorded below so the results can be reproduced and checked without changing
the pre-outcome protocol or runner.

## Artifact identity

- Implementation commit used by the runner: `c6895dbf`
- Protocol SHA-256:
  `49e06028bd65928fd1f092ee5f01a558dbbcbeb6b62b8bb1dc88875b291f614f`
- Runner SHA-256:
  `90d6c3e5b1effb83eb8bec65ce30200288fd198314f09305124032feccdd6ce4`
- Raw-results SHA-256:
  `46aafada58d0b408e33b7ed3874aa0dd2a24f3db7116d1bdae5fcae5adff0781`
- Summary SHA-256:
  `25ef25a5d9261bf9811375a60495f1295e85ec6a28f46e9890f99cd511e70108`
- Complete grid: 18 panels and 144 candidate-panel rows.

The summary was independently rebuilt from all checkpoint panels and matched
`summary.json` exactly.

## Main result

| Ranker | Mean Spearman | Mean regret (R2) | Exact best | Within 0.01 R2 | Median candidate time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tuned linear probe | 0.8545 | 0.0734 | 0.778 | 0.889 | 0.0190 s |
| Full Fisher + target relevance | 0.7235 | 0.0748 | 0.667 | 0.778 | 0.1362 s |
| Full Fisher, no target relevance | 0.6759 | 0.0760 | 0.611 | 0.722 | 0.1355 s |
| Rank-8 Fisher + target relevance | 0.2447 | 0.1172 | 0.333 | 0.389 | 0.1430 s |
| Rank-16 Fisher + target relevance | 0.1944 | 0.1271 | 0.444 | 0.500 | 0.1435 s |
| Fused L2 + target relevance | -0.0304 | 0.1378 | 0.167 | 0.167 | 0.1408 s |
| Legacy MiniBatchKMeans COI | 0.0450 | 0.1551 | 0.167 | 0.167 | 0.1527 s |

Target-aware distance relevance improved full Fisher's mean Spearman by 0.0476,
exact-best rate by 0.0556, and within-0.01 rate by 0.0556, with essentially no
change in runtime. Full Fisher was about 11% faster than legacy COI in this
small experiment, but about 7.2 times slower than the tuned linear probe.

## What worked

Full Fisher plus target relevance ranked the linear, heteroscedastic, and
correlated-nuisance panels almost perfectly and incurred zero selection regret
in those families. It also remained competitive on the folded family. This is
direct evidence that train-only target-cell scaling and full nuisance removal
can repair continuous OI under strong nuisance geometry.

The target-distance relevance term made a smaller but consistently useful
contribution once a meaningful metric space had been recovered. It cannot, by
itself, repair a representation whose useful geometry has already been removed.

## What failed and why

The quadratic family was the decisive failure. The best candidate contains a
relationship of the form `y = u * v`. Within a target interval, the conditional
means of `u` and `v` are close to zero by symmetry. A Fisher transform based on
differences between target-cell means therefore sees little discriminant
direction and projects away the second-order sign interaction. Full Fisher plus
target relevance had mean quadratic Spearman -0.0635 and regret 0.4266.

Legacy COI retained more of that nonlinear geometry (quadratic Spearman 0.4524,
regret 0.1796), although it was not reliable enough overall. Target relevance
cannot recover the quadratic interaction after the Fisher projection has
discarded it.

The low-rank Fisher variants did not provide a useful compromise in the rotated
correlated-noise construction. Rank 8 and rank 16 were both substantially worse
than full rank. The randomized low-rank correction is therefore not monotonic
in rank and should not be assumed to preserve the most predictive nonlinear
subspace.

## Scope limitation

The best-head envelope was dominated by linear Ridge: it won 120 of 144
candidate rows, versus 11 for kNN, 10 for RBF SVR, and 3 for quadratic Ridge.
Consequently, the strong aggregate linear-probe result is expected and the
aggregate alone is not a sufficient test of nonlinear-envelope ranking. The
periodic and folded generators did not produce strong, frequent nonlinear-head
wins at this sample size and nuisance level.

The Ridge fits emitted ill-conditioning warnings for some low-regularization
configurations on the correlated-nuisance panels. The frozen grid completed and
selected among all declared configurations, but future panels should use a
numerically stable formulation or explicitly reject unstable fits.

## Recommendation

Do not promote the target-cell Fisher projection as the general continuous-OI
transform. Preserve the useful pieces and ablate the harmful one:

1. Keep row normalization, within-target-cell diagonal residual scaling, and
   optional low-rank nuisance deflation.
2. Add a residual-only transform that stops before the target-cell mean-contrast
   projection, so nonlinear structure remains in the metric space.
3. Cross residual-only/full Fisher with target relevance in a new frozen panel
   suite that requires substantial wins from each nonlinear reference family
   before aggregate interpretation.
4. Advance to real continuous embedding panels only if residual-only scaling
   retains the quadratic advantage while matching full Fisher on the nuisance
   families.

