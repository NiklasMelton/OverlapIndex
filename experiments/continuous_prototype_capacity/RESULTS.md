# Continuous prototype-capacity result

This note records the completed frozen k=4 versus k=8 experiment. It changes
only FusedKMeans prototype capacity inside the experiment runner and leaves
public OI behavior unchanged.

## Identity and validation

- Frozen implementation commit: `92befc2`
- Protocol SHA-256:
  `f5a3e99b0309c22ae9dffde0a894e77fc44896c53bb3616bd1f842f8af2dd9bd`
- Runner SHA-256:
  `cdfff766c825411b376d8d2a92f3ead6a98e6bd0c9492be88712238ff3b83567`
- Raw-results SHA-256:
  `811a853635cdc4ccda726a8c2a68e2cbc3cbe83ff044d2c28c9077cf4143d24d`
- Summary SHA-256:
  `a17de9b120b47c1f64b4d536b90ec978b8d3255719aa2ddb2a8dbe1c8aed5ac4`
- Manifest SHA-256:
  `e50612bd391ed525fa3a042f104b35ec9cdae69b45063ab2de1723bc6dc1a58a`
- Complete grid: 18 panels, 288 candidate rows, 2,304 selector rows,
  9,216 fold rows, and 18 matching atomic checkpoints.

All source/file hashes, checkpoint panels, method sets, finite scores, and
prototype-support totals were independently verified. Every fold's assignment
support summed exactly to its 375 training rows. Rebuilding the summary from
raw panels matched exactly.

## Reference gate passed

All 18 panels exceeded their frozen best-head R2 floors. Unique head wins were
linear Ridge 80, quadratic Ridge 106, kNN 29, and RBF SVR 73.

## Capacity hypothesis rejected

No k=8 method passed the paired mechanism gate or the unchanged universal
gate. Triangle-wave rho decreased for every geometry:

| Geometry | k=4 triangle rho | k=8 triangle rho | Change |
| --- | ---: | ---: | ---: |
| SAVE | 0.427 | 0.032 | -0.394 |
| RFF-mean | 0.115 | -0.086 | -0.201 |
| SAVE + RFF-mean | -0.173 | -0.213 | -0.040 |
| SAVE + RFF-mean + residual | 0.004 | -0.040 | -0.043 |

The frozen mechanism required an improvement of at least 0.30 while losing at
most 0.10 in each other family. Every pair failed the triangle criterion.

## Overall results

| Geometry | k | Mean Spearman | Mean regret R2 | Exact best | Within 0.01 | Median time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SAVE | 4 | **0.591** | 0.153 | 0.167 | 0.611 | 0.132 s |
| SAVE | 8 | 0.468 | 0.267 | 0.167 | 0.333 | 0.148 s |
| RFF-mean | 4 | 0.382 | 0.129 | 0.500 | 0.611 | 0.136 s |
| RFF-mean | 8 | 0.382 | 0.118 | 0.611 | 0.667 | 0.153 s |
| SAVE + RFF-mean | 4 | **0.551** | **0.106** | 0.556 | 0.667 | 0.152 s |
| SAVE + RFF-mean | 8 | 0.548 | 0.120 | 0.556 | 0.667 | 0.167 s |
| SAVE + RFF-mean + residual | 4 | 0.545 | 0.108 | **0.667** | **0.778** | 0.154 s |
| SAVE + RFF-mean + residual | 8 | 0.521 | **0.104** | 0.611 | 0.722 | 0.169 s |

k=8 was not uniformly harmful. It improved SAVE's product rho from 0.918 to
0.948, RFF-mean's radial/absolute behavior, and the two multi-view metrics on
product, radial, and absolute-value families. It did not produce the selective
periodic repair predicted by the branch-capacity hypothesis.

The k=4 two-block method remains the best balance of ordering and regret.

## Prototype diagnostics rule out failed resolution

All k=8 folds resolved all eight prototypes in all eight target cells, for 64
total prototypes. Post-fit assignment occupied every prototype in every k=8
method. Median support fell from 11--11.5 rows under k=4 to 5--6 rows under
k=8; minimum support was one row in both settings. Only SAVE k=8 ever reported
an empty prototype during an intermediate Lloyd iteration, with a maximum of
one, and final assignment still occupied all prototypes.

The result is therefore not explained by density capping, empty final
prototypes, or missing training coverage. The additional prototypes genuinely
partitioned the folds and still did not expose triangle ordering.

## Why extra prototypes did not help

For the two-block metric, k=8 reduced the median triangle actual loss from
1.205 to 1.044, but also reduced its refit-permutation null loss from 1.211 to
1.053. The normalized OI score therefore remained near zero. More capacity
benefited the observed and null structures almost equally.

The triangle failure is not merely a cluster-count bottleneck. Continuous OI
scores whether prototype adjacency carries more target-distribution coherence
than a target-refit permutation. Under repeated periodic branches, added local
prototypes do not create enough excess coherence over that null. Fragmenting
supports to roughly five rows also increases empirical Wasserstein variance.

Component evidence reinforces this diagnosis. SAVE k=4 triangle rho was 0.521
from negative actual loss but only 0.427 after null normalization. RFF-mean k=4
was 0.564 from negative actual loss but 0.115 after normalization. The target
signal exists in local loss for those views; candidate-dependent null behavior
removes or destabilizes much of its ordering.

## Next recommendation

Stop increasing prototype count or changing preprocessing. Run a scoring-stage
diagnostic on the frozen k=4 geometries:

1. retain exact actual adjacency loss, current four-permutation refit null, and
   final OI score;
2. add 16- and 32-permutation refit nulls with common, frozen permutation banks
   across candidates in each panel;
3. report ranking from negative actual loss, loss ratio, and final bounded OI;
4. measure the variance and candidate correlation of the null estimate,
   especially for triangle and sinusoid;
5. do not promote negative actual loss as a replacement metric unless it
   preserves the other families under a separately frozen experiment.

If more precise paired nulls recover triangle ordering, the issue is null
estimation variance. If they do not, the candidate-dependent refit null is
normalizing away real periodic structure, and a different continuous-OI null
or local target-continuity estimand is required. That would be a scoring change,
not another geometry preprocessing change.
