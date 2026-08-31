# Continuous RFF-SAVE result

This note records the completed frozen random-feature SAVE experiment. The
implementation is experiment-local and does not change public OI behavior.

## Identity and validation

- Frozen implementation commit: `199cadf`
- Protocol SHA-256:
  `d5b33337e68bdffdbed5312c204d0cd1349c31e453da1d9f09a1f6209929aaaa`
- Runner SHA-256:
  `1f9f89f861137a2c51e14ef51604a729bbac8f6635bfd1b4f40c4280c7e1d524`
- Geometry SHA-256:
  `e47387cee364e6fe000e034140ecd8f70bc1c6148f3b47dab88c138b05f60ca2`
- Raw-results SHA-256:
  `5c2a6e2f000bd5e65304ac7ab188a458fedb42aabdab6a3825cea2bb4bb443d3`
- Summary SHA-256:
  `08c8647fff5645aaa0e2f4212a210e41eb83b7b9d45c64086bfbdfd0c22d7b61`
- Manifest SHA-256:
  `85d3bf39b51cf3b0577d6952c56afbd101564ee1a37eb67998ce851e6360593f`
- Complete grid: 18 panels, 288 candidate rows, 2,016 selector rows,
  and 18 matching atomic checkpoints.

All terminal hashes, source hashes, checkpoint panels, finite selector scores,
and table counts were independently verified. Rebuilding the complete summary
from the raw panels matched exactly.

## Reference gate passed

All 18 panels exceeded their frozen best-head R2 floors. Unique head wins were
unchanged from the parent SAVE experiment: linear Ridge 80, quadratic Ridge
106, kNN 29, and RBF SVR 73. The result therefore exercises the intended
mixture of linear, quadratic, and local reference behavior.

## Predeclared RFF-SAVE gate failed

Neither frozen RFF-SAVE dimension passed. RFF-64 plus SAVE failed seven of the
eight checks; its only success was the required sinusoid improvement. RFF-128
plus SAVE failed all eight checks.

| Geometry | Mean Spearman | Mean regret R2 | Exact best | Within 0.01 | Median time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear probe | 0.354 | 0.282 | 0.444 | 0.500 | 0.015 s |
| Observed OI | 0.342 | 0.146 | 0.444 | 0.500 | 0.123 s |
| Residual rank 8 | 0.354 | **0.081** | **0.556** | **0.778** | 0.131 s |
| SAVE rank 8 | **0.591** | 0.153 | 0.167 | 0.611 | 0.130 s |
| RFF-64 only | 0.312 | 0.093 | **0.556** | 0.611 | 0.128 s |
| RFF-64 + SAVE-8 | 0.336 | 0.218 | 0.167 | 0.333 | 0.130 s |
| RFF-128 + SAVE-8 | 0.180 | 0.221 | 0.278 | 0.333 | 0.135 s |
| Oracle quadratic lift | 0.707 | 0.134 | 0.222 | 0.500 | 0.111 s |

The nonlinear sketch was not computationally expensive on these small panels;
the failure is statistical/geometric rather than a runtime failure.

## Family behavior

Mean Spearman by family:

| Geometry | Linear | Product | Radial | Absolute | Sinusoid | Triangle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear probe | 1.000 | -0.331 | 0.705 | -0.426 | 0.295 | **0.883** |
| Residual-8 | 0.276 | 0.409 | 0.461 | 0.381 | 0.535 | 0.060 |
| SAVE-8 | **0.846** | **0.918** | **0.580** | 0.730 | 0.042 | 0.427 |
| RFF-64 only | 0.132 | 0.175 | 0.334 | 0.362 | **0.693** | 0.177 |
| RFF-64 + SAVE-8 | 0.276 | 0.574 | 0.344 | 0.347 | 0.329 | 0.147 |
| RFF-128 + SAVE-8 | 0.230 | 0.306 | 0.025 | 0.380 | 0.044 | 0.095 |
| Oracle lift | 0.935 | 0.863 | 0.613 | **0.950** | 0.827 | 0.052 |

RFF-64 alone is an informative isolation result. It improved sinusoid ranking
substantially and achieved low overall selection regret, showing that the
nonlinear representation contains useful local information. It did not
preserve the strong SAVE ordering on the first/second-moment families and did
not solve triangle-wave ranking.

Applying SAVE after RFF was destructive. RFF-64 plus SAVE recovered some
sinusoid signal but lost linear, product, radial, and absolute-value ordering.
Doubling the sketch dimension made nearly every family worse. The
fixed-nuisance quality Spearman fell from 0.664 for SAVE to 0.312 for
RFF-64+SAVE and 0.101 for RFF-128+SAVE.

## Interpretation

The tested composition is not a universal continuous-OI geometry. There are
two compatible reasons:

1. A characteristic RBF feature map represents nonlinear dependence primarily
   through conditional feature means. Applying covariance-only SAVE in that
   expanded space is an indirect, higher-order statistic and discards the most
   direct kernel mean signal.
2. Estimating and whitening 64- or 128-dimensional conditional covariance from
   roughly 375 training-fold rows and eight target cells is noisy. The worse
   128-feature result is the expected direction if this variance dominates.

The result does not say random features are useless. It rejects the specific
`RFF -> pooled residual correction -> whitening -> covariance-only SAVE`
composition at the two frozen dimensions and median bandwidth.

## Next recommendation

Test kernel mean geometry, not another SAVE dimension or bandwidth sweep:

1. keep the exact train-only median-bandwidth RFF map;
2. estimate target-cell conditional means in RFF space with a regularized
   low-rank projection (kernel SIR / RFF-mean);
3. compare RFF-mean alone with a predeclared normalized RFF-mean+SAVE variant;
4. include SAVE-8, RFF-64 only, residual-8, and the oracle lift unchanged;
5. require the same preservation and periodic-family gates.

This is the mathematically direct use of a characteristic feature map: target
dependence becomes a mean contrast in feature space. If it also fails, the
evidence favors a deliberately structured multi-view score over one universal
projection.
