# Continuous block-preserving multi-view result

This note records the completed frozen equal-block multi-view experiment. The
implementation is experiment-local and does not change public OI behavior.

## Identity and validation

- Frozen implementation commit: `8ece684`
- Protocol SHA-256:
  `574f7e3d13b86601a40aa810c37d57082059999fc7b9947bd0c917aabc074e16`
- Runner SHA-256:
  `f24187328dd149650a31818b73ed21f01ef9d2ea31f0f52025608fc6fd5d3490`
- Raw-results SHA-256:
  `c4caa102fe10790061593abc88d9f29e52df2237402c8ccc71b5a6105d4e036d`
- Summary SHA-256:
  `92f1fded3221796187dd38cd7b8601958118236d807ab18de04a47715f352197`
- Manifest SHA-256:
  `009bb1f6e1ab2493d4f64ad880ae5dd30fa44d7999d636efdde82ad9ebbafa94`
- Complete grid: 18 panels, 288 candidate rows, 2,304 selector rows,
  and 18 matching atomic checkpoints.

All terminal/source hashes, checkpoint panels, row counts, method sets, and
finite selector scores were independently verified. Rebuilding the complete
summary from raw panels matched exactly.

## Reference gate passed

All 18 panels exceeded their frozen best-head R2 floors. Unique head wins were
linear Ridge 80, quadratic Ridge 106, kNN 29, and RBF SVR 73.

## The method moved substantially closer but did not pass

No frozen block set passed the universal gate. The two-block SAVE+RFF-mean
metric was strongest on overall ordering. The residual three-block metric was
strongest on selection and fixed-nuisance quality ordering.

| Geometry | Mean Spearman | Mean regret R2 | Exact best | Within 0.01 | Median time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear probe | 0.354 | 0.282 | 0.444 | 0.500 | 0.015 s |
| Observed OI | 0.342 | 0.146 | 0.444 | 0.500 | 0.124 s |
| Residual rank 8 | 0.354 | **0.081** | 0.556 | **0.778** | 0.131 s |
| SAVE rank 8 | **0.591** | 0.153 | 0.167 | 0.611 | 0.131 s |
| RFF-64 mean | 0.382 | 0.129 | 0.500 | 0.611 | 0.135 s |
| SAVE + RFF-mean | 0.551 | **0.106** | 0.556 | 0.667 | 0.152 s |
| SAVE + RFF-mean + observed | 0.502 | 0.127 | 0.556 | 0.667 | 0.149 s |
| SAVE + RFF-mean + residual | 0.545 | 0.108 | **0.667** | **0.778** | 0.154 s |
| Oracle quadratic lift | 0.707 | 0.134 | 0.222 | 0.500 | 0.112 s |

Equal-block concatenation recovered much of SAVE's ordering while preserving
the kernel mean's sinusoid signal. Compared with kernel mean alone, the
two-block metric increased rho from 0.382 to 0.551 and reduced regret from
0.129 to 0.106. It also improved over raw SAVE's regret by 0.047.

## Family behavior

Mean Spearman by family:

| Geometry | Linear | Product | Radial | Absolute | Sinusoid | Triangle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear probe | **1.000** | -0.331 | **0.705** | -0.426 | 0.295 | **0.883** |
| Residual-8 | 0.276 | 0.409 | 0.461 | 0.381 | 0.535 | 0.060 |
| SAVE-8 | **0.846** | **0.918** | **0.580** | **0.730** | 0.042 | 0.427 |
| RFF-64 mean | 0.590 | 0.316 | 0.252 | 0.331 | 0.690 | 0.115 |
| SAVE + RFF-mean | 0.789 | 0.713 | 0.542 | 0.707 | 0.725 | -0.173 |
| + observed | 0.675 | 0.590 | 0.516 | 0.631 | **0.790** | -0.192 |
| + residual | 0.682 | 0.617 | 0.559 | 0.655 | 0.752 | 0.004 |
| Oracle lift | 0.935 | 0.863 | 0.613 | 0.950 | 0.827 | 0.052 |

The two-block metric passed the frozen linear, radial, absolute, sinusoid, and
regret checks. It missed product preservation and triangle improvement, and
its overall rho of 0.551 remained below SAVE's 0.591. Adding observed or
residual geometry did not fix those failures.

The residual three-block metric had the strongest quality-at-fixed-nuisance
rho (0.737), exact-best rate (0.667), and within-0.01 rate (0.778). This is
practically attractive, but it does not satisfy the predeclared full-ranking
objective.

## Triangle wave is now the isolated blocker

The triangle family behaved differently from the other five. Its envelope has
one sharply favorable low-nuisance/high-quality cell, while every multi-view OI
score stayed near zero and was dominated by small stochastic/geometric
changes. The two-block triangle quality-at-fixed-nuisance rho was 0.059 and its
nuisance-at-fixed-quality rho was -0.183.

The frozen selector uses exactly four prototypes in each of eight target
cells, and every triangle fold produced all 32 prototypes. The triangle target
contains roughly three periods over the latent interval, so a target value can
have about six spatially separated inverse branches. Four prototypes per cell
cannot represent all of those branches. By contrast, the shorter sinusoid has
at most about three major inverse branches and is successfully recovered by
the same kernel-mean/multi-view geometry.

This is a concrete capacity hypothesis rather than a reason to reinterpret the
failed gate. It predicts that increasing prototypes per target cell should
help triangle much more than sinusoid or the first/second-moment families.

## Calibration is secondary

For the two-block metric, negative actual-loss rho was 0.553, negative
loss-ratio rho 0.537, and final OI rho 0.551. Calibration is not responsible
for its remaining failure. The corresponding residual three-block values were
0.524, 0.536, and 0.545.

## Next recommendation

Run a targeted, frozen prototype-capacity diagnostic before changing the
metric again:

1. compare k=4 and k=8 prototypes per target cell for SAVE, RFF-mean,
   SAVE+RFF-mean, and SAVE+RFF-mean+residual;
2. keep every geometry, fold, seed, bandwidth, target cell, and gate unchanged;
3. predeclare triangle-wave improvement as the mechanism check and require no
   material regression on the other five families;
4. record prototype occupancy/support and runtime to determine whether extra
   prototypes represent inverse branches or merely add variance.

If k=8 selectively repairs triangle while preserving the current multi-view
strengths, the block-preserving metric becomes a credible universal candidate.
If it does not, the triangle family requires a different adjacency or local
aggregation rule rather than more preprocessing.
