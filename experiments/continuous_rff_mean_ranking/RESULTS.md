# Continuous kernel-mean result

This note records the completed frozen RFF target-mean experiment. All new
geometry remains experiment-local and leaves public OI behavior unchanged.

## Identity and validation

- Frozen implementation commit: `517635b`
- Protocol SHA-256:
  `e440c2a7bc5096c7f225507b3c70db45cd2ffcd7574979a9dd963444746969cb`
- Runner SHA-256:
  `fd1e4ce42e9caaccbee41afee7d6bda305d4d4914055092588927e98d989a3f2`
- Raw-results SHA-256:
  `3e27431430055cda9b077241d2ff255c78691de6c0205453152157ecd7862528`
- Summary SHA-256:
  `e2667ac44e4d465b4d20cd1916ea07459895cee0221f4112dc37fa293901a906`
- Manifest SHA-256:
  `8a2ca741126a79a05f71ab8353a45b6acab376bf05e4f09d2fcd567f6ebce350`
- Complete grid: 18 panels, 288 candidate rows, 2,016 selector rows,
  and 18 matching atomic checkpoints.

Every source/file hash, checkpoint panel, row count, and finite selector score
was independently verified. Rebuilding the summary from raw panels matched
exactly.

## Reference gate passed

All 18 panels exceeded their frozen best-head R2 floors. Unique reference-head
wins remained linear Ridge 80, quadratic Ridge 106, kNN 29, and RBF SVR 73.

## Universal gate failed, but the mechanism was partially validated

Neither kernel target-moment method passed the predeclared universal gate.
Both passed the regret constraint and sinusoid-improvement constraint. Both
failed overall ordering, preservation of the raw-space SAVE families, and the
triangle-wave improvement requirement.

| Geometry | Mean Spearman | Mean regret R2 | Exact best | Within 0.01 | Median time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear probe | 0.354 | 0.282 | 0.444 | 0.500 | 0.015 s |
| Observed OI | 0.342 | 0.146 | 0.444 | 0.500 | 0.124 s |
| Residual rank 8 | 0.354 | **0.081** | **0.556** | **0.778** | 0.132 s |
| SAVE rank 8 | **0.591** | 0.153 | 0.167 | 0.611 | 0.132 s |
| RFF-64 only | 0.312 | 0.093 | **0.556** | 0.611 | 0.129 s |
| RFF-64 mean | 0.382 | 0.129 | 0.500 | 0.611 | 0.136 s |
| RFF-64 mean+SAVE | 0.415 | **0.126** | 0.500 | 0.611 | 0.135 s |
| Oracle quadratic lift | 0.707 | 0.134 | 0.222 | 0.500 | 0.113 s |

Kernel means were much more effective than covariance-only SAVE in RFF space:
the earlier RFF-64+SAVE method reached only rho 0.336 with regret 0.218. The
direct mean method improved both, and the fixed mean+SAVE combination improved
them again without weight tuning.

## Family behavior

Mean Spearman by family:

| Geometry | Linear | Product | Radial | Absolute | Sinusoid | Triangle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear probe | **1.000** | -0.331 | **0.705** | -0.426 | 0.295 | **0.883** |
| Residual-8 | 0.276 | 0.409 | 0.461 | 0.381 | 0.535 | 0.060 |
| SAVE-8 | 0.846 | **0.918** | 0.580 | **0.730** | 0.042 | 0.427 |
| RFF-64 only | 0.132 | 0.175 | 0.334 | 0.362 | 0.693 | 0.177 |
| RFF-64 mean | 0.590 | 0.316 | 0.252 | 0.331 | 0.690 | 0.115 |
| RFF-64 mean+SAVE | 0.531 | 0.309 | 0.402 | 0.445 | **0.742** | 0.059 |
| Oracle lift | 0.935 | 0.863 | 0.613 | 0.950 | 0.827 | 0.052 |

The direct kernel-mean prediction was correct for sinusoid: rho rose from
0.042 under raw-space SAVE to 0.690, and fixed kernel mean+SAVE reached 0.742.
The same projection did not retain raw-space SAVE's strong product, radial, or
absolute-value ordering. Triangle wave remained unresolved.

The fixed mean+SAVE variant was better than kernel mean alone on overall rho,
regret, radial, absolute, and sinusoid behavior. This is useful evidence that
mean and covariance information are complementary. It still combines them
inside one RFF-space rank-8 projection, where they compete for the same limited
subspace.

## Calibration is not the main failure

For RFF mean+SAVE, negative actual-loss rho was 0.504 versus final OI rho
0.415; negative loss-ratio rho was 0.400. Calibration costs some ordering but
cannot recover the gap to raw-space SAVE or solve triangle wave. For RFF mean,
the corresponding values were 0.481, 0.379, and 0.382.

## Interpretation

The experiment validates kernel conditional means as the right statistic for
the sinusoidal relationship, but rejects replacing raw-space SAVE with one
kernel-space projection. A universal characteristic feature map does not imply
that a finite, globally bandwidthed, rank-8 projection will preserve every
useful geometry. Raw conditional covariance and kernel conditional means carry
different signal, and compressing either one away is the recurring failure.

Triangle wave is a distinct warning. Neither RFF-64, kernel means, kernel
SAVE, nor the quadratic lift ranks it well, while the linear probe's candidate
ordering is excellent. That family likely needs a local/multiscale view or an
unprojected geometry rather than another global target-moment operator.

## Next recommendation

Stop replacing one geometry with another. Test a fixed block-preserving
multi-view metric:

1. fit raw-space SAVE-8 and RFF-64 mean-8 independently inside each fold;
2. retain an observed/residual block for local geometry;
3. normalize each block to a fixed equal norm and concatenate them, with no
   learned or outcome-tuned weights;
4. run one FusedKMeans/OI score in the concatenated space;
5. ablate SAVE+RFF-mean versus SAVE+RFF-mean+observed/residual and require the
   same frozen family-preservation and periodic-improvement gates.

This remains a single OI geometry, not score fusion, while preventing the
linear/quadratic and periodic/local signals from competing for one rank-8
projection. If fixed block concatenation still fails, the evidence favors an
explicit multi-view score with a stated aggregation rule rather than a
universal Euclidean embedding.
