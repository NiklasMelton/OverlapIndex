# Continuous SAVE geometry result

This note records the completed frozen covariance-aware continuous OI
experiment. The target-moment implementations are experiment-local and do not
change public OI behavior.

## Identity and validation

- Implementation commit recorded by the runner: `46188ae`
- Protocol SHA-256:
  `ba2386b730aed5c0fc436890b670ee5516229c3564d58fd78acee2f05cd99702`
- Runner SHA-256:
  `8bfb305269043003f5f6b9911e77aa7bd32b81c4cbc3c92f1c8a51a050eb41d5`
- Geometry SHA-256:
  `8f9fb383f87af27712f705212730040d5985c15c2353480dea5661d0770eac33`
- Raw-results SHA-256:
  `d93c6fc5964c7eed9d8cbdb803e6af5fc88d88d280cd85b455639b6a569ffd67`
- Summary SHA-256:
  `700e6519cf7cdf1cadf3a30a2d7d85f01d984f7b4d420f90aceaff663b76306f`
- Manifest SHA-256:
  `b0ae7a3eac47624347728095302b7566ad19643c7d143f913eca42bbc7432688`
- Complete grid: 18 panels, 288 candidate rows, and 18 matching atomic
  checkpoints.

All source hashes, terminal hashes, checkpoint contents, and summary values
were independently verified. Rebuilding the summary matched exactly. The full
repository suite passes: 316 passed and 14 skipped; the 34 warnings are the
existing under-prototyped/multilabel diagnostics.

## Reference gate passed

Every one of the 18 panels exceeded its frozen best-head R2 floor. Unique head
wins were well distributed:

- linear Ridge: 80;
- quadratic Ridge: 106;
- kNN: 29;
- RBF SVR: 73.

The quadratic minimum was 24 and observed 106. The combined kNN/RBF minimum
was 16 and observed 102. Aggregate comparisons are therefore interpretable.

## Overall result

| Geometry | Mean Spearman | Mean regret R2 | Exact best | Within 0.01 | Median time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear probe | 0.354 | 0.282 | 0.444 | 0.500 | 0.019 s |
| Observed OI | 0.342 | 0.146 | 0.444 | 0.500 | 0.140 s |
| Residual rank 8 | 0.354 | **0.081** | **0.556** | **0.778** | 0.149 s |
| Full Fisher | 0.469 | 0.181 | 0.278 | 0.444 | 0.149 s |
| Mean SDR rank 8 | 0.428 | 0.248 | 0.167 | 0.389 | 0.150 s |
| SAVE rank 8 | **0.591** | 0.153 | 0.167 | 0.611 | 0.149 s |
| Fixed mean+SAVE rank 8 | 0.575 | 0.202 | 0.167 | 0.333 | 0.150 s |
| Oracle quadratic lift | 0.707 | 0.134 | 0.222 | 0.500 | 0.127 s |

SAVE is the strongest deployable geometry for full-panel ordering and clearly
outperformed the linear probe. Residual-only remains best for selecting a
near-best candidate despite its much weaker global ordering. These are
different objectives and must not be conflated.

## Family behavior

Mean Spearman by family:

| Family | Linear probe | Residual-8 | Full Fisher | SAVE-8 | Mean+SAVE-8 | Oracle lift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear | 1.000 | 0.276 | 0.897 | 0.846 | 0.707 | 0.935 |
| Product quadratic | -0.331 | 0.409 | 0.620 | **0.918** | 0.770 | 0.863 |
| Radial quadratic | 0.705 | 0.461 | 0.138 | **0.580** | 0.398 | 0.613 |
| Absolute value | -0.426 | 0.381 | 0.683 | 0.730 | **0.748** | 0.950 |
| Sinusoid | 0.295 | **0.535** | 0.476 | 0.042 | 0.448 | 0.827 |
| Triangle wave | **0.883** | 0.060 | 0.004 | 0.427 | 0.377 | 0.052 |

SAVE successfully extends OI beyond conditional means. At fixed nuisance its
signal-quality Spearman was 1.000 for linear, 0.967 for product quadratic,
0.800 for radial quadratic, and 0.967 for absolute value. This directly
confirms the predicted value of conditional covariance.

SAVE is not universal. It failed on sinusoidal geometry and remained only
moderate on triangle-wave geometry. First and second conditional moments do
not characterize arbitrary periodic/local dependence.

## Fixed mean+SAVE was not the answer

The predeclared equal trace-normalized combination did not dominate SAVE:

- overall rho fell from 0.591 to 0.575;
- regret worsened from 0.153 to 0.202;
- radial rho fell from 0.580 to 0.398;
- product rho fell from 0.918 to 0.770.

This rejects the simplest fixed addition of mean and covariance moment
matrices. Outcome-tuning their relative weight would be post-hoc and was not
attempted.

## Calibration remained secondary

For SAVE, mean Spearman was 0.583 from negative actual loss, 0.578 from
negative loss ratio, and 0.591 from final OI. Normalization did not create its
advantage or its periodic failure. Mean+SAVE raw actual-loss rho was 0.595,
only modestly above final rho 0.575.

## Interpretation

The experiment establishes that conditional covariance is a major missing
piece in mean-based Fisher geometry. It produces a substantially more general
ranker and beats the linear probe on a balanced, strongly predictive synthetic
head envelope.

It does not establish a production-ready universal transform:

- periodic and local relationships require information beyond the first two
  conditional moments;
- SAVE had a large worst-panel regret despite strong average ordering;
- residual-only and SAVE still optimize different practical behaviors;
- all evidence remains synthetic.

## Next recommendation

Do not promote fixed mean+SAVE or jump immediately to score fusion. Test a
single target-independent nonlinear feature map followed by SAVE:

1. apply the existing row normalization and residual correction;
2. construct a fixed-seed random Fourier or Nyström feature sketch with a
   predeclared dimension and bandwidth rule;
3. fit regularized SAVE in that sketch inside every training fold;
4. compare RFF+SAVE with SAVE, residual-only, the quadratic-lift ceiling, and
   the linear probe;
5. require preservation of SAVE's linear/quadratic results and improvement on
   sinusoid/triangle families without weight tuning.

A characteristic nonlinear feature map can represent dependence beyond means
and covariances while retaining one OI geometry. This is now the most plausible
route to a universal approach; a two-score hybrid remains the fallback if that
experiment fails.
