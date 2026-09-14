# Continuous OI mechanism diagnostic v1

This frozen diagnostic asks whether the continuous selectors distinguish
information that improves the conditional mean from information that changes a
target distribution without improving squared-error prediction. It also
separates selector-seed variability from training-sample variability.

The protocol was frozen before outcomes. The terminal run used protocol hash
`a37e2cd0b245cf85e27ccf005195dc1fd8f27fb457d99d52f452c3d5926a0d09`, five
families, three matched representation qualities (`q=0.95, 0.65, 0.35`),
512 training rows, 8192 independent evaluation rows, three fixed folds, three
algorithm seeds, three training-sample seeds, and 32 fixed target permutations.
It produced 45 algorithm panels and 45 training-sample panels, or 270 method
rows across the three selectors.

The methods were current observed continuous OI, existing SAVE OI, and the
locked broad-RFF score. The broad-RFF score is the prototype-free,
null-adjusted multibandwidth kernel-alignment score; it does not fit a
ContinuousOverlapIndex prototype model. All methods used identical rows and
fixed folds within each panel.

The independent predictive reference is the analytic conditional-mean R2 on
the fixed evaluation draw. The empirical conditional-mean reference agreed
with its analytic value within `0.00597` at every row, below the frozen
`0.015` tolerance.

| method | predictive Spearman, every family | predictive top-1, every family | mean algorithm-score SD | max algorithm-score SD | mean sample-score SD | max sample-score SD | variance-only score range | null score range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| observed OI | 1.00 | 3/3 | 0.00750 | 0.01632 | 0.00818 | 0.02159 | 0.00672 | 0.00081 |
| broad RFF | 1.00 | 3/3 | 0.00345 | 0.00824 | 0.00665 | 0.01863 | 0.00777 | 0.00741 |
| SAVE OI | 1.00 | 3/3 | 0.01117 | 0.02284 | 0.00953 | 0.01743 | 0.01229 | 0.00547 |

The variance-only family was a real distributional control. Its population
conditional mean is exactly zero for every candidate, so its population
predictive R2 is exactly zero. Its conditional variance nevertheless changes
with the representation because `Y=(0.4+0.8*abs(U))*epsilon` and the first
coordinate is a noisy measurement of `U`; this is a nonconstant conditional
distribution even though the predictive estimand is fixed. The source-bound
raw artifact records zero analytic predictive R2 for every variance-only
candidate. The null-noise family has both constant conditional mean and no
target-dependent representation distribution.

The predeclared classifier returned the same result for all three methods:

```json
{
  "systematic_estimand_misalignment": false,
  "algorithmic_variance": false,
  "sampling_variance": false
}
```

This classifier uses the frozen predictive Spearman minimum `0.80`, predictive
top-1 minimum `0.67`, material control score-range maximum `0.05`, algorithm
score-SD maximum `0.025`, and training-sample score-SD maximum `0.050`. The
nonzero control ranges are retained above; they are below the predeclared
materiality threshold and are not treated as exactly zero. Since every
variance-only and null candidate has the same predictive truth, any ordering
within those controls is a false predictive ordering; the observed nonzero
ranges are small false-order signals rather than evidence of useful ranking.

The result rules out a simple mechanism explanation for the earlier continuous
real-panel failures. On this controlled one-dimensional measurement problem,
all three methods recover linear, symmetric quadratic, and periodic
conditional-mean orderings, while their variance-only and null score ranges
remain small. Random selector seeds and training samples also remain below the
frozen variance gates.

This is a capability result, not evidence that continuous OI is a sufficient
real-world backbone ranker. The grid is deliberately monotone: each candidate
is a noisy measurement of one latent scalar, and the predictive ordering is
known in advance. It does not reproduce competing high-dimensional geometry,
multiple equivalent representations, or the close candidate ties in the real
panels. The next diagnostic should construct matched representations with the
same analytic predictive R2 but deliberately different global/local geometry,
then test whether the selectors preserve that tie and whether a head-library
envelope breaks it. No method or parameter is promoted from this result.

Terminal artifacts (source-bound to the pre-outcome runner):

- raw results SHA-256: `78300164957d1fc9535642c18e34cfc3dfd6d2ab6cff4dfe81f210543a092ad2`
- summary SHA-256: `1e1f51ff51d5fc4de8ebf122d6f8a1286229de1896b11f40acbc0566798b2dac`
- manifest SHA-256: `52d2ea57bd4d9af3e5c66e7c41853266373f3e1019390cda8cca7e629bf3e4ac`
