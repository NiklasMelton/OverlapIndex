# RFF convergence diagnostic results

Run date: 2026-09-01.

Protocol SHA-256:
`36ded2d6757942a76bffdb115a76d283ddc7620eb2916fe06bf6598edafdf863`.

Terminal artifact hashes:

- raw results: `5528eb47b614a7c78da24f54100e6bc13f998f63998bfb2a02606af433ae8114`
- summary: `29068235e8a9bbf2c2f09c0d53abb207ec4919187af72f2eb18bc4d0e554493f`
- report: `8f298f81cb946aa8043fe4bce1efad948dd24456b71ad182ba45501c8778054b`

The run completed all 45 candidate checkpoints, six panels, 585 geometry
candidate rows, 1,170 method scores, and 2,340 fold geometries. Every
64-feature zero-offset broad/SAVE score reproduced its archived source value
exactly (maximum absolute residual `0.0`). The empirical exact maps reconstructed
their fold-training RBF Gram matrices with maximum relative error
`4.31e-15`.

## Primary result

On the validated nonlinear SmallNORB elevation target:

| Method | Configuration | Mean Spearman | Mean top-1 regret R2 | Exact-best rate |
|---|---|---:|---:|---:|
| linear CV | fixed | 0.8465 | 0.01902 | 0.333 |
| broad RFF | four-map RFF64 ensemble | 0.8465 | 0.00397 | 0.667 |
| broad RFF | four-map RFF128 ensemble | 0.7253 | 0.03768 | 0.333 |
| broad RFF | four-map RFF256 ensemble | 0.6768 | 0.03768 | 0.333 |
| broad RFF | empirical exact RBF span | 0.6848 | 0.07753 | 0.000 |
| SAVE+RFF | four-map RFF64 ensemble | 0.8384 | 0.00397 | 0.667 |
| SAVE+RFF | four-map RFF128 ensemble | 0.6283 | 0.09247 | 0.000 |
| SAVE+RFF | four-map RFF256 ensemble | 0.4828 | 0.04376 | 0.333 |
| SAVE+RFF | empirical exact RBF span | 0.4141 | 0.13592 | 0.000 |

On the strongly linear GTSRB box-area control, linear CV remained perfect.
Broad RFF256 and the broad exact reference also selected the best backbone in
all three panels, while SAVE remained less reliable. Thus higher rank did not
universally destroy linear information, but it clearly failed to improve the
nonlinear ranking that motivated the diagnostic.

## Approximation sanity check

After the frozen run, an outcome-independent geometry check compared each raw
RFF Gram matrix with its exact same-bandwidth RBF Gram matrix. Mean relative
heldout-to-training Gram errors were:

| Task | RFF64 | RFF128 | RFF256 |
|---|---:|---:|---:|
| SmallNORB elevation | 0.1180 | 0.0840 | 0.0594 |
| GTSRB box-area control | 0.1257 | 0.0895 | 0.0632 |

There were 480 SmallNORB and 240 GTSRB fold-map comparisons per rank. The
kernel approximation converged monotonically while ranking degraded. This
rules out insufficient Fourier feature count as the primary explanation for
the observed failure.

## Interpretation

The 64-dimensional sketch is acting as useful statistical regularization. As
the nonlinear feature space grows, the train-only residual correction,
whitening, and target-moment eigensystem must be estimated from only 150 or 375
fold-training rows. The richer map more accurately represents the RBF kernel,
but it also offers more sample-specific directions for the supervised moment
projection to fit. The exact empirical span is the limiting case and performs
worst. SAVE+prototype aggregation adds another sensitive stage and shows the
largest map-seed instability.

This is evidence for an estimator/objective-and-aggregation limitation after
the Fourier map, not a finite-RFF approximation limitation. The exact reference
is still only a finite-training-span diagnostic, so it does not establish a
population-kernel impossibility.

## Recommended next experiment

Do not increase the single-map rank. Freeze a low-rank bagging experiment that
keeps every supervised projection at 64 dimensions and aggregates scalar broad
scores across independent sketches. Compare four and eight 64-feature maps,
mean versus median score aggregation, and a 256-feature single map at matched
feature-computation cost. In parallel, test shrinkage or cross-fitting of the
target-moment operator before revisiting an exact kernel. Broad score bagging is
the leading candidate; SAVE should remain diagnostic until its map/prototype
instability is controlled.

These findings are development-only and do not promote a method.
