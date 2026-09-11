# Continuous residual-anchor n=300 diagnostic

This experiment tests whether continuous OI can add nonlinear evidence without
discarding the raw linear signal captured by an outer-training Ridge anchor.
For every outer fold, a three-fold out-of-fold Ridge prediction on the outer
training rows defines a leakage-safe residual target.  SAVE/RFF/prototype
geometry is fitted only to those residual targets.  A deterministic inner
selection split chooses a correction shrinkage from `{0, .25, .5, .75, 1}`;
zero is an explicit no-harm baseline option.  The selected correction is then
fit on all outer-training residuals and added to the anchor prediction on the
untouched holdout.

The roster, source-selected targets, scout backbones, n=300 cohort, outer
folds, source truth and all controls are inherited from the frozen information
ladder.  This package is diagnostic-only: it does not change OverlapIndex or
Vertebrae and its outcomes must not be promoted without independent review.

The outer anchor's Ridge CV uses the parent ladder's `INNER_SEED_OFFSET=1729`
so its predictions and selected alpha are bit-for-bit comparable with the
parent anchor.  Residual OOF and nested calibration use separately named,
frozen stage offsets (the residual-OOF and calibration-split offsets are both
`1103`; nested OOF is `701`, nested anchor `907`, geometry `1301`, target
permutation geometry `1601`, and original-y geometry `1701`).  These are
recorded in `protocol.json` and are not interchangeable with the outer-anchor
offset.
`no_harm_option_present` means that literal zero is available in every frozen
shrinkage grid.  The observed frequency with which selection chooses zero is
reported separately as `zero_shrinkage_selection_fraction_by_method`.

Run only after the pre-outcome review gate:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_MAX_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
LOKY_MAX_CPU_COUNT=1 PYTHONPYCACHEPREFIX=/private/tmp/coi-residual-pycache \\
PYTHONPATH=. python -m experiments.continuous_residual_anchor_n300.run
```

`--validate-only` validates an existing terminal artifact without executing
outcomes.  There is no reserved-backbone or future-v2 authorization in this
diagnostic.
