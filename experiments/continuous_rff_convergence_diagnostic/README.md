# Continuous RFF convergence diagnostic

This development-only diagnostic asks a narrow question: are the ranking errors
of the current continuous broad-RFF and SAVE+RFF selectors caused by using only
64 random Fourier features, or do they persist when random-feature error is
reduced?

It reuses immutable downstream-head outcomes. It does not refit or retune any
head. The selector is evaluated on the validated nonlinear SmallNORB elevation
target and on GTSRB bounding-box area as a strongly linear control.

The frozen comparison is:

- 64, 128, and 256 Fourier features;
- four predeclared independent map offsets at every feature count;
- candidate scores averaged across maps before ranking;
- one empirical exact-RBF training-span reference; and
- the already-recorded linear-CV ranking as a comparator.

Only the Fourier map varies across map offsets. Fold splits, target cells,
target-moment fitting seeds, fused-prototype fitting seeds, and permutation
banks are fixed. The 64-feature zero-offset score must reproduce the prior
artifacts exactly within `1e-12`.

Run with the archived Python 3.9 environment:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
PYTHONPYCACHEPREFIX=/tmp/oi_structured_pycache \
python -m experiments.continuous_rff_convergence_diagnostic.run
```

The empirical exact reference reconstructs the fold-training RBF Gram matrix
and Nyström-projects held-out rows into that training span. It is a finite-data
diagnostic, not a population-kernel claim. Nothing in this experiment is
eligible for method promotion.

See `RESULTS.md` for the completed diagnostic and interpretation.
