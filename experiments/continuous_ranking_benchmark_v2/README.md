# Continuous ranking benchmark V2 evaluator pilot

This bounded pilot validates the downstream best-head ranking target before a
larger continuous OverlapIndex benchmark is built. It reuses the exact cached
five-backbone embeddings and fixed 200-image evaluation cohorts for CUB leg,
CUB wing, and LSP wrist from the completed V1 diagnostic.

The evaluator corrects two V1 problems. Every nominal training label now names
a distinct sample, and row normalization, PCA-32, and scaling are fitted inside
each inner fold. It retains the historical linear Ridge, degree-two Ridge, kNN,
and RBF-SVR grids to isolate the effect of those corrections. This is an
evaluator pilot, so it neither fits nor selects a continuous OI method.

CUB targets use ten deterministic, target-balanced draws of unique samples.
The recovered LSP pool contains exactly 590 embedded eligible samples, so its
pilot reference uses that single complete training cohort; uncertainty there
comes only from paired evaluation-row resampling. The limitation is explicit.

The synthetic panel contains linear, quadratic, and smooth-periodic mechanisms.
For nonlinear mechanisms, it includes a clean nonlinear representation and a
weaker representation with a direct linear shortcut. This ensures that passing
the control requires the best-head envelope to identify useful nonlinear
geometry rather than merely order a common signal-to-noise ladder.

Run with one numerical thread:

```text
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
python -m experiments.continuous_ranking_benchmark_v2.run
```

The full benchmark proposal is in `DESIGN.md`. Pilot outputs are diagnostic and
cannot authorize method promotion or confirmation execution.

