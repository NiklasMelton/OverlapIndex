# Scalable one-update relevance-weighted K-means

This experiment-local implementation replaces the toy's dense
row-by-prototype-by-feature allocation with bounded row tiles. It preserves one
initial K-means fit, one relevance update, one weighted Lloyd update, and the
current fixed-center OI event definition.

```bash
PYTHONPATH=. pytest -q tests/test_scalable_relevance_kmeans.py
```

The bounded Food timing anchor is run only after `protocol.sha256` is present:

```bash
VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
PYTHONPATH=. python -m experiments.scalable_relevance_kmeans.benchmark \
  --output artifacts/scalable_relevance_kmeans/run_v1
```

It measures one excluded warm-up and one call on the first frozen fold for the
archived Food-101 ResNet50 baseline panel at 128 and 640 rows per class. The
anchor is intentionally too small for ranking or promotion claims. This code
does not modify or export public `OverlapIndex` behavior.

## Frozen Food-101 screen

The full retrospective development screen has its own immutable protocol and
does not modify the completed timing-anchor protocol:

```bash
VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
PYTHONPATH=. python -m experiments.scalable_relevance_kmeans.food101_screen \
  --output artifacts/scalable_relevance_kmeans/food101_screen_v1

# The same command with --resume continues only identity-matching checkpoints.

PYTHONPATH=. python -m experiments.scalable_relevance_kmeans.food101_analysis \
  --input artifacts/scalable_relevance_kmeans/food101_screen_v1 \
  --output artifacts/scalable_relevance_kmeans/food101_analysis_v1
```

Use `--smoke` with a separate output directory for the one-panel structural
check. Smoke results are never accepted by the full analysis command.

## Speed-focused v2 experiment

The v2 experiment leaves every v1 source and artifact unchanged. Its first
stage compares float32 preservation, the one-matmul-per-tile scorer, capped
relevance sampling, and omitting the final Lloyd update on a frozen 30-panel
Food-101 grid. The development analysis locks the first eligible variant in
the predeclared order; the full replay cannot substitute a runner-up.

The preserved outcome and provenance checkpoint is in
[`SPEED_RESULTS_V2.md`](SPEED_RESULTS_V2.md). The valid development decision
stopped without an eligible candidate, so the full command below documents the
gated workflow but was not authorized or executed for v2.

```bash
PYTHONPATH=. pytest -q \
  tests/test_scalable_relevance_speed.py \
  tests/test_scalable_speed_experiment.py

VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
PYTHONPATH=. python -m experiments.scalable_relevance_kmeans.speed_experiment \
  development \
  --output artifacts/scalable_relevance_kmeans/speed_development_v1

PYTHONPATH=. python -m experiments.scalable_relevance_kmeans.speed_analysis \
  development \
  --input artifacts/scalable_relevance_kmeans/speed_development_v1 \
  --output artifacts/scalable_relevance_kmeans/speed_development_analysis_v1

VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
PYTHONPATH=. python -m experiments.scalable_relevance_kmeans.speed_experiment \
  full \
  --development-lock \
    artifacts/scalable_relevance_kmeans/speed_development_analysis_v1/development_lock.json \
  --output artifacts/scalable_relevance_kmeans/speed_full_v1

PYTHONPATH=. python -m experiments.scalable_relevance_kmeans.speed_analysis \
  full \
  --input artifacts/scalable_relevance_kmeans/speed_full_v1 \
  --development-lock \
    artifacts/scalable_relevance_kmeans/speed_development_analysis_v1/development_lock.json \
  --output artifacts/scalable_relevance_kmeans/speed_full_analysis_v1
```

All outcomes remain retrospective development evidence. No public constructor,
package export, Vertebrae file, or future confirmation protocol is changed.
