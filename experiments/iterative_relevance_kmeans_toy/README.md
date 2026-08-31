# One-update relevance-weighted K-means toy

This experiment keeps all earlier toy artifacts unchanged. It replaces the
two-stage prototype metric's second complete K-means fit with one supervised
diagonal relevance update and one weighted Lloyd update.

```bash
PYTHONPATH=. pytest -q tests/test_iterative_relevance_kmeans_toy.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
PYTHONPATH=. python -m experiments.iterative_relevance_kmeans_toy.experiment \
  --output artifacts/iterative_relevance_kmeans_toy/run_v1
```

This is development-only and makes no product or real-backbone runtime claim.
