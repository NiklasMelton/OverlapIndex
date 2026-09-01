# Real-panel OI rank aggregation

This post-outcome diagnostic asks whether broad RFF is retrieving useful candidates that can be ordered more effectively using a simple OI-only consensus with SAVE. It consumes the three completed real-panel raw artifacts and fits no new backbone, head, or OI model.

Every candidate rule first retrieves the complete broad top-10 pool. It then reranks that pool by a fixed weighted average of broad and SAVE ordinal scores. A leave-dataset-out rule chooses the weight using only the other dataset group. The two UTKFace targets stay in one group because they share exact images and embeddings.

Linear-probe performance is used only as an external comparator. It is never an aggregation feature.

```bash
PYTHONPATH=. LOKY_MAX_CPU_COUNT=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/niklasmelton/Library/Caches/pypoetry/virtualenvs/vertebrae-XeUus8FY-py3.9/bin/python \
  -m experiments.continuous_real_rank_aggregation.run
```
