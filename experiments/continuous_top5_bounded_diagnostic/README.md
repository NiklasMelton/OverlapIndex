# Continuous bounded top-five diagnostic

This post-outcome diagnostic applies the already-frozen dynamic `bounded_2k`
ranking rule at `k=5` to the 30 stored unused-seed confirmation panels. It does
not refit any model. The broad-RFF winner is preserved; the remaining members
are selected by SAVE+RFF only from the complete broad-RFF top-10 retrieval pool.

The diagnostic asks whether bounding winner filling preserves the average
top-five gain while avoiding the triangle-wave/family degradation that caused
the unbounded confirmation to fail. A pass can justify a new unused-seed
confirmation only; it is not promotion evidence.

Run after freezing the implementation commit:

```bash
CONTINUOUS_OI_COMMIT=$(git rev-parse HEAD) PYTHONPATH=. \
  python -m experiments.continuous_top5_bounded_diagnostic.run \
  --output artifacts/continuous_top5_bounded_diagnostic
```
