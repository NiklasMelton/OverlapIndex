# Continuous bounded shared evaluator

This experiment removes redundant computation from the confirmed bounded
top-five shortlist evaluator without changing either component score.
SAVE+RFF already contains the exact RFF-mean fold block needed by broad RFF.
The shared evaluator retains that block, fits only the SAVE+RFF prototype model,
and computes broad kernel alignment directly using the same target scaling.

The frozen benchmark requires exact score/rank parity and at least a 25% median
panel runtime reduction across six synthetic target families. It does not alter
the public `ContinuousOverlapIndex` API.

```bash
CONTINUOUS_OI_COMMIT=$(git rev-parse HEAD) PYTHONPATH=. \
  python -m experiments.continuous_bounded_shared_evaluator.run \
  --output artifacts/continuous_bounded_shared_evaluator
```
