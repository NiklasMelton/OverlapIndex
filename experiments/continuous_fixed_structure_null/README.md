# Strict fixed-structure continuous-OI null

This frozen experiment compares the current held-out target permutation null
with a stricter null that freezes the actual held-out row-to-prototype
assignments, supports, and prototype adjacency. Only target values are
permuted before the existing support-weighted adjacency Wasserstein loss is
recomputed.

The implementation is experiment-local and does not change public
`ContinuousOverlapIndex` behavior.

```bash
python -m experiments.continuous_fixed_structure_null.run \
  --output artifacts/continuous_fixed_structure_null
```
