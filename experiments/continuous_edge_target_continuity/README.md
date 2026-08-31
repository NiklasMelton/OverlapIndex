# Continuous edge-level target continuity

This frozen experiment replaces prototype-distribution Wasserstein loss with
pairwise target disagreement on fixed feature-space edges. It compares a
directed row-level 5-nearest-neighbor graph with the existing directed
prototype-adjacency graph.

The implementation is experiment-local and does not change public
`ContinuousOverlapIndex` behavior.

```bash
python -m experiments.continuous_edge_target_continuity.run \
  --output artifacts/continuous_edge_target_continuity
```
