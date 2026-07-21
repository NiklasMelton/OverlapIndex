# Backend guide

Every backend builds prototypes owned by individual labels. OverlapIndex then
uses a common scoring interface, so the public `fit`, `score`, and `predict`
methods stay the same. The backend changes how prototypes are constructed and
whether state can be updated incrementally.

## Choosing a backend

| Backend | Update model | Sparse `X` | Multi-label | Best fit |
| --- | --- | --- | --- | --- |
| {doc}`minibatch_kmeans` | Offline/refit | Yes | Yes | Default choice and large batch datasets |
| {doc}`kmeans` | Offline/refit | Yes | Yes | Smaller datasets where full KMeans is affordable |
| {doc}`ballcover` | Offline/refit | No | Yes | Shape- or support-sensitive geometry |
| {doc}`fuzzy_artmap` | Incremental | No | No online multi-label path | Streaming bounded features with Fuzzy ART prototypes |
| {doc}`hypersphere_artmap` | Incremental | No | No online multi-label path | Streaming data with radius-based prototypes |

Start with MiniBatchKMeans unless the use case provides a specific reason to
choose another backend. Backend scores are not automatically interchangeable:
prototype geometry and granularity affect the measured overlap. Compare models
using the same backend and settings.

## Shared requirements

- Validate and normalize features before fitting. The package does not apply a
  general-purpose scaler.
- Use at least two useful prototypes per label for a well-resolved top-two
  overlap comparison. Inspect `under_prototyped_labels_` after fitting.
- `fit(X, y)` always starts a fresh fit. For offline backends,
  `partial_fit(X, y)` also refits on only the provided batch.
- Only Fuzzy and Hypersphere ARTMAP preserve learned state across
  `partial_fit` calls.
- Set backend random seeds when results need to be reproducible.
- For very large ARTMAP runs, a lower `rho` in roughly the `0.5` to `0.7`
  range can reduce prototype growth and improve runtime. Treat that range as a
  starting point and validate the resulting resolution on representative data.

```{toctree}
:hidden:
:maxdepth: 1

minibatch_kmeans
kmeans
ballcover
fuzzy_artmap
hypersphere_artmap
```
