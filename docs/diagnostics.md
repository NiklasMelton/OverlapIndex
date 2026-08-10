# Diagnostics and troubleshooting

A global score should be the start of interpretation, not the end. This page
maps fitted attributes and common warnings back to useful checks.

## Discrete estimator attributes

| Attribute | Meaning |
| --- | --- |
| `index` | Macro mean of evaluable, non-excluded per-label scores. |
| `weighted_index` | Per-label scores weighted by positive support. |
| `singleton_index[label]` | Label's minimum directional pairwise score. |
| `pairwise_index[(a, b)]` | Directional score for source `a` against `b`. |
| `pairwise_cardinality[(a, b)]` | Multi-label denominator for the pair. |
| `cluster_cardinality[label]` | Positive sample support for the label. |
| `rev_map[label]` | Global prototype IDs owned by the label. |
| `competitors_[label]` | Selected multi-label competitors after fitting. |
| `under_prototyped_labels_` | Labels owning fewer than two prototypes. |
| `unevaluable_pairs_` | Lazy set-like view of selected multi-label pairs with no suitable rows. |
| `unevaluable_labels_` | Labels with no evaluable selected pair. |
| `n_features_in_` | Feature count recorded at fit time. |

The mappings preserve the historical mapping API. Pairwise mappings are sparse:
iteration and `dict(...)` expose only materialized non-default entries. Direct
lookup remains useful for diagnostics: an evaluable pair with no observed
overlap resolves to `1.0`, while an unevaluable pair resolves to its
zero-cardinality/`NaN` representation.

```python
worst_pairs = sorted(
    oi.pairwise_index.items(),
    key=lambda item: item[1],
)[:10]

for (source, competitor), score in worst_pairs:
    print(source, competitor, score)
```

For multi-label data, filter non-finite scores before sorting because
unevaluable pairs are represented by `NaN`.

`unevaluable_pairs_` is a lazy set-like diagnostic view on multi-label fits:
iterate it or use membership testing to enumerate/check every selected
directional pair whose `pairwise_cardinality` is zero. It retains only class
and competitor metadata during fitting, so all-mode diagnostics do not store a
quadratic pair list. `set(oi.unevaluable_pairs_)` remains a convenient way to
materialize the result for small datasets. Before fitting, after reset, and for
single-label fits the attribute remains the historical empty tuple.

## Common warnings

### Fewer than two prototypes

The top-two comparison is degenerate for labels in
`under_prototyped_labels_`. Increase `kmeans_k`, reduce overly broad BallCover
radii, or provide more samples if finer resolution is justified. Do not tune
the warning away blindly; changing prototype count also changes the evaluated
resolution.

### Empty data, one class, one target cell, or one prototype

The estimator retains its default value of `1.0` because overlap cannot be
evaluated. Treat this as an undefined/degenerate input condition, not perfect
separation.

### Unevaluable multi-label labels

Inspect label co-occurrence. A source and competitor that always occur together
have no row satisfying “source present, competitor absent.” Consider whether
the labels are distinguishable for the intended analysis or change
`top_m`/competitor selection.

## Common errors

### `Sparse X is supported only ...`

Use KMeans or MiniBatchKMeans, or convert to a dense matrix only if it safely
fits in memory. BallCover and ARTMAP do not accept sparse feature matrices.

### ARTMAP requires `[0, 1]`

Fit an appropriate scaler and transform every batch with the same fitted
scaler. Offline backends do not enforce this range.

### `predict` says the estimator is not fit

Call `fit`, `partial_fit`, or `add_batch` before prediction. Remember that the
output is a global prototype ID rather than a class prediction.

### An offline `partial_fit` forgot earlier data

This is expected. KMeans, MiniBatchKMeans, and BallCover refit on only the
provided batch. Combine the desired training rows and call `fit`, or select an
ARTMAP backend when true incremental state is required.

### State reset and explicit continuation

Full `fit(X, y)` and `score(X, y)` calls always construct a fresh backend.
The lower-level `fit_offline(..., reset_state=False)` form is accepted only
when explicitly continuing an ARTMAP backend. KMeans, MiniBatchKMeans, and
BallCover reject it because global prototype IDs cannot be safely accumulated
across independent offline refits. `ContinuousOverlapIndex` is offline-first
and always resets its fitted state.

### Indicator labels are not the expected names

Indicator columns become integer labels `0..n_labels-1`. Maintain an external
column-to-name mapping when original names are needed, or pass collections of
named labels instead.

## Parameter validation

Count, neighborhood, projection, permutation, threshold, and chunk-size
parameters require genuine integers where documented; booleans, strings, and
fractional values are not silently coerced. Radii and temperatures must be
finite and positive. Class-specific dictionaries such as `kmeans_k`,
`ballcover_k`, and `ballcover_radius` must provide a valid entry for every
observed label.

## Performance checks

- Use MiniBatchKMeans as the first choice for large offline datasets.
- Use sparse feature matrices with a centroid backend when the source data is
  sparse; avoid unnecessary densification.
- Lower `offline_chunk_size` to cap scored rows per tile. Set
  `offline_memory_budget_mb` to bound the temporary prototype-score scratch
  tile as well; the default 256 MiB is scratch only and does not cap fitted
  data or model size. Neither setting changes the result.
- Use multi-label `top_m` to avoid evaluating every directional label pair.
- For continuous targets, inspect `null_mode_`; refit permutations can dominate
  runtime. The fixed-structure null is faster but approximate.
- Set random seeds before comparing runtimes or scores across configurations.

## Reporting checklist

Record the data split, feature preprocessing, backend, prototype settings,
random seeds, target format, global aggregation, and all warnings. Report the
global score together with the worst per-label or local prototype diagnostics.
For continuous targets, also record the resolved target cover, target distance,
null mode, and number of null permutations.
