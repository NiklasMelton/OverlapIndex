# Multi-label data

Multi-label scoring is available with MiniBatchKMeans, KMeans, and BallCover.
It accepts per-sample label collections and dense or sparse binary indicator
matrices; see {doc}`data_and_targets` for the exact input forms.

## Fitting and support counts

Before fitting the backend, each sample is duplicated once for every positive
label. This allows every label to own prototypes representing all of its
positive samples. The public `cluster_cardinality[label]` remains the number of
original positive samples, not the expanded row count.

Suppose a sample has labels `{a, b}`. It contributes one training row to the
prototype model for `a` and one for `b`, but it is not valid evidence about
whether `a` overlaps `b`: both labels are true for that row.

## Directional pair denominators

A directional pair `(a, b)` is evaluated only on rows where `a` is present and
`b` is absent. Its denominator is stored as
`pairwise_cardinality[(a, b)]`. Consequently, the denominator can differ
between competitors and between the two directions of a pair.

```python
from overlapindex import OverlapIndex

label_sets = [
    {"cat", "pet"},
    {"dog", "pet"},
    {"cat"},
    {"wildlife"},
]

oi = OverlapIndex(
    model_type="MiniBatchKMeans",
    kmeans_k=2,
    kmeans_kwargs={"random_state": 0},
).fit(X, label_sets)

print(oi.pairwise_index[("cat", "dog")])
print(oi.pairwise_cardinality[("cat", "dog")])
```

## Selecting competitors

`multilabel_pair_mode="all"` is the default and compares every source label
with every other observed label. This gives the most complete diagnostics, but
the number of directional pairs grows quadratically.

For many labels, `multilabel_pair_mode="top_m"` selects up to `top_m` nearest
competitors for each source label using minimum prototype-to-prototype
distance:

```python
oi = OverlapIndex(
    multilabel_pair_mode="top_m",
    top_m=5,
    kmeans_kwargs={"random_state": 0},
).fit(X, label_sets)

for source, selected in oi.competitors_.items():
    print(source, selected.tolist())
```

The selected competitors are fitted-data dependent. Hold the backend,
preprocessing, prototype settings, and random seed constant when comparing
runs.

## Unevaluable comparisons

A pair is unevaluable when there is no row where its source label is present
and its competitor is absent. The fitted estimator records:

- `unevaluable_pairs_`: directional pairs with a zero denominator and a `NaN`
  pairwise score.
- `unevaluable_labels_`: source labels with no evaluable selected competitor;
  their `singleton_index` is `NaN`.

Unevaluable labels are omitted from `index` and `weighted_index`. Fitting
raises `ValueError` if no non-excluded source label has any evaluable selected
pair. Do not replace these `NaN` values with a perfect score; they mean that
the requested comparison was not identified by the data.

## Current limitations

- ARTMAP `add_batch` rejects multi-label batches.
- `add_sample` is a single-label ARTMAP update path.
- `partial_fit` on an offline multi-label estimator refits on the provided
  batch; it does not accumulate earlier batches.

For high-cardinality problems, use sparse indicator targets, centroid backends
with sparse `X` where appropriate, `top_m`, and `offline_chunk_size` to control
the scored row block size.
