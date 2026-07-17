# Getting started

This guide takes the shortest supported path from installation to an
interpretable result.

## Installation

Install the default batch-oriented package from PyPI:

```bash
python -m pip install overlapindex
```

To use the incremental Fuzzy or Hypersphere ARTMAP backends, install the
optional dependencies:

```bash
python -m pip install "overlapindex[art]"
```

OverlapIndex supports Python 3.9 through 3.14. The ART extra is not imported by
the default offline backends.

## A first classification score

The normal workflow is: prepare features, construct the estimator, fit, then
inspect the global and detailed results.

```python
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
from overlapindex import OverlapIndex

X, y = load_wine(return_X_y=True)
X = MinMaxScaler().fit_transform(X)

oi = OverlapIndex(
    model_type="MiniBatchKMeans",
    kmeans_k=8,
    kmeans_kwargs={"random_state": 0},
)
oi.fit(X, y)

print("macro:", oi.index)
print("weighted:", oi.weighted_index)
print("per label:", dict(oi.singleton_index))
```

`fit` returns the estimator, following scikit-learn conventions. The same
calculation can be written as `score = oi.add_batch(X, y)` when a direct float
return is more convenient.

## Preprocessing

The package validates input but does not apply a general-purpose feature
scaler. Normalize features consistently before comparing representations;
distance-based backends are sensitive to relative feature scale.

- Offline backends consume features directly. They do not complement-code
  them or require the `[0, 1]` interval.
- ARTMAP backends require dense finite features in `[0, 1]` and apply
  complement coding internally.
- KMeans and MiniBatchKMeans accept dense arrays and SciPy sparse feature
  matrices. BallCover and ARTMAP require dense arrays.

Fit preprocessing only on the appropriate training data when leakage matters.
For example, use a scikit-learn `Pipeline` or fit the scaler on the training
split before transforming an evaluation split.

## Choose the right update method

| Method | Return value | Behavior |
| --- | --- | --- |
| `fit(X, y)` | `self` | Fresh fit on a complete labeled dataset. |
| `score()` | `float` | Return the stored score without refitting. |
| `score(X, y)` | `float` | Fresh fit on `X, y`, then return the score. |
| `add_batch(X, y)` | `float` | Update/refit and immediately return the score. |
| `partial_fit(X, y)` | `self` | Incremental only for ARTMAP; offline backends refit on this batch. |
| `add_sample(x, y)` | `float` | Single-sample update for ARTMAP only. |
| `predict(X)` | integer array | Return global prototype IDs, not class labels. |
| `fit_predict(X, y)` | integer array | Fresh fit followed by prototype-ID prediction. |

```{warning}
`MiniBatchKMeans` is an offline backend in OverlapIndex. Despite its name,
calling `OverlapIndex.partial_fit` does not accumulate its scikit-learn model
across batches; it refits on the supplied batch.
```

## Make comparisons reproducible

When comparing embeddings, layers, or preprocessing choices:

1. Use the same samples, labels, preprocessing, backend, and prototype count.
2. Set `random_state` in `kmeans_kwargs` or `ballcover_kwargs`.
3. Report `index`, `weighted_index` when imbalance matters, and the worst
   per-label scores.
4. Inspect `under_prototyped_labels_` and any multi-label unevaluable
   diagnostics before interpreting the summary.

Continue with {doc}`concepts` for the scoring mechanics or
{doc}`backends/index` for backend selection.
