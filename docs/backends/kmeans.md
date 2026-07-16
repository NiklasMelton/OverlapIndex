# KMeans

The `KMeans` backend fits one full scikit-learn KMeans model per label. Its
prototype ownership and squared-Euclidean scoring are the same as the
MiniBatchKMeans backend, but center fitting uses the complete label-specific
dataset on each iteration.

## When to use it

Choose KMeans for smaller offline datasets when full-batch optimization is
affordable and you prefer it over minibatch approximation. It supports dense
and SciPy sparse feature matrices as well as all supported multi-label target
formats.

Use MiniBatchKMeans for larger data or faster iterations. Like every offline
backend, KMeans does not retain state across `partial_fit` calls.

## Invocation

```python
from overlapindex import OverlapIndex

oi = OverlapIndex(
    model_type="KMeans",
    kmeans_k=10,
    kmeans_kwargs={"random_state": 0, "n_init": "auto"},
)
oi.fit(X, y)
print(oi.index)
```

`kmeans_kwargs` is forwarded to scikit-learn's KMeans constructor. When
`n_init` is omitted, the adapter uses `n_init="auto"`. `kmeans_k` may be an
integer or a dictionary keyed by every observed label.

## Tuning guidance

- Hold `kmeans_k` constant when comparing representations.
- Increase it when label support is too complex for the current centers, while
  watching runtime and `under_prototyped_labels_`.
- Fix `random_state` for reproducible comparisons.
- Prefer normalized features because Euclidean distance is sensitive to feature
  scale.

The repository's `examples/compare_backends_on_breast_cancer.py` demonstrates
KMeans alongside the other backend families.
