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

### Optional balanced-median refinement

Set `prototype_refinement=True` to opt in to a deterministic, one-pass
balanced-median refinement after the KMeans fit:

```python
oi = OverlapIndex(
    model_type="KMeans",
    kmeans_k=10,
    kmeans_kwargs={"random_state": 0, "n_init": "auto"},
    prototype_refinement=True,
)
oi.fit(X, y)  # y must contain one scalar label per row
print(oi.prototype_refinement_["applied_count"])
```

Eligibility is frozen from the fitted data: a parent prototype must have
support of at least two and zero outgoing own-runner-up count. Each eligible
parent is projected along a deterministic farthest-pair axis, split into
balanced halves, and represented by the actual observation nearest each
half's coordinate-wise median. No child scikit-learn fit, gate, or rescue
step is run. The pass is single-label only; multi-label targets are rejected.
`score_fixed` scores with these already-refined centers and does not refit or
refine them again.

The default `prototype_refinement=False` preserves the ordinary KMeans
centers. Refinement can improve resolution for broad, isolated supports, but
it adds a tiled fit-isolation pass and can increase the prototype count and
subsequent scoring cost. Sparse input is accepted; only local support blocks
used by a candidate split may be densified. Use the opt-in when that runtime
and prototype-growth trade-off is acceptable. The fitted
`prototype_refinement_` diagnostics expose the resolved mode as
`"none"` or `"balanced_median"`.

## Tuning guidance

- Hold `kmeans_k` constant when comparing representations.
- Increase it when label support is too complex for the current centers, while
  watching runtime and `under_prototyped_labels_`.
- Fix `random_state` for reproducible comparisons.
- Prefer normalized features because Euclidean distance is sensitive to feature
  scale.

The repository's `examples/compare_backends_on_breast_cancer.py` demonstrates
KMeans alongside the other backend families.
