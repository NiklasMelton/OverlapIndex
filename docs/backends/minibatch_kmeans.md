# MiniBatchKMeans

`MiniBatchKMeans` is the default backend and the recommended starting point for
offline analysis. OverlapIndex fits one scikit-learn MiniBatchKMeans model per
label, concatenates their centers into global prototype IDs, and scores samples
using squared Euclidean distance to those centers.

## When to use it

Choose this backend for most batch workflows, especially when the dataset is
too large for repeated full KMeans fits. It supports dense arrays, SciPy sparse
feature matrices, and all supported multi-label target formats.

Despite the backend's name, OverlapIndex treats it as an offline backend.
Calling `partial_fit` on `OverlapIndex` refits and recomputes the score from the
provided batch; it does not call scikit-learn's incremental MiniBatchKMeans API
across batches.

## Invocation

Because it is the default, `model_type` may be omitted:

```python
from overlapindex import OverlapIndex

oi = OverlapIndex(
    kmeans_k=10,
    kmeans_kwargs={
        "random_state": 0,
        "batch_size": 256,
        "max_no_improvement": 5,
        "compute_labels": False,
        "n_init": 1,
    },
)
oi.fit(X, y)
print(oi.index)
```

The adapter defaults to `batch_size=256`, `max_no_improvement=5`,
`compute_labels=False`, `n_init=1`, and `init="random"`. Entries in
`kmeans_kwargs` override those defaults and are forwarded to scikit-learn.
`compute_labels=False` avoids a final full-data label-assignment pass because
OverlapIndex uses the fitted centers rather than the estimator's `labels_` or
exact inertia. `kmeans_k` may be one positive integer for every label or a
dictionary of label-specific counts.

### Optional balanced-median refinement

Pass `prototype_refinement=True` when a single-label fit can benefit from a
deterministic balanced-median prototype split:

```python
oi = OverlapIndex(
    kmeans_k=10,
    kmeans_kwargs={"random_state": 0, "batch_size": 256, "n_init": 1},
    prototype_refinement=True,
)
oi.fit(X, y)  # y must contain one scalar label per row
print(oi.prototype_refinement_["prototype_count_after"])
```

The one-pass eligibility check runs on the original fitted centers. A parent
needs support of at least two and zero outgoing own-runner-up count. Its rows
are ordered along a deterministic farthest-pair projection, divided into
balanced halves, and each child is the actual observation nearest that half's
coordinate-wise median. No child scikit-learn fit, gate, or rescue step is
performed, and appended children are not reconsidered in the same pass.
Multi-label targets are rejected for this mode. `score_fixed` uses the
already-refined centers without another fit or refinement pass.

`prototype_refinement=False` (the default) retains the standard
MiniBatchKMeans centers. Balanced refinement adds one tiled fit-isolation scan
and may increase both the prototype count and scoring time. Sparse matrices
remain supported, although a candidate's local support block may be
densified. Treat the option as an explicit runtime-versus-prototype-resolution
trade-off and enable it only when the extra resolution is useful. The fitted
`prototype_refinement_` diagnostics expose the resolved mode as `"none"` or
`"balanced_median"`.

## Tuning guidance

- Increase `kmeans_k` when a label has multimodal or curved support that a few
  centers cannot represent. More centers increase runtime and may also resolve
  finer overlap.
- Never request fewer than two centers per label when a top-two overlap result
  needs to be well resolved. If a label has fewer samples than requested
  centers, the backend caps its center count at its sample count.
- Use a fixed `random_state` for comparisons.
- Adjust `batch_size` for memory and throughput; it does not control
  `OverlapIndex.partial_fit` semantics.
- Adjust `max_no_improvement` or `max_iter` when trading convergence work for
  runtime; validate that the resulting overlap scores remain stable.
- Use `offline_chunk_size` to bound the number of rows scored at once when the
  fitted centroid set is large.

See the runnable `examples/iris_default_minibatch_kmeans.py` example in the
repository.
