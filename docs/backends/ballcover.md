# BallCover

BallCover is the package's offline greedy geometric backend. It fits a separate
set of landmark balls for each label and treats every ball as a class-owned
prototype. Unlike the centroid backends, it explicitly represents a support
radius around each landmark.

## When to use it

Choose BallCover when preserving the shape or extent of class support matters,
such as non-convex distributions or embedding spaces where a centroid-only
summary is too coarse. It supports multi-label targets but requires dense
feature arrays.

`metric="auto"` selects Euclidean geometry below the configured high-dimensional
threshold and cosine geometry for higher-dimensional inputs. Cosine mode
normalizes rows internally and uses chord distance on the unit sphere.

## Invocation

BallCover permits one automatic structural parameter at a time.

Use a fixed radius and automatically add balls until the requested coverage is
reached:

```python
oi = OverlapIndex(
    model_type="BallCover",
    ballcover_k="auto",
    ballcover_radius=0.25,
    ballcover_kwargs={
        "metric": "auto",
        "cover_fraction": 1.0,
        "random_state": 0,
    },
)
```

Or select a fixed number of landmarks and infer the radius:

```python
oi = OverlapIndex(
    model_type="BallCover",
    ballcover_k=40,
    ballcover_radius="auto",
    ballcover_kwargs={"metric": "euclidean", "random_state": 0},
)
oi.fit(X, y)
```

## Tuning guidance

- Smaller fixed radii usually produce more balls and finer support detail.
- Larger `ballcover_k` provides more landmarks when radius is automatic.
- `cover_fraction` below `1.0` allows the cover to ignore a tail of samples and
  can reduce sensitivity to outliers.
- Set `max_balls` to cap work in automatic-count mode.
- Override `metric="auto"` when domain knowledge favors Euclidean or cosine
  geometry.
- Use `chunk_size` to bound distance-computation batches.

BallCover is offline-only: `partial_fit` refits on the provided batch and
`add_sample` is unsupported. See `examples/wine_ballcover.py` and the
{doc}`../api` reference for the backend's complete parameter documentation.
