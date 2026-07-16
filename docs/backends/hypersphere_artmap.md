# Hypersphere ARTMAP

The `Hypersphere` backend uses `artlib`'s Hypersphere ARTMAP implementation.
Its class-owned prototypes use hypersphere geometry, and it supports true
incremental updates through `partial_fit` and `add_sample`.

## When to use it

Choose Hypersphere ARTMAP for streaming dense data when radius-based prototype
geometry is a natural fit. It is often easier to reason about geometrically
than complement-coded hyperrectangular categories, while Fuzzy ARTMAP may be a
better fit when Fuzzy ART behavior is specifically desired.

Install the optional ARTMAP dependencies with:

```bash
pip install "overlapindex[art]"
```

## Invocation

Scale every feature to `[0, 1]`; OverlapIndex performs complement coding
internally.

```python
from overlapindex import OverlapIndex

oi = OverlapIndex(
    model_type="Hypersphere",
    rho=0.9,
    r_hat=0.5,
    match_tracking="MT+",
)

for X_batch, y_batch in stream:
    oi.partial_fit(X_batch, y_batch)
    print(oi.index)
```

`fit(X, y)` is also available when a complete batch should initialize a fresh
model.

## Tuning guidance

- `rho` controls vigilance; increasing it generally creates finer categories.
- `r_hat` constrains hypersphere radius. Use a finite domain-appropriate value
  when the default unbounded constraint is not suitable.
- `match_tracking` selects the supervised match-tracking behavior.
- Monitor both score and prototype count when selecting `rho` and `r_hat`.

This backend requires dense, single-label online data. It does not support
sparse feature matrices or multi-label `add_batch` updates. See
`examples/iris_online_artmap.py` and `examples/partial_fit_batches.py` for
complete streaming examples.
