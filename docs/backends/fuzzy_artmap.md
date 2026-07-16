# Fuzzy ARTMAP

The `Fuzzy` backend uses `artlib`'s Fuzzy ARTMAP implementation to build and
update class-owned prototypes. It is one of the two backends that preserves
learned state across `partial_fit` calls.

## When to use it

Choose Fuzzy ARTMAP for online or continual-learning workflows with dense,
bounded features when adaptive Fuzzy ART prototype geometry is appropriate.
Use an offline backend when all data is available at once, sparse matrices are
required, or multi-label targets are required.

ARTMAP support is optional. Install it with:

```bash
pip install "overlapindex[art]"
```

## Invocation

Input features must already be scaled to `[0, 1]`. OverlapIndex applies
complement coding internally before passing data to ARTMAP.

```python
from overlapindex import OverlapIndex

oi = OverlapIndex(
    model_type="Fuzzy",
    rho=0.9,
    match_tracking="MT+",
)

for X_batch, y_batch in stream:
    oi.partial_fit(X_batch, y_batch)
    print(oi.index)
```

For a true single-sample stream, `add_sample(x, y)` updates the model and
returns the current score directly.

## Tuning guidance

- `rho` is the vigilance parameter. Higher vigilance generally creates finer
  prototype partitions; lower vigilance permits broader categories.
- `match_tracking` controls the supervised match-tracking strategy used during
  updates.
- Track prototype growth as well as OI when tuning vigilance. Excessively fine
  partitions can increase runtime and change the resolution at which overlap
  is measured.

ARTMAP online updates require single-label targets. Multi-label `add_batch`
inputs are rejected, and sparse feature matrices are unsupported.
