# Data and target formats

## Feature matrix `X`

`X` must be a two-dimensional numeric matrix with shape
`(n_samples, n_features)`. Values must be finite, and prediction data must have
the same number of features used during fitting.

| Backend | Dense NumPy-like `X` | SciPy sparse `X` | Feature range |
| --- | --- | --- | --- |
| MiniBatchKMeans | Yes | Yes | Finite; scale consistently |
| KMeans | Yes | Yes | Finite; scale consistently |
| FusedKMeans | Yes | No | Finite; optional internal L2/shared Fisher |
| BallCover | Yes | No | Finite; scale consistently |
| Fuzzy ARTMAP | Yes | No | Must be in `[0, 1]` |
| Hypersphere ARTMAP | Yes | No | Must be in `[0, 1]` |

For KMeans and MiniBatchKMeans, sparse inputs are normalized to CSR internally and remain sparse through
fitting, slicing, prediction, multi-label expansion, and chunked scoring. The
centroid and bounded score blocks are dense.

## Single-label targets

A one-dimensional sequence assigns one label to each row:

```python
y = ["cat", "cat", "dog", "dog"]
oi.fit(X, y)
```

Labels may be strings, integers, or other hashable scalar values. `None`, NaN,
unhashable values, and ambiguous array-like labels are rejected. Every fit must
have the same number of target entries as feature rows.

## Multi-label target collections

A one-dimensional outer sequence may contain a collection of positive labels
for each row:

```python
y = [
    {"animal", "cat"},
    {"animal", "dog"},
    {"vehicle", "car"},
]
oi.fit(X, y)
```

Every sample must have at least one positive label. Duplicate labels within a
sample are removed. See {doc}`multilabel` for scoring semantics and competitor
selection.

## Binary indicator matrices

Dense NumPy arrays and SciPy sparse matrices with shape
`(n_samples, n_labels)` are accepted. They must contain only `0` and `1`.
Positive column indices become integer labels; original external class names
are not inferred.

```python
import numpy as np
from scipy import sparse

Y_dense = np.array([
    [1, 1, 0],
    [1, 0, 0],
    [0, 1, 1],
])
oi.fit(X, Y_dense)       # labels are 0, 1, and 2
oi.fit(X, sparse.csr_matrix(Y_dense))
```

```{caution}
A rectangular Python list such as `[[1, 0], [0, 1]]` retains
sequence-of-label-collections semantics and emits a warning when it looks
binary. Convert it to a NumPy or SciPy matrix to explicitly request indicator
semantics.
```

## Continuous targets

{class}`overlapindex.ContinuousOverlapIndex` accepts a one-dimensional numeric
target or a two-dimensional numeric matrix for multivariate regression. All
values must be finite. It does not accept categorical or multi-label targets.

```python
from overlapindex import ContinuousOverlapIndex

coi.fit(X, y_regression)                  # shape (n_samples,)
coi.fit(X, Y_multivariate_regression)     # shape (n_samples, n_targets)
```

The continuous estimator currently supports only the three offline backends.
See {doc}`continuous_targets` for target scaling, target cells, and null
calibration.

## Empty and degenerate inputs

An empty dataset or a dataset with only one observed discrete class cannot
provide a class-overlap comparison. The estimator emits `RuntimeWarning` and
leaves `index` at its default `1.0`; this value should not be interpreted as
evidence of separation. Continuous fitting behaves similarly when it forms
only one target cell or one prototype.
