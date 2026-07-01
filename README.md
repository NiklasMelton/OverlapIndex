[![OverlapIndex logo](https://raw.githubusercontent.com/NiklasMelton/OverlapIndex/develop/img/overlap_index_logo.png)](https://github.com/NiklasMelton/OverlapIndex)
# 
OverlapIndex (OI)

This package provides an implementation of the **Overlap Index (OI)**, a cluster-validity measure designed to quantify the degree of overlap between data classes or clusters. The OI can be updated online with ARTMAP-based backends, or computed in batch with offline clustering backends, making it useful for streaming, continual learning, large-scale representation analysis, and embedding-space diagnostics.

The implementation supports multiple swappable clustering backends:

- **Fuzzy ARTMAP** and **Hypersphere ARTMAP** for incremental / online updates.
- **KMeans** and **MiniBatchKMeans** for offline centroid-based analysis.
- **BallCover** for offline greedy landmark-ball covers, useful when the goal is to preserve class-support geometry for downstream shape or topology analysis.

---

## Installation

To install OverlapIndex, simply use pip:

```bash
pip install overlapindex
```

That installs the default batch-oriented dependencies. To enable the incremental
ART backends as well, install the optional ART extra:

```bash
pip install "overlapindex[art]"
```

The core package and optional `art` extra support Python 3.9 through 3.14.

Or to install directly from the most recent source:

```bash
pip install git+https://github.com/NiklasMelton/OverlapIndex.git@develop
```

---

## Overview

The Overlap Index is bounded in the interval **[0, 1]** and has the following interpretation:

- **OI = 1.0**  
  Indicates perfect class separation (no overlap).

- **OI = 0.5**  
  Indicates complete overlap between classes.

- **OI < 0.5**  
  Indicates a degenerate or pathological case in the data distribution.

The index is computed incrementally by tracking shared cluster activations between pairs of classes and aggregating class-wise overlap into a global measure.

---

## Key Properties

- **Incremental and Offline Modes**  
  ARTMAP backends support streaming updates via `add_sample` and mini-batch updates via `add_batch`.
  Offline backends such as `KMeans`, `MiniBatchKMeans`, and `BallCover` support batch computation through `add_batch`.

- **Label-Aware**  
  Can be applied both to labeled raw data and to intermediate representations (e.g., neural network activations).

- **Geometry-Agnostic**  
  Works well on arbitrary geometric structures of data. No geometric constraints are 
  assumed.

---

## Typical Use Cases

The Overlap Index can be used in several settings:

- **Unsupervised clustering evaluation**  
  As an iCVI, OI provides insight into the quality of a clustering partition as it evolves over time.

- **Class separability analysis**  
  Measures the degree of overlap in labeled datasets without requiring a classifier.

- **Representation monitoring in deep learning**  
  Tracks how class separation changes across layers or training epochs.

- **Backbone evaluation for transfer learning**  
  Compares feature extractors, where higher OI values indicate better class 
  separation in the backbone embeddings.

---

## Implementation Notes

- ART-based clustering is performed using `artlib`’s `FuzzyARTMAP` or `HypersphereARTMAP`.
- `artlib` is an optional dependency and is only required when using the
  `"Fuzzy"` or `"Hypersphere"` backends.
- Offline centroid backends fit one clustering model per class and concatenate the resulting class-owned prototypes into global cluster ids.
- The `BallCover` backend fits one greedy ball cover per class and treats ball centers as class-owned prototypes.
- Normalize input features before fitting. Examples in this repository use `MinMaxScaler` for convenience.
- ART backends complement-code inputs internally and therefore require features in the `[0, 1]` interval.
- Offline backends (`KMeans`, `MiniBatchKMeans`, and `BallCover`) consume normalized features directly and do not apply complement coding.
- Overlap is estimated by monitoring shared best-matching units (BMUs) or top prototype activations between class pairs.
- The global OI is computed as the macro mean of per-class minimum pairwise overlap scores, so each observed class contributes equally to `index`.
- A support-weighted companion score is available through `weighted_index` for workflows that need the score to reflect observed class frequencies.
- Global aggregation can exclude one or more label ids through `exclude_classes` without removing those labels from fitting, singleton scores, or pairwise scores.

---

## Basic Usage

```python
from sklearn.preprocessing import MinMaxScaler
from overlapindex import OverlapIndex

# Normalize features before fitting.
X = MinMaxScaler().fit_transform(X)

# MiniBatchKMeans is the default backend and is recommended for most offline use cases.
oi = OverlapIndex(
    kmeans_k=10,
    kmeans_kwargs={"random_state": 0},
)

# sklearn-style API
oi.fit(X, y)
score = oi.index
```


The fitted value is available through `oi.index`. For users who prefer update methods that return the current score directly, `add_batch(X, y)` is also supported.

### Excluding Classes From Global Aggregation

`exclude_classes` lets you keep a label fully involved in overlap evaluation
while omitting it from the two global summary scores:

```python
oi = OverlapIndex(exclude_classes=0)
oi = OverlapIndex(exclude_classes=[0, "unlabeled"])
```

This is useful for segmentation workflows where only foreground objects are
labeled but background-only samples should still contribute to pairwise overlap
counts. A common pattern is to create one background class containing those
samples, then pass that class id to `exclude_classes`. The background class will
still appear in `singleton_index`, `pairwise_index`, and prototype ownership;
only `index` and `weighted_index` omit it from aggregation.

### Online ARTMAP Usage

```python
from overlapindex import OverlapIndex

# For ARTMAP backends, batches should already be scaled into [0, 1].

oi = OverlapIndex(
    model_type="Hypersphere",
    rho=0.9,
    match_tracking="MT+",
)

for X_batch, y_batch in stream:
    oi.partial_fit(X_batch, y_batch)
    score = oi.index
```

For single-sample streams, ARTMAP backends also support `add_sample(x, y)`, which updates the model and returns the current score directly. Labeled mini-batches can also be passed to `add_batch(X, y)`.

### API Styles

`OverlapIndex` supports both sklearn-style methods and direct score-returning update methods:

| Method | Returns | Typical use                                                   |
| --- | --- |---------------------------------------------------------------|
| `fit(X, y)` | `self` | Full offline fitting on a labeled dataset.                    |
| `partial_fit(X, y)` | `self` | Incremental batch updates for ARTMAP backends; offline backends refit on the provided batch. |
| `score()` / `score(X, y)` | `float` | Read the current index, or refit on labeled data and return the new score. |
| `predict(X)` | `np.ndarray` | Return the highest-scoring global prototype id for each sample. |
| `fit_predict(X, y)` | `np.ndarray` | Fit and return per-sample prototype ids. |
| `add_batch(X, y)` | `float` | Batch update when the current OI score is needed immediately. |
| `add_sample(x, y)` | `float` | Single-sample online update for ARTMAP backends.              |

After `fit` or `partial_fit`, read the current score from `oi.index` or call `score()`.

For `model_type="KMeans"`, `model_type="MiniBatchKMeans"`, and
`model_type="BallCover"`, `partial_fit(X, y)` is a convenience wrapper around
recomputing the index on the provided labeled batch. Only the ARTMAP backends
perform true incremental updates across calls.

If a batch is empty or contains only one unique class, `OverlapIndex` emits a
`RuntimeWarning` and leaves the score at its default value of `1.0`.

### Clustering Backends

`OverlapIndex` uses `model_type="MiniBatchKMeans"` by default and supports several backend families through the `model_type` parameter:

| `model_type` | Update mode | Description |
| --- | --- | --- |
| `"Fuzzy"` | Online / batch | Incremental Fuzzy ARTMAP backend. Requires the optional `art` extra. |
| `"Hypersphere"` | Online / batch | Incremental Hypersphere ARTMAP backend. Requires the optional `art` extra. |
| `"KMeans"` | Offline batch only | Fits one scikit-learn `KMeans` model per class. |
| `"MiniBatchKMeans"` | Offline batch only | Default backend. Fits one scikit-learn `MiniBatchKMeans` model per class; recommended for larger datasets. |
| `"BallCover"` | Offline batch only | Fits one greedy landmark-ball cover per class. Useful when preserving class-support geometry is important. |

Offline backends should be used with `fit` or `add_batch`. They do not support `add_sample` because their prototypes are fit from a complete labeled batch.

#### KMeans backend

```python
from overlapindex import OverlapIndex

OI = OverlapIndex(
    model_type="KMeans",
    kmeans_k=10,
    kmeans_kwargs={"random_state": 0},
)

OI.fit(X, y)
score = OI.index
```

#### MiniBatchKMeans backend

```python
from overlapindex import OverlapIndex

OI = OverlapIndex(
    model_type="MiniBatchKMeans",
    kmeans_k=10,
    kmeans_kwargs={
        "random_state": 0,
        "batch_size": 8192,
        "n_init": 1,
    },
)

OI.fit(X, y)
score = OI.index
```

#### BallCover backend

```python
from overlapindex import OverlapIndex

OI = OverlapIndex(
    model_type="BallCover",
    ballcover_k="auto",
    ballcover_radius=0.25,
    ballcover_kwargs={
        "metric": "auto",
        "cover_fraction": 1.0,
    },
)

OI.fit(X, y)
score = OI.index
```

The BallCover backend supports one automatic cover parameter at a time:

- `ballcover_k="auto"` with a fixed `ballcover_radius` greedily adds balls until the requested cover fraction is reached.
- `ballcover_k=<int>` with `ballcover_radius="auto"` selects a fixed number of landmarks and infers the radius needed to cover the requested fraction of samples.

`metric="auto"` uses Euclidean distance in lower-dimensional spaces and cosine geometry for high-dimensional inputs such as embedding vectors. Users can override this with `metric="euclidean"` or `metric="cosine"`.

### Iris Dataset Example
```python

from sklearn.datasets import load_iris
import numpy as np
from overlapindex import OverlapIndex

# Load dataset
iris = load_iris()

# Feature matrix (shape: [150, 4])
X = iris.data.astype(np.float64)

# Target vector (shape: [150,])
y = iris.target.astype(np.int64)

# Normalize the data (required)
x_max = X.max(axis=0)
x_min = X.min(axis=0)
X = (X - x_min) / (x_max - x_min)

# Instantiate the OI object
OI = OverlapIndex()

# Calculate the Overlap Index
OI.fit(X, y)
print(OI.index)

# Output:
# 0.9266666666666666
```

Additional runnable examples are available in the `examples/` directory.

---

## Continuous Targets

`ContinuousOverlapIndex` is a regression-capable companion estimator for
continuous targets. It preserves the OI interpretation by measuring whether
feature-space prototype overlap occurs between incompatible empirical target
distributions:

- **COI = 1.0** indicates no observed harmful continuous-target overlap.
- **COI = 0.5** indicates overlap no better than a permutation/null target
  assignment.
- **COI < 0.5** indicates pathological overlap relative to the permutation
  null.

Version 1 is offline-first and supports `model_type="MiniBatchKMeans"`,
`model_type="KMeans"`, and `model_type="BallCover"`. ARTMAP online support is
intentionally deferred for continuous targets.

```python
from sklearn.preprocessing import MinMaxScaler
from overlapindex import ContinuousOverlapIndex

X = MinMaxScaler().fit_transform(X)

coi = ContinuousOverlapIndex(
    model_type="MiniBatchKMeans",
    kmeans_k=8,
    kmeans_kwargs={"random_state": 0},
    n_target_cells="auto",
    n_null_permutations=20,
    random_state=0,
)

coi.fit(X, y_regression)
score = coi.index
```

For univariate regression targets, `target_cover="auto"` uses quantile target
cells and `target_distance="auto"` uses 1D Wasserstein distance. For
multivariate regression targets, `target_cover="auto"` uses KMeans target cells
and `target_distance="auto"` uses sliced Wasserstein distance.

COI stores empirical target measures per feature prototype instead of reducing
targets to means or variances. The permutation null refits the target cover and
feature prototypes for each target shuffle so that random target assignments
calibrate near 0.5. As with discrete OI, use enough prototypes per target cell
for overlap structure to be observable; one prototype per cell is usually too
coarse for separation diagnostics.

Key diagnostics after fitting include:

- **`actual_loss_`**, **`null_loss_`**, and **`loss_ratio_`**
- **`raw_index_`** before optional clipping
- **`macro_index_`** and **`weighted_index`**
- **`prototype_index_`**, **`prototype_loss_`**, and
  **`prototype_target_values_`**

---

## Release Verification

For release testing, start from a fresh Poetry environment so the package under
test matches `pyproject.toml` and `poetry.lock`:

```bash
poetry env remove --all
poetry sync --with dev
poetry run python -c "from overlapindex import OverlapIndex; OverlapIndex(model_type='MiniBatchKMeans')"
poetry run python -m pytest -q tests/test_overlap_index_regression.py

poetry sync --with dev --extras art
poetry run python -c "from overlapindex import OverlapIndex; OverlapIndex(model_type='Hypersphere')"
poetry run python -m pytest -q tests/test_overlap_index_regression.py

poetry check
python -m build
twine check dist/*
```

The first install verifies that offline backends work without the optional
`artlib` dependency. The second install verifies the `art` extra and ARTMAP
backends.

---

## Parameters

- `rho` *(float)*  
  Vigilance parameter controlling cluster granularity for ARTMAP backends.

- `r_hat` *(float, Hypersphere ARTMAP only)*  
  Maximum cluster radius for the Hypersphere backend.

- `model_type` *("Fuzzy" | "Hypersphere" | "KMeans" | "MiniBatchKMeans" | "BallCover")*  
  Clustering backend used to create class-owned prototypes. Defaults to `"MiniBatchKMeans"`.

- `match_tracking` *(str)*  
  Match-tracking strategy used during ARTMAP learning.

- `kmeans_k` *(int or dict)*  
  Number of clusters per class for `KMeans` and `MiniBatchKMeans` backends.

- `kmeans_kwargs` *(dict, optional)*  
  Keyword arguments forwarded to the selected scikit-learn KMeans backend.

- `ballcover_k` *(int, dict, or "auto")*  
  Number of balls per class, class-specific ball counts, or `"auto"` for greedy fixed-radius covering.

- `ballcover_radius` *(float, dict, or "auto")*  
  Ball radius, class-specific radii, or `"auto"` when using a fixed number of balls.

- `ballcover_kwargs` *(dict, optional)*  
  Additional BallCover options such as `metric`, `cover_fraction`, `chunk_size`, `max_balls`, and `random_state`.

- `exclude_classes` *(None, scalar label, or iterable of labels)*  
  Label ids to omit from the global `index` and `weighted_index`
  aggregation while leaving all fitting and per-class overlap outputs intact.

---

The default parameters are intended for offline batch use with `MiniBatchKMeans`. For online or continual-learning workflows, explicitly choose `model_type="Fuzzy"` or `model_type="Hypersphere"`. For very large ART-based runs, smaller `rho` values (0.5-0.7) may improve run-time performance.

---

## Output

- **`index`**  
  Global macro Overlap Index across all observed classes that are not listed in
  `exclude_classes`. This is the default class-balanced score and is usually
  preferred for imbalance-sensitive separation analysis.

- **`weighted_index`**  
  Support-weighted Overlap Index across observed classes that are not listed in
  `exclude_classes`. This weights each included class's `singleton_index` value
  by its positive sample count, which can be useful when reporting should
  reflect observed class frequencies.

- **`singleton_index[y]`**  
  Minimum pairwise overlap score for class `y`.

- **`pairwise_index[(y, b)]`**  
  Pairwise overlap score between classes `y` and `b`.

---

## Intended Audience

This package is intended for researchers and practitioners working on:

- incremental and continual learning,
- clustering validation,
- representation learning,
- transfer learning
