# OverlapIndex (OI)

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
- Inputs are normalized internally before clustering; ART backends use complement coding following standard ART practice.
- Overlap is estimated by monitoring shared best-matching units (BMUs) or top prototype activations between class pairs.
- The global OI is computed as the mean of per-class minimum pairwise overlap scores.

---

## Basic Usage

```python
from overlapindex import OverlapIndex

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

### Online ARTMAP Usage

```python
from overlapindex import OverlapIndex

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
| `partial_fit(X, y)` | `self` | Incremental or repeated batch updates. (ARTMAP Only)          |
| `add_batch(X, y)` | `float` | Batch update when the current OI score is needed immediately. |
| `add_sample(x, y)` | `float` | Single-sample online update for ARTMAP backends.              |

After `fit` or `partial_fit`, read the current score from `oi.index`.

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

---

The default parameters are intended for offline batch use with `MiniBatchKMeans`. For online or continual-learning workflows, explicitly choose `model_type="Fuzzy"` or `model_type="Hypersphere"`. For very large ART-based runs, smaller `rho` values (0.5-0.7) may improve run-time performance.

---

## Output

- **`index`**  
  Global Overlap Index across all observed classes.

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
