# OverlapIndex (OI)

This package provides an implementation of the **Overlap Index (OI)**, an *incremental cluster validity index (iCVI)* designed to quantify the degree of overlap between data classes or clusters. The OI is updated online, sample by sample or in batches, and is particularly suited for streaming, continual learning, and representation analysis.

The implementation is built on **ARTMAP-based clustering** (Fuzzy ART or Hypersphere 
ART), leveraging the dynamic clustering properties of Adaptive Resonance Theory to 
track class overlap as new data (and classes) arrive.

---

## Installation

To install OverlapIndex, simply use pip:

```bash
pip install overlapindex
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

- **Incremental / Online**  
  Supports streaming updates via `add_sample` and mini-batch updates via `add_batch`.
  New classes can be introduced at any time, enabling analysis of incremental 
  learning scenarios.

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
- Inputs are **complement coded**, following standard ART practice.
- Overlap is estimated by monitoring shared best-matching units (BMUs) between class pairs.
- The global OI is computed as the mean of per-class minimum pairwise overlap scores.

---

## Basic Usage

    from overlapindex import OverlapIndex

    oi = OverlapIndex(
        rho=0.9,
        ART="Hypersphere",
        match_tracking="MT+"
    )

    # Incremental update
    for x, y in stream:
        score = oi.add_sample(x, y)

    # Or batch update
    score = oi.add_batch(X, Y)

The returned value is the current Overlap Index after the update.

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
oi = OI.add_batch(X, y)
print(oi)

# Output:
# 0.9266666666666666
```



---

## Parameters

- `rho` *(float)*  
  Vigilance parameter controlling cluster granularity.

- `r_hat` *(float, Hypersphere ART only)*  
  Maximum cluster radius.

- `ART` *("Fuzzy" | "Hypersphere")*  
  Choice of ART module.

- `match_tracking` *(str)*  
  Match-tracking strategy used during ARTMAP learning.

The default parameters are likely to satisfy most use cases. For very large datasets,
it may be necessary to use smaller `rho` values (0.5-0.7) to improve run-time 
performance. 

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
