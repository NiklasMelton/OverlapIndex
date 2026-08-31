# OverlapIndex

```{image} ../img/overlap_index_logo.png
:alt: OverlapIndex logo
:width: 360px
:align: center
```

**OverlapIndex** measures how well labels are separated in a feature space by
looking at overlap between label-owned prototypes. Use it to compare
embeddings, diagnose confusing classes, monitor a stream, or evaluate
continuous-target structure without training a classifier.

The package provides two scikit-learn-style estimators:

- {class}`overlapindex.OverlapIndex` for single-label and multi-label data.
- {class}`overlapindex.ContinuousOverlapIndex` for univariate or multivariate
  regression targets.

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from overlapindex import OverlapIndex

X, y = load_iris(return_X_y=True)
X = MinMaxScaler().fit_transform(X)

oi = OverlapIndex(
    kmeans_k=10,
    kmeans_kwargs={"random_state": 0},
).fit(X, y)

print(f"macro OI: {oi.index:.3f}")
print(f"support-weighted OI: {oi.weighted_index:.3f}")
print(dict(oi.singleton_index))
```

## What the score means

Higher is better. A score of `1.0` means no overlap was observed in the fitted
prototype representation, values between `0.0` and `1.0` represent partial
overlap or separation, and `0.0` represents complete overlap. The score is a
representation diagnostic, not classifier accuracy, a probability, or a
significance test.

The default `index` gives every evaluable label equal weight. The
`weighted_index` property follows the observed label support. Per-label and
directional pairwise results are retained so a global score can always be
traced back to the labels that produced it.

## Where to begin

- {doc}`getting_started` — install the package and compute a first score.
- {doc}`concepts` — understand prototypes, pairwise overlap, aggregation, and
  the discrete clouds, bars, and rings examples.
- {ref}`discrete-visual-examples` — jump directly to the discrete visual
  examples and their score-response curves.
- {doc}`data_and_targets` — check accepted feature and target formats.
- {doc}`backends/index` — select MiniBatchKMeans, KMeans, the opt-in fused
  shared-Fisher ranking path, BallCover, or an incremental ARTMAP backend.
- {doc}`multilabel` — use label collections or indicator matrices correctly.
- {doc}`continuous_targets` — evaluate regression representations.
- {doc}`diagnostics` — interpret fitted attributes, warnings, and common
  failure modes.
- {doc}`use_cases` — apply OI to representation comparison, streaming, and
  dataset analysis.
- {doc}`api` — browse the complete source-generated API.

```{note}
`MiniBatchKMeans` is the default and the recommended starting point for batch
data. Only the `Fuzzy` and `Hypersphere` ARTMAP backends preserve learned state
across `partial_fit` calls, and they require the optional `art` extra.
```

```{toctree}
:hidden:
:maxdepth: 2

getting_started
concepts
data_and_targets
backends/index
multilabel
continuous_targets
diagnostics
use_cases
api
```
