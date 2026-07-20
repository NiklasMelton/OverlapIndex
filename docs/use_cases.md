# Use cases and interpretation

OverlapIndex measures whether samples from different labels activate the same
regions of a prototype representation. It is useful when the question is not
simply whether a classifier is accurate, but whether the feature space itself
separates the concepts represented by the labels.

## Reading the score

For the discrete-label estimator, larger values indicate better separation:

| Score | Interpretation |
| --- | --- |
| `1.0` | No observed class overlap in the fitted prototype representation. |
| Between `0.0` and `1.0` | Partial overlap or separation at the fitted prototype resolution. |
| `0.0` | Complete class overlap at the fitted prototype resolution. |

The score describes the supplied representation and labels; it is not a
classifier accuracy, probability, or statistical significance test. Compare
scores only when preprocessing, backend choice, and important backend
hyperparameters are held constant.

`index` is the macro average: every evaluable, non-excluded label contributes
equally. `weighted_index` weights each label by its positive sample support and
therefore follows the observed class distribution. On imbalanced data, report
both when class-balanced and population-weighted conclusions are relevant.

Use the fitted diagnostics to understand a summary score:

- `singleton_index[label]` identifies the least-separated competitor for each
  label.
- `pairwise_index[(label, competitor)]` localizes overlap to a directional
  label pair.
- `cluster_cardinality` records positive support by label.
- `under_prototyped_labels_` identifies labels whose top-two comparison is
  based on fewer than two owned prototypes.
- For multi-label targets, `unevaluable_pairs_` and `unevaluable_labels_`
  identify comparisons that lacked suitable positive/negative rows.

## Common use cases

### Representation and embedding comparison

Compute OI on embeddings from multiple feature extractors, model layers, or
training checkpoints. A larger value indicates that the representation makes
the supplied labels more separable under the same OI configuration.

```python
from overlapindex import OverlapIndex

def separation_score(embeddings, labels):
    oi = OverlapIndex(
        model_type="MiniBatchKMeans",
        kmeans_k=10,
        kmeans_kwargs={"random_state": 0},
    )
    return oi.fit(embeddings, labels).index
```

Normalize every representation consistently and keep `kmeans_k` fixed when
comparing results.

### Dataset diagnostics

Use per-label and pairwise results to find classes that are difficult to
distinguish before training a classifier. Low pairwise scores can reveal label
ambiguity, insufficient features, or closely related categories that may need
additional data or a revised taxonomy.

### Monitoring a stream

Use an ARTMAP backend when the representation arrives over time and the index
must preserve state across updates. A falling score can flag drift in class
separation. Streaming applications should also track sample counts and backend
prototype growth; a score change can reflect either changing data or changing
prototype resolution.

```python
oi = OverlapIndex(model_type="Hypersphere", rho=0.9)

for X_batch, y_batch in stream:
    oi.partial_fit(X_batch, y_batch)
    monitor(oi.index)
```

### Multi-label analysis

Offline backends accept label collections and dense or sparse binary indicator
matrices. Directional pair `(a, b)` is evaluated using samples where `a` is
present and `b` is absent. For many labels, `multilabel_pair_mode="top_m"`
limits each label to nearby competitors and exposes the chosen labels through
`competitors_`.

```python
oi = OverlapIndex(
    model_type="MiniBatchKMeans",
    multilabel_pair_mode="top_m",
    top_m=5,
)
oi.fit(X, label_sets)
```

### Foreground-focused aggregation

`exclude_classes` removes labels such as background or unknown from `index`
and `weighted_index` without removing them from fitting or pairwise analysis.
This preserves their effect as competitors while focusing the global summary.

```python
oi = OverlapIndex(exclude_classes="background")
oi.fit(X, y)
```

### Continuous targets

Use `ContinuousOverlapIndex` for regression targets. It asks whether nearby
feature prototypes contain compatible empirical target distributions and
calibrates the result against a permutation null. Its anchor points are
similar—`1.0` indicates no observed harmful overlap and `0.0` indicates
complete or permutation-equivalent overlap—but the calibration is different
from discrete OI. Worse-than-null disagreement also remains at `0.0` and is
identified by `loss_ratio_ > 1`. Do not directly compare scores from the two
estimators.

## A practical reporting checklist

When publishing or tracking a result, record:

1. Feature preprocessing and the evaluated data split.
2. Backend and all non-default backend settings.
3. Macro `index`, and `weighted_index` when class support matters.
4. Per-label or worst pairwise results, not only the global summary.
5. Any under-prototyped or unevaluable labels.
6. Random seeds for centroid and ball-cover backends.

See {doc}`backends/index` to choose and configure a backend, and {doc}`api` for
the complete estimator interface.
