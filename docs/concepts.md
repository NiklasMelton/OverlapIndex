# How the discrete index works

OverlapIndex separates prototype construction from overlap scoring. The chosen
backend builds prototypes owned by labels, and the estimator applies the same
directional class-pair logic to those prototypes.

## From samples to prototypes

For an offline backend, a separate KMeans, MiniBatchKMeans, or BallCover model
is fit for every label. Their local prototypes are concatenated into one set of
global prototype IDs while ownership is retained. A multi-label sample is
copied once for every positive label during prototype fitting.

For an ARTMAP backend, supervised incremental learning creates and updates the
label-owned prototypes directly.

`predict(X)` returns these global prototype IDs. It is therefore useful for
inspecting the fitted representation, but it does **not** return predicted
class labels.

## Directional pairwise scores

For a source label $a$ and competitor $b$, each source sample is compared only
against prototypes owned by $a$ or $b$. The estimator finds the sample's own
best-matching prototype, then checks whether the strongest alternative belongs
to $b$. Let $O_{a,b}$ be the number of these overlap events and $N_{a,b}$ the
number of evaluable source rows. The directional score is

$$
I_{a,b} = 1 - \frac{O_{a,b}}{N_{a,b}}.
$$

The direction matters: $I_{a,b}$ and $I_{b,a}$ can differ because their source
rows and denominators differ. On ordinary single-label data,
$N_{a,b}$ is the support of $a$. On multi-label data, it contains only rows
where $a$ is present and $b$ is absent.

## Per-label and global aggregation

The per-label score is its worst evaluable competitor:

$$
I_a = \min_{b \ne a} I_{a,b}.
$$

The public `index` is the macro mean of $I_a$ over evaluable labels not listed
in `exclude_classes`. Every included label has equal influence. The
`weighted_index` property instead weights each $I_a$ by that label's positive
sample support in `cluster_cardinality`.

This hierarchy is reflected directly in fitted attributes:

| Level | Attribute | Key |
| --- | --- | --- |
| Directional class pair | `pairwise_index` | `(source, competitor)` |
| Source label | `singleton_index` | `source` |
| Global macro | `index` | scalar |
| Global support-weighted | `weighted_index` | scalar property |

## Interpreting values

- `1.0`: no overlap event was observed at the chosen prototype resolution.
- Between `0.0` and `1.0`: partial overlap or separation.
- `0.0`: every evaluable source row produced an overlap event, indicating
  complete overlap at the chosen prototype resolution.

These are interpretation anchors, not universal quality thresholds. The score
depends on the feature representation, scaling, backend geometry, prototype
count, and evaluated data. It is meaningful to compare runs only when those
choices are controlled.

## Prototype resolution

Top-two scoring is best resolved with at least two prototypes per label. When
a label owns fewer than two, fitting continues but emits a warning and records
the label in `under_prototyped_labels_`. This often occurs when `kmeans_k=1`, a
class contains only one sample, or a BallCover configuration produces a single
ball.

More prototypes are not automatically better. They can represent curved or
multimodal support more faithfully, but also increase runtime and change the
scale at which overlap is measured. Tune resolution on a representative
dataset, then hold it fixed for comparisons.

## Excluded labels

`exclude_classes` affects only the final macro and weighted aggregations. An
excluded label still owns prototypes, acts as a competitor, and appears in
`singleton_index`, `pairwise_index`, and support bookkeeping. This is useful
for a background or unknown label that should influence foreground overlap but
should not receive equal weight in the global summary.
