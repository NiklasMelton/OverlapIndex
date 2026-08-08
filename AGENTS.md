# Agent Notes

## Repository Shape

This repository contains the `overlapindex` Python package. The public entry
point is `overlapindex.OverlapIndex`, re-exported from `overlapindex/__init__.py`.

Key modules:

- `overlapindex/OverlapIndex.py` contains input validation, public estimator
  methods, overlap bookkeeping, single-label scoring, and multi-label scoring.
- `overlapindex/clustering.py` defines the private many-to-one backend adapter
  interface and wraps ARTMAP, KMeans, MiniBatchKMeans, and BallCover backends.
- `overlapindex/BallCover.py` implements the offline greedy ball-cover backend.
- `overlapindex/utils.py` contains complement coding and pairwise top-k helpers.
- `examples/` contains runnable examples for the default MiniBatchKMeans path,
  ARTMAP usage, BallCover, backend comparison, and `partial_fit` behavior.
- `tests/` contains behavior and regression coverage. The yeast fixture is kept
  at `tests/data_yeast.data`.

There are no planning files in the repository. Treat `README.md`, tests, and the
package source as the source of truth for behavior.

## Package Behavior

`OverlapIndex` is a scikit-learn-style estimator for computing an overlap index
over class-owned prototypes. The default backend is `MiniBatchKMeans`.

Supported `model_type` values:

- `"MiniBatchKMeans"`: offline batch backend; fits one MiniBatchKMeans model per
  label and concatenates class-owned centers into global prototype ids.
- `"KMeans"`: offline batch backend with the same many-to-one center layout.
- `"BallCover"`: offline batch backend; fits one greedy ball cover per label and
  treats balls as global prototypes.
- `"Fuzzy"` and `"Hypersphere"`: ARTMAP backends; require the optional `art`
  extra and support incremental updates.

Offline backends use normalized features directly and do not complement-code
inputs. ARTMAP backends complement-code internally and require features in the
`[0, 1]` interval.

`fit(X, y)` returns `self`, `add_batch(X, y)` returns the computed float score,
`score()` returns the stored score, and `score(X, y)` refits before returning a
score. `predict(X)` returns global prototype ids and raises before fitting.
`partial_fit(X, y)` performs true incremental updates only for ARTMAP backends;
for offline backends it refits and recomputes on the provided batch.

`index` is the macro mean of per-label minimum pairwise overlap scores.
`weighted_index` weights each label's `singleton_index` by positive sample
support from `cluster_cardinality`.

## Multi-Label Support

`OverlapIndex` accepts these target formats:

- A 1D single-label vector.
- A 1D sequence where each sample entry is a collection of labels.
- A dense or sparse 2D binary indicator matrix, where positive columns become
  integer label ids.

Every sample must have at least one label. Dense and sparse 2D indicator targets
must contain only `0` and `1` values.

Multi-label scoring is implemented for offline backends with vectorized
prototype scoring: `KMeans`, `MiniBatchKMeans`, and `BallCover`. Multi-label
data is expanded before backend fitting so each positive label owns a copy of
the sample. Cardinalities remain per-label positive sample counts.

For multi-label overlap scoring, a pair `(a, b)` is evaluated on rows where
label `a` is present and label `b` is absent. Pairwise denominators are stored in
`pairwise_cardinality`. The model also caches sparse label indicator matrices
and label maps in `label_to_index_` and `index_to_label_`.

`multilabel_pair_mode="all"` compares each label against every other label.
`multilabel_pair_mode="top_m"` limits each source label to its nearest competing
labels by prototype-to-prototype distance and requires a positive integer
`top_m`. Selected competitors are exposed through `competitors_`.

ARTMAP `add_batch` rejects multi-label batches, and `add_sample` remains a
single-label online update path.

## Testing Expectations

Use Poetry for package-aware test commands:

```bash
poetry run python -m pytest -q tests/test_overlap_index_regression.py
poetry run python -m pytest -q tests/test_real_world_dataset_regression.py
```

The ARTMAP tests are skipped unless the optional `art` extra is installed.
Offline backend tests must pass without importing `artlib`.

When changing multi-label behavior, run
`tests/test_overlap_index_regression.py`; it covers sequence-of-labels targets,
binary indicator targets, `top_m` validation, and competitor limiting.

When changing BallCover behavior, include tests that exercise
`model_type="BallCover"` because it is offline-only and exposes the same
prototype-scoring interface used by the multi-label path.

## Development Notes

Keep the backend adapter contract in `overlapindex/clustering.py` intact:
backends expose global prototype ids, `class_to_clusters`, `topk`, and
class-restricted BMU helpers.

Preserve sklearn estimator conventions for `get_params`, `set_params`, `fit`,
`partial_fit`, `score`, `predict`, and `fit_predict`.

Avoid importing `artlib` at module import time. It should be loaded only when an
ARTMAP backend is requested.

Keep error messages precise around input shape, non-finite values, binary
indicator validation, empty data, single-class data, and unsupported multi-label
online paths; the tests assert several of these messages.
