# Optional prototype refinement

`OverlapIndex(prototype_refinement=True)` enables one balanced
observation-median pass after fitting `KMeans` or `MiniBatchKMeans`.
The default is `False`; the default backend remains `MiniBatchKMeans`.
Refinement changes prototype resolution, so it can change both the score and
the cost of subsequent scoring. It does not guarantee better rankings.

```python
from overlapindex import OverlapIndex

# Use the same training-only preprocessing and prototype count as your caller.
oi = OverlapIndex(
    model_type="MiniBatchKMeans",
    kmeans_k=10,
    kmeans_kwargs={"random_state": 0},
    prototype_refinement=True,
).fit(X_train, y_train)

holdout_score = oi.score_fixed(X_holdout, y_holdout)
print(oi.prototype_refinement_["applied_count"])
```

The holdout must cover the same labels as the training set. `score_fixed`
updates evaluation overlap/support diagnostics while preserving fitted
prototypes, ownership, and fit-time refinement diagnostics. It never refits or
refines on holdout rows. `fit`, `score(X, y)`, and offline `partial_fit` build
a fresh backend and perform at most one refinement pass per fit.

## The fixed algorithm

Eligibility is computed once from the original centers. A parent needs at least
two assigned own-class observations and zero outgoing own-runner-up isolation
count: every assigned row has some competing class prototype strictly beating
its own-class second-best score.

For each eligible parent, the algorithm chooses the observation farthest from
the parent, then the observation farthest from that first point. It projects
the support onto their connecting axis and splits the ordered rows into two
halves whose sizes differ by at most one. Each child is the observed row nearest
its half's coordinate-wise median. Original row indices resolve ties, and
lexicographic child ordering fixes the resulting IDs. The pass is deterministic
for fixed fitted centers and input row order; seed the underlying centroid
backend to make the complete fit reproducible.

The first child keeps its parent's global ID; the second is appended. Both
retain the parent's class ownership. Children are never reconsidered during
the same pass. Duplicate observations/representatives or invalid splits leave
the parent unchanged, with a recorded skip reason. No second clustering fit,
recursive splitting, or tunable refinement threshold is introduced.

## Contract and memory

- `prototype_refinement` accepts Python `bool` only, participates in
  `get_params`, `set_params`, and sklearn cloning, and defaults to `False`.
- Enabling it on BallCover, Fuzzy ARTMAP, or Hypersphere ARTMAP raises an error.
  Targets with multiple positive labels per row are rejected during both
  fitting and fixed scoring, including collections and dense/sparse indicators.
  Single-positive indicator rows retain single-label semantics.
- Dense and sparse features are supported by both centroid backends. Sparse
  input remains sparse for the isolation scan; selecting representatives
  densifies one eligible parent's support block. That block is not capped by
  `offline_memory_budget_mb`, which budgets scorer scratch space rather than
  total fit memory. More children also increase retained center storage.
- `prototype_refinement_` reports `method` (`"none"` or `"balanced_median"`),
  prototype counts before/after, eligible/attempted/applied/skipped counts,
  parent ID lists, and per-parent `records`. Applied records identify child
  IDs, support sizes, class ownership, and selected training-row indices;
  skipped records include a reason. See [diagnostics](diagnostics.md).

## Release and evidence scope

The refinement implementation in `develop` came from
[`606c5f0`](https://github.com/NiklasMelton/OverlapIndex/commit/606c5f00e43d419a6a733389c27aa303bd5d31a0).
`score_fixed`, sparse scoring, and the universal bounded-memory scorer already
exist in released 0.1.4. The 0.1.5 release preparation adds frozen parity
tests and evidence for refinement. `develop` also includes the separate
Fisher/FusedKMeans integration; the evidence below does not evaluate that path.

The [canonical evidence page](native_refinement_evidence.md) describes the
tested KMeans16-plus-refinement shortlist policy and its limitations. Those
results do not establish the same numerical outcomes for MiniBatchKMeans,
adaptive counts, other preprocessing, or regression, and do not isolate the
effect of refinement itself.
