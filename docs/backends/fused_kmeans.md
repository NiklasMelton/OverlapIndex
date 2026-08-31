# FusedKMeans and shared Fisher

`FusedKMeans` is an opt-in offline backend for ranking labeled embedding
spaces. It fits class-owned prototypes with three fixed, full-batch Lloyd
updates in one padded NumPy kernel instead of constructing one scikit-learn
estimator per class. It can then learn a capped diagonal relevance weight for
the fixed prototypes. No weighted Lloyd refit is performed.

The backend is most useful with the opt-in shared-Fisher transform. That
transform is supervised preprocessing: fit it only on training rows and score
held-out rows with `score_fixed`, or use `cross_fit_score`.

## Recommended ranking configuration

```python
from overlapindex import OverlapIndex

oi = OverlapIndex(
    model_type="FusedKMeans",
    kmeans_k=10,
    feature_normalization="l2",
    feature_transform="shared_fisher",
    fisher_rank=32,
    fisher_random_state=0,
    fused_kmeans_kwargs={
        "random_state": 0,
        "n_iter": 3,
        "min_samples_per_prototype": 5,
        "relevance_weighting": True,
        "margin_rows_per_class": 32,
        "weight_floor": 0.05,
    },
)

score = oi.cross_fit_score(X, y, n_splits=5, random_state=0)
```

`kmeans_k` is a maximum for this backend. The resolved count for a class is
also capped at `floor(class_rows / min_samples_per_prototype)`, with a minimum
of one. Use at least enough rows to resolve two prototypes in every fold.

## Fisher rank

A nonnegative integer `fisher_rank` selects the scalable transform:

1. optionally row-L2 normalize;
2. center by the global training mean;
3. divide by pooled within-class diagonal standard deviations with a fixed
   variance floor;
4. estimate the requested number of correlated residual directions with
   randomized SVD;
5. deflate only residual directions whose estimated eigenvalue exceeds one;
6. project corrected class-mean contrasts into at most `classes - 1`
   discriminant dimensions.

Rank zero retains diagonal within-class scaling but skips correlated-residual
deflation. Ranks larger than the available residual dimension are safely capped. Set
`fisher_rank="full"` to use the full SVD Fisher/LDA transform instead. Full
Fisher is a quality-oriented ceiling and can be substantially more expensive
on wide embeddings; ranks 16 and 32 are the intended scalable settings.

## Leakage and score interpretation

Calling `fit(X, y)` exposes the in-sample score as `index`, as it does for
every existing backend. For backbone ranking, prefer `cross_fit_score`: every
fold learns normalization-independent Fisher state, prototypes, and relevance
weights only from its training rows, then evaluates the untouched fold with
fixed state. The method leaves the original estimator unfitted and records the
fold values in `cross_fit_scores_`.

This path requires dense features and scalar labels. It does not support
`prototype_refinement=True`. Existing defaults remain unchanged unless these
options are selected explicitly.

After a direct fit, inspect `fisher_diagnostics_`,
`fused_kmeans_diagnostics_`, and `feature_weights_` for the resolved rank,
dimensions, prototype counts, and relevance-weight range.
