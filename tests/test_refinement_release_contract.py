"""Portable release parity and frozen-geometry contracts for native refinement."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import sklearn
from scipy import sparse
from sklearn.base import clone

from overlapindex import OverlapIndex
import overlapindex.clustering as clustering


FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/overlapindex_0_1_4_centroid_parity.json").read_text()
)
BACKENDS = ("KMeans", "MiniBatchKMeans")


def _assert_observation(model, X, expected, *, backend):
    assert model.index == pytest.approx(expected["index"], abs=1e-7)
    assert model.weighted_index == pytest.approx(expected["weighted_index"], abs=1e-7)
    assert {str(k): v for k, v in model.singleton_index.items()} == pytest.approx(
        expected["singleton_index"], abs=1e-7
    )
    assert {str(k): v for k, v in model.cluster_cardinality.items()} == expected[
        "cluster_cardinality"
    ]
    assert {str(k): sorted(v) for k, v in model.rev_map.items()} == expected[
        "class_to_clusters"
    ]
    for a, b, value in expected["pairwise_index"]:
        assert model.pairwise_index[a, b] == pytest.approx(value, abs=1e-7)
    # MiniBatchKMeans sampling changed between sklearn releases. Compare its
    # centers to the published-wheel snapshot only in the generating runtime;
    # scores, ownership and predictions remain checked on every runtime.
    if backend == "KMeans" or sklearn.__version__ == FIXTURE["generation_environment"]["scikit_learn"]:
        np.testing.assert_allclose(model._model.centers, expected["centers"], atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(model.predict(X), expected["predictions"])


@pytest.mark.parametrize("explicit_false", [False, True])
@pytest.mark.parametrize(
    "case", FIXTURE["cases"], ids=lambda case: f'{case["backend"]}-{case["storage"]}'
)
def test_default_off_matches_published_0_1_4(case, explicit_false):
    """Compare to observations from the hash-verified PyPI wheel, not to ourselves."""
    X, held = np.asarray(FIXTURE["X"]), np.asarray(FIXTURE["X_eval"])
    if case["storage"] == "csr":
        X, held = sparse.csr_matrix(X), sparse.csr_matrix(held)
    params = deepcopy(case["parameters"])
    if explicit_false:
        params["prototype_refinement"] = False
    model = OverlapIndex(**params).fit(X, FIXTURE["y"])
    assert model.prototype_refinement is False
    _assert_observation(model, X, case["fit"], backend=case["backend"])
    model.score_fixed(held, FIXTURE["y_eval"])
    _assert_observation(model, held, case["score_fixed"], backend=case["backend"])


def _refined_model(backend):
    return OverlapIndex(
        model_type=backend,
        kmeans_k=1,
        kmeans_kwargs={"random_state": 0, "n_init": 1},
        prototype_refinement=True,
    )


def _data():
    X = np.asarray([[0.0], [1.0], [2.0], [3.0], [10.0], [11.0], [12.0], [13.0]])
    return X, np.asarray([0] * 4 + [1] * 4)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("storage", ["dense", "csr", "csc", "csr_array"])
def test_fixed_scoring_never_refits_or_refines_and_preserves_fit_diagnostics(
    backend, storage, monkeypatch
):
    X, y = _data()
    converters = {
        "dense": np.asarray,
        "csr": sparse.csr_matrix,
        "csc": sparse.csc_matrix,
        "csr_array": sparse.csr_array,
    }
    convert = converters[storage]
    model = _refined_model(backend).fit(convert(X), y)
    backend_before = model._model
    centers_before = model._model.centers.copy()
    ownership_before = deepcopy(model._model.class_to_clusters)
    summary_before = deepcopy(model.prototype_refinement_)
    assert summary_before["applied_count"] == 2

    def forbidden(*args, **kwargs):
        pytest.fail("score_fixed attempted to fit or refine held-out rows")

    monkeypatch.setattr(clustering, "apply_balanced_median_refinement", forbidden)
    monkeypatch.setattr(model._model, "fit_offline", forbidden)
    # Fitted geometry is frozen; evaluation support/overlap diagnostics update.
    for offset in (0.25, -0.25):
        assert np.isfinite(model.score_fixed(convert(X[::2] + offset), y[::2]))
        assert model._model is backend_before
        np.testing.assert_array_equal(model._model.centers, centers_before)
        assert model._model.class_to_clusters == ownership_before
        assert model.prototype_refinement_ == summary_before
        assert dict(model.cluster_cardinality) == {0: 2, 1: 2}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("target_format", ["collections", "dense_indicator", "sparse_indicator"])
def test_multilabel_fit_and_fixed_score_are_rejected_before_geometry_changes(
    backend, target_format
):
    X, y = _data()
    targets = np.eye(2, dtype=int)[y]
    targets[0] = [1, 1]
    if target_format == "collections":
        targets = [set(np.flatnonzero(row)) for row in targets]
    elif target_format == "sparse_indicator":
        targets = sparse.csr_matrix(targets)
    with pytest.raises(ValueError, match="multi-label"):
        _refined_model(backend).fit(X, targets)
    model = _refined_model(backend).fit(X, y)
    summary = deepcopy(model.prototype_refinement_)
    centers, score = model._model.centers.copy(), model.index
    with pytest.raises(ValueError, match="multi-label"):
        model.score_fixed(X, targets)
    np.testing.assert_array_equal(model._model.centers, centers)
    assert model.prototype_refinement_ == summary
    assert model.index == score


@pytest.mark.parametrize("backend", ["BallCover", "Fuzzy", "Hypersphere", "FusedKMeans"])
def test_unsupported_backends_reject_refinement_before_optional_import(backend, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported refinement reached ARTMAP construction")

    monkeypatch.setattr(clustering._ARTMAPManyToOne, "__init__", forbidden)
    with pytest.raises(ValueError, match="supported only.*KMeans.*MiniBatchKMeans"):
        OverlapIndex(model_type=backend, prototype_refinement=True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_refinement_clones_and_parameter_updates_preserve_default_contract(backend):
    X, y = _data()
    model = _refined_model(backend)
    copied = clone(model).fit(X, y)
    assert copied.get_params() == model.get_params()
    assert copied.prototype_refinement_["method"] == "balanced_median"
    assert model.set_params(prototype_refinement=False) is model
    assert clone(model).get_params()["prototype_refinement"] is False
    model.fit(X, y)
    assert model.prototype_refinement_["method"] == "none"
    assert model.prototype_refinement_["prototype_count_after"] == 2
