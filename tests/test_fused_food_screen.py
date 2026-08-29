from __future__ import annotations

import numpy as np

from experiments.scalable_relevance_kmeans import fused_food_screen as screen


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260829)
    labels = np.repeat(np.asarray(["a", "b", "c"], dtype=object), 30)
    values = rng.normal(size=(labels.size, 24)).astype(np.float32)
    for position, label in enumerate(("a", "b", "c")):
        values[labels == label, position] += np.float32(2.0)
    return values, labels


def test_fused_food_protocol_matches_closed_implementation() -> None:
    digest, protocol = screen.validate_protocol()
    assert len(digest) == 64
    assert protocol["development_grid"]["methods"] == list(screen.METHODS)
    assert protocol["candidate_specs"] == screen._candidate_specs()


def test_full_schedule_is_cyclic_near_counterbalanced() -> None:
    counts = {method: [0] * len(screen.METHODS) for method in screen.METHODS}
    for panel_index in range(30):
        order = screen._method_order(panel_index)
        assert set(order) == set(screen.METHODS)
        for position, method in enumerate(order):
            counts[method][position] += 1
    assert all(max(values) - min(values) <= 1 for values in counts.values())
    assert all(sum(values) == 30 for values in counts.values())


def test_looped_and_fused_crossfit_states_and_scores_match_exactly() -> None:
    values, labels = _fixture()
    looped = screen.cross_fitted_candidate(
        values, labels, candidate_id="LOOP3", seed=17
    )
    fused = screen.cross_fitted_candidate(
        values, labels, candidate_id="FUSED3", seed=17
    )
    assert looped["score"] == fused["score"]
    assert [row["score"] for row in looped["folds"]] == [
        row["score"] for row in fused["folds"]
    ]
    assert [row["numerical_state_sha256"] for row in looped["folds"]] == [
        row["numerical_state_sha256"] for row in fused["folds"]
    ]
    assert [row["score_diagnostics"] for row in looped["folds"]] == [
        row["score_diagnostics"] for row in fused["folds"]
    ]
    assert all(row["state_unchanged_after_score_fixed"] for row in fused["folds"])


def test_determinism_ignores_nested_runtime_fields_only() -> None:
    values, labels = _fixture()
    first = screen.cross_fitted_candidate(
        values, labels, candidate_id="FUSED3", seed=19
    )
    second = screen.cross_fitted_candidate(
        values, labels, candidate_id="FUSED3", seed=19
    )
    assert screen._determinism(first, second)["exact"] is True
    second["folds"][0]["runtime_diagnostics"]["prototype_fit_wall_seconds"] += 10.0
    assert screen._determinism(first, second)["exact"] is True
    second["folds"][0]["score"] += 0.01
    assert screen._determinism(first, second)["exact"] is False


def test_valid_prior_speed_artifact_has_exact_parity_surface() -> None:
    lookup = screen._load_prior_speed_rows()
    assert len(lookup) == 60
    assert set(method for _panel, method in lookup) == {"NOLLOYD", "LP-FULL"}


def test_source_identity_declares_new_runner_analysis_and_dependencies() -> None:
    names = {path.name for path in screen._source_paths()}
    assert {"fused_food_screen.py", "fused_food_analysis.py", "fused_prototypes.py"} <= names
    assert "OverlapIndex.py" in names
    assert "pyproject.toml" in names
