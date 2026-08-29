from __future__ import annotations

from collections import Counter

import numpy as np

from experiments.scalable_relevance_kmeans import fused_food_full as full


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260829)
    labels = np.repeat(np.asarray(["a", "b", "c"], dtype=object), 30)
    values = rng.normal(size=(labels.size, 16)).astype(np.float32)
    for position, label in enumerate(("a", "b", "c")):
        values[labels == label, position] += np.float32(2.0)
    return values, labels


def test_full_protocol_and_screen_lock_are_exact() -> None:
    digest, protocol = full.validate_protocol()
    assert len(digest) == 64
    assert protocol["grid"]["panel_count"] == 600
    assert protocol["grid"]["executed_methods"] == list(full.METHODS)
    decision = full._validate_screen_lock()
    assert decision["selected_candidate"] == "FUSED3"
    assert decision["status"] == "pass_for_full_food_design"


def test_full_schedule_is_exactly_counterbalanced() -> None:
    counts: Counter[tuple[str, int]] = Counter()
    for panel_index in range(600):
        order = full._method_order(panel_index)
        assert set(order) == set(full.METHODS)
        for position, method in enumerate(order):
            counts[(method, position)] += 1
    assert set(counts.values()) == {200}


def test_frozen_full_comparator_inputs_are_complete() -> None:
    prior, screen, archived = full._load_frozen_inputs()
    assert len(prior) == 1200
    assert len(screen) == 30
    assert len(archived) == 600
    assert {method for *_panel, method in prior} == {
        "LP-FULL",
        "REFINED-OI-ARCHIVED",
    }


def test_fresh_refined_and_probe_use_closed_crossfit_contracts() -> None:
    values, labels = _fixture()
    refined = full._execute_method("REFINED-OI", values, labels, 17)
    probe = full._execute_method("LP-FULL", values, labels, 17)
    assert refined["candidate_id"] == "REFINED-OI"
    assert refined["prototype_refinement"] is True
    assert probe["candidate_id"] == "LP-FULL"
    assert probe["prototype_refinement"] is False
    assert len(refined["folds"]) == len(probe["folds"]) == full.FOLDS
    assert [row["seed"] for row in refined["folds"]] == list(range(17, 22))
    assert {row["split_seed"] for row in probe["folds"]} == {17}


def test_full_determinism_excludes_clocks_but_not_scores() -> None:
    first = {
        "candidate_id": "FUSED3",
        "candidate_name": "fused",
        "mode": "fused",
        "prototype_refinement": False,
        "score": 0.25,
        "folds": [{"fold": 0, "score": 0.25, "fit_wall_seconds": 1.0}],
    }
    second = {
        **first,
        "folds": [{"fold": 0, "score": 0.25, "fit_wall_seconds": 9.0}],
    }
    assert full._determinism(first, second)["exact"] is True
    second["folds"][0]["score"] = 0.26
    assert full._determinism(first, second)["exact"] is False


def test_full_source_identity_declares_runner_analysis_and_local_oi() -> None:
    names = {path.name for path in full._source_paths()}
    assert {"fused_food_full.py", "fused_food_full_analysis.py"} <= names
    assert "OverlapIndex.py" in names
    assert "pyproject.toml" in names
