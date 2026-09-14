import copy

import numpy as np
import pytest

from experiments.continuous_reference_historical_screen import run


def test_protocol_is_frozen_and_grid_counts_match_inventory():
    protocol = run.load_protocol()
    assert protocol["scope"]["overlapindex_evaluated"] is False
    assert protocol["scope"]["reserved_backbones_evaluated"] is False
    assert len(protocol["targets"]) == 8
    assert sum(len(row["draw_seeds"]) for row in protocol["targets"]) == 44
    assert protocol["expected_grid"] == {
        "target_count": 8,
        "candidate_count": 40,
        "candidate_draw_count": 220,
        "overlapindex_fit_count": 0,
        "reserved_backbone_count": 0,
    }


def test_every_archived_target_has_exact_declared_capacity():
    protocol = run.load_protocol()
    cohort, _ = run.q1.v1_extract.load_bank()
    for config in protocol["targets"]:
        dataset_id, target_id = config["target_key"].split("::", 1)
        rows = cohort["datasets"][dataset_id]["target_cohorts"][target_id]
        training = [row for row in rows if row["split"] == "training"]
        evaluation = [row for row in rows if row["split"] == "evaluation"]
        assert len(training) >= config["training_pool_rows"]
        assert len(evaluation) == config["evaluation_rows"]
        assert len({row["sample_id"] for row in training}) == len(training)
        assert {row["sample_id"] for row in training}.isdisjoint(
            row["sample_id"] for row in evaluation
        )


def _panel_for_bootstrap():
    rng = np.random.default_rng(31)
    y = rng.normal(size=300)
    candidates = []
    for candidate, noise in zip(run.BACKBONES, [0.05, 0.15, 0.25, 0.35, 0.45]):
        prediction = y + noise * rng.normal(size=len(y))
        heads = {
            head: {"prediction": prediction.tolist()}
            for head in run.HEADS
        }
        score = 1.0 - noise
        candidates.append(
            {
                "candidate_id": candidate,
                "head_mean_evaluation_r2": {head: score for head in run.HEADS},
                "draws": [{"heads": heads}],
            }
        )
    return {
        "target_key": "synthetic-target",
        "evaluation_target": y.tolist(),
        "candidates": candidates,
    }


def test_screen_bootstrap_is_deterministic_and_uses_simultaneous_quantiles():
    protocol = copy.deepcopy(run.load_protocol())
    protocol["uncertainty"]["resamples"] = 100
    panel = _panel_for_bootstrap()
    left = run.screen_panel(panel, protocol)
    right = run.screen_panel(panel, protocol)
    assert left == right
    assert left["simultaneous_quantiles"] == pytest.approx([0.00125, 0.99875])
    assert left["envelope_best_candidate_ids"] == [run.BACKBONES[0]]
    assert left["linear_best_candidate_ids"] == [run.BACKBONES[0]]
    assert left["linear_top1_regret_point_r2"] == pytest.approx(0.0)


def test_summary_requires_two_nonlinear_sources_for_sufficiency():
    protocol = run.load_protocol()
    base_screen = {
        "classification": "nonlinear_challenge_lead",
        "learnable": True,
        "nonlinear_opportunity": True,
        "ranking_conflict": True,
        "winner_identifiable": True,
    }
    panels = [
        {"target_key": "a::one", "dataset_id": "a", "draw_count": 1, "draw_limitation": None, "screen": dict(base_screen)},
        {"target_key": "a::two", "dataset_id": "a", "draw_count": 1, "draw_limitation": None, "screen": dict(base_screen)},
    ]
    assert run.summarize(protocol, panels)["decision"]["sufficient_historical_nonlinear_diversity"] is False
    panels[1]["dataset_id"] = "b"
    assert run.summarize(protocol, panels)["decision"]["sufficient_historical_nonlinear_diversity"] is True


def test_validate_grid_fails_closed_on_partial_checkpoint():
    with pytest.raises(ValueError, match="executed grid changed"):
        run.validate_grid(run.load_protocol(), {"panels": []})


def test_qualified_library_is_exact_and_passing():
    protocol = run.load_protocol()
    library = run.q3.load_protocol()
    assert run.base.sha256_path(run.q3.PROTOCOL_PATH) == protocol["ancestry"]["qualification_v3_protocol_sha256"]
    terminal = run.q3.validate_artifacts(run.Q3_ROOT)
    assert terminal["decision"]["qualification_pass"] is True

