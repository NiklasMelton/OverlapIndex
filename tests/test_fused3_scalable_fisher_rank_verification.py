import json
from pathlib import Path

import numpy as np
import pytest

from experiments.fused3_confirmation_v2 import analysis as frozen_analysis
from experiments.fused3_confirmation_v2 import datasets, recipes
from experiments.fused3_envelope_reanalysis import scalable_fisher_approximation as prior
from experiments.fused3_envelope_reanalysis import scalable_fisher_rank_verification as rank


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "artifacts" / "fused3_confirmation_v2" / "inputs"
FULL = ROOT / "artifacts" / "fused3_confirmation_v2" / "full"
PRIOR = ROOT / "artifacts" / "fused3_confirmation_v2" / "scalable_fisher_approximation_v1"


def _toy(seed=12):
    rng = np.random.default_rng(seed)
    common = rng.normal(size=(90, 1))
    correlated = np.hstack(
        [common + 0.05 * rng.normal(size=(90, 1)) for _ in range(3)]
    )
    signal = np.vstack(
        [
            rng.normal((-1.5, 0.0, 0.0), 0.25, size=(30, 3)),
            rng.normal((0.0, 1.5, 0.0), 0.25, size=(30, 3)),
            rng.normal((1.5, 0.0, 0.0), 0.25, size=(30, 3)),
        ]
    )
    return np.hstack([signal, correlated]).astype(np.float32), np.asarray(
        ["a"] * 30 + ["b"] * 30 + ["c"] * 30
    )


def _first_panel():
    registry = json.loads((INPUTS / "audited_registry.json").read_text())
    row = {item["dataset_id"]: item for item in registry["datasets"]}[
        "torchvision_cifar10"
    ]
    return datasets.load_panel(
        row,
        "dinov2-small",
        recipes.REPLICATE_SEEDS[0],
        32,
        registry_root=INPUTS,
    )


def test_protocol_and_candidate_surface_are_closed():
    digest, protocol = rank.validate_design_protocol()
    assert len(digest) == 64
    assert rank.RANK_BY_METHOD == {
        "FUSED3-LR8-FISHER-OI-U": 8,
        "FUSED3-LR16-FISHER-OI-U": 16,
        "FUSED3-LR32-FISHER-OI-U": 32,
    }
    assert protocol["scheduled_methods"] == list(rank.SCHEDULED_METHODS)
    assert protocol["scope"]["promotion_or_reselection_allowed"] is False


def test_schedule_is_exactly_position_balanced_and_deterministic():
    assert rank.canonical_panel_keys() == rank.canonical_panel_keys()
    assert len(rank.canonical_panel_keys()) == 500
    assert rank.execution_position_counts() == {
        method: [125, 125, 125, 125] for method in rank.SCHEDULED_METHODS
    }
    assert rank.execution_order(0) == rank.execution_order(0)
    with pytest.raises(ValueError, match="panel_index"):
        rank.execution_order(500)


def test_rank16_transform_is_numerically_identical_to_prior_implementation():
    values, labels = _toy()
    current = rank.fit_rank_transform(values, labels, seed=9, requested_rank=16)
    archived = prior.fit_scalable_fisher_transforms(values, labels, seed=9)
    left = current["transform"]
    right = archived["transforms"][prior.LOW_RANK]
    assert np.array_equal(left.center, right.center)
    assert np.array_equal(left.feature_scale, right.feature_scale)
    assert np.array_equal(left.nuisance_vectors, right.nuisance_vectors)
    assert np.array_equal(left.nuisance_gains, right.nuisance_gains)
    assert np.array_equal(left.discriminant_basis, right.discriminant_basis)
    assert np.array_equal(left.transform(values), right.transform(values))


def test_rank_ablation_changes_only_requested_nuisance_rank():
    values, labels = _toy()
    outputs = {
        requested: rank.fit_rank_transform(
            values, labels, seed=4, requested_rank=requested
        )
        for requested in (8, 16, 32)
    }
    # The toy has only six features, so every requested rank clips to six.
    assert {value["effective_rank"] for value in outputs.values()} == {6}
    for fitted in outputs.values():
        transform = fitted["transform"]
        assert not transform.center.flags.writeable
        assert not transform.nuisance_vectors.flags.writeable
        assert np.all(transform.nuisance_gains <= 0.0)
    with pytest.raises(ValueError, match="requested_rank"):
        rank.fit_rank_transform(values, labels, seed=4, requested_rank=12)


def test_one_real_panel_matches_frozen_LP_and_prior_rank16_exactly():
    panel = _first_panel()
    seed = recipes.REPLICATE_SEEDS[0]
    lp = rank.execute_method(
        rank.LINEAR_PROBE,
        panel["training_values"],
        panel["training_labels"],
        seed=seed,
    )
    rank16 = rank.execute_method(
        rank.RANK16,
        panel["training_values"],
        panel["training_labels"],
        seed=seed,
    )
    source = [json.loads(line) for line in (FULL / "selector_rows.jsonl").read_text().splitlines()]
    expected_lp = next(
        row["score"]
        for row in source
        if row["dataset_id"] == "torchvision_cifar10"
        and row["backbone"] == "dinov2-small"
        and row["replicate_seed"] == seed
        and row["budget"] == 32
        and row["candidate_id"] == rank.LINEAR_PROBE
    )
    prior_summary = json.loads((PRIOR / "summary.json").read_text())
    expected_rank16 = next(
        row["score"]
        for row in prior_summary["state_rows"]
        if row["dataset_id"] == "torchvision_cifar10"
        and row["backbone"] == "dinov2-small"
        and row["replicate_seed"] == seed
        and row["budget"] == 32
        and row["candidate_id"] == rank.RANK16
    )
    assert lp["score"] == expected_lp
    assert rank16["score"] == expected_rank16
    assert lp["outer_wall_seconds"] >= 0.0
    assert rank16["outer_wall_seconds"] >= 0.0


def test_warmup_signature_excludes_clocks_but_not_score_or_state():
    base = {
        "candidate_id": rank.RANK8,
        "score": 0.5,
        "recipe_sha256": "a" * 64,
        "folds": [
            {
                "fold": 0,
                "score": 0.5,
                "transform_state_sha256": "b" * 64,
                "fit_wall_seconds": 1.0,
            }
        ],
    }
    changed_clock = json.loads(json.dumps(base))
    changed_clock["folds"][0]["fit_wall_seconds"] = 999.0
    changed_score = json.loads(json.dumps(base))
    changed_score["score"] = 0.6
    assert rank._clock_free_signature(base) == rank._clock_free_signature(changed_clock)
    assert rank._clock_free_signature(base) != rank._clock_free_signature(changed_score)


def test_prior_rank16_loader_is_hash_and_identity_bound():
    verified = frozen_analysis.verify_completed_artifact(FULL)
    identity = {
        key: verified["raw"][key]
        for key in (
            "protocol_sha256",
            "code_identity_sha256",
            "run_identity_sha256",
            "audited_registry_sha256",
            "lineage_lock_sha256",
        )
    }
    rows, source = rank.load_prior_rank16_rows(PRIOR, frozen_identity=identity)
    assert len(rows) == 500
    assert len(source["manifest_sha256"]) == 64
    with pytest.raises(ValueError, match="identity differs"):
        rank.load_prior_rank16_rows(
            PRIOR,
            frozen_identity={**identity, "run_identity_sha256": "0" * 64},
        )


def test_report_exposes_parity_and_full_panel_runtime():
    summary = {
        "overall_aggregate": [
            {
                "candidate_id": rank.RANK16,
                "equal_dataset_mean_regret_pp": 0.4,
                "equal_dataset_exact_best_rate": 0.7,
                "equal_dataset_within_one_pp_rate": 0.8,
                "equal_dataset_mean_spearman": 0.9,
            }
        ],
        "overall_runtime": [
            {
                "candidate_id": rank.RANK16,
                "median_of_dataset_budget_call_wall_ratios": 0.3,
                "median_of_dataset_budget_full_panel_wall_ratios": 0.35,
                "all_dataset_budget_full_panels_faster": True,
            }
        ],
        "LP_FULL_parity_rows": [{}] * 500,
        "rank16_parity_rows": [{}] * 500,
        "warmup_verification_rows": [{}] * 40,
        "execution_position_counts": rank.execution_position_counts(),
        "regret_contrasts": [],
    }
    report = rank.render_report(summary)
    assert "500/500" in report
    assert "Full-panel ratios" in report
    assert "40/40" in report


def test_heterogeneous_LP_and_rank_fold_schemas_write_separate_tables():
    payloads = rank._fold_csv_payloads(
        [
            {
                "candidate_id": rank.LINEAR_PROBE,
                "fold": 0,
                "fit_cpu_seconds": 0.1,
            },
            {
                "candidate_id": rank.RANK8,
                "fold": 0,
                "transform_state_sha256": "a" * 64,
            },
        ]
    )
    assert set(payloads) == {"LP_FULL_fold_rows.csv", "rank_fold_rows.csv"}
    assert b"fit_cpu_seconds" in payloads["LP_FULL_fold_rows.csv"]
    assert b"transform_state_sha256" in payloads["rank_fold_rows.csv"]
