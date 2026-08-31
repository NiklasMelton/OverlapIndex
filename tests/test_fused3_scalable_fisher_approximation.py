import json
from pathlib import Path

import numpy as np
import pytest

from experiments.fused3_confirmation_v2 import analysis as frozen_analysis
from experiments.fused3_envelope_reanalysis import scalable_fisher_approximation as sf
from experiments.m50_backbone_ranking import food101
from experiments.scalable_relevance_kmeans import fused_food_screen


ROOT = Path(__file__).resolve().parents[1]


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
    values = np.hstack([signal, correlated]).astype(np.float32)
    labels = np.asarray(["a"] * 30 + ["b"] * 30 + ["c"] * 30)
    return values, labels


def test_closed_candidate_surface_and_protocol():
    assert sf.SCALABLE_METHODS == (
        "FUSED3-MEAN-SUBSPACE-OI-U",
        "FUSED3-DIAG-FISHER-OI-U",
        "FUSED3-LR16-FISHER-OI-U",
    )
    assert sf.NUISANCE_RANK == 16
    assert sf.RANDOMIZED_POWER_ITERATIONS == 1
    digest, protocol = sf.validate_design_protocol()
    assert len(digest) == 64
    assert protocol["candidates"] == list(sf.SCALABLE_METHODS)
    assert protocol["scope"]["promotion_or_reselection_allowed"] is False


def test_scalable_transforms_are_deterministic_shared_and_detached():
    values, labels = _toy()
    first = sf.fit_scalable_fisher_transforms(values, labels, seed=9)
    second = sf.fit_scalable_fisher_transforms(values, labels, seed=9)
    assert set(first["transforms"]) == set(sf.SCALABLE_METHODS)
    for method in sf.SCALABLE_METHODS:
        left = first["transforms"][method]
        right = second["transforms"][method]
        assert left.classes == ("a", "b", "c")
        assert left.output_dimension == 2
        assert left.state_sha256 == right.state_sha256
        assert np.array_equal(left.transform(values), right.transform(values))
        assert not left.center.flags.writeable
        assert not left.feature_scale.flags.writeable
        assert not left.discriminant_basis.flags.writeable
    assert first["transforms"][sf.MEAN_ONLY].effective_nuisance_rank == 0
    assert first["transforms"][sf.DIAGONAL].effective_nuisance_rank == 0
    assert first["transforms"][sf.LOW_RANK].effective_nuisance_rank == 6


def test_low_rank_correction_only_deflates_correlated_nuisance():
    values, labels = _toy()
    fitted = sf.fit_scalable_fisher_transforms(values, labels, seed=4)
    low_rank = fitted["transforms"][sf.LOW_RANK]
    assert fitted["nuisance_eigenvalues"].max() > 1.0
    assert np.all(low_rank.nuisance_gains <= 0.0)
    assert np.all(low_rank.nuisance_gains > -1.0)
    assert np.any(low_rank.nuisance_gains < 0.0)
    before = low_rank.state_sha256
    low_rank.transform(values + np.float32(50.0))
    assert low_rank.state_sha256 == before


def test_original_heterogeneous_labels_are_preserved():
    values, _labels = _toy()
    labels = np.asarray([1] * 30 + ["two"] * 30 + [3.5] * 30, dtype=object)
    before = labels.copy()
    result = sf.fit_scalable_fisher_transforms(values, labels, seed=3)
    assert np.array_equal(labels, before)
    assert all(
        transform.classes == (1, "two", 3.5)
        for transform in result["transforms"].values()
    )


def test_crossfit_executes_all_candidates_and_raw_control_is_exact():
    values, labels = _toy()
    result = sf.crossfit_scalable_fisher(values, labels, seed=23)
    assert set(result["scores"]) == set(sf.SCALABLE_METHODS)
    assert len(result["folds"]) == 5
    assert all(np.isfinite(value) for value in result["scores"].values())
    assert all(value >= 0.0 for value in result["method_wall_seconds"].values())
    for row in result["folds"]:
        assert row["mean_only_state_unchanged"] is True
        assert row["diagonal_state_unchanged"] is True
        assert row["low_rank_16_state_unchanged"] is True
        assert row["output_dimension"] == 2

    normalized = food101._row_l2(values)
    direct = []
    for fold, (train, holdout) in enumerate(
        food101._stratified_folds(labels, n_splits=5, seed=23)
    ):
        k_per_class = fused_food_screen._k_per_class(labels, train)
        selector = fused_food_screen._build_selector("FUSED3", k_per_class, 23 + fold)
        selector.fit(normalized[train], labels[train])
        direct.append(selector.score_fixed(normalized[holdout], labels[holdout]))
    assert result["raw_score"] == float(np.mean(direct))


def test_full_fisher_source_is_hash_and_identity_bound():
    full = ROOT / "artifacts" / "fused3_confirmation_v2" / "full"
    fisher = (
        ROOT
        / "artifacts"
        / "fused3_confirmation_v2"
        / "fisher_metric_aggregation_2x2x2_v1"
    )
    verified = frozen_analysis.verify_completed_artifact(full)
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
    rows, source = sf.load_full_fisher_rows(fisher, frozen_identity=identity)
    assert len(rows) == 500
    assert all(row["candidate_id"] == sf.FULL_FISHER for row in rows)
    assert len(source["manifest_sha256"]) == 64
    with pytest.raises(ValueError, match="identity differs"):
        sf.load_full_fisher_rows(fisher, frozen_identity={**identity, "run_identity_sha256": "0" * 64})


def test_report_exposes_regret_and_speed_tradeoff():
    summary = {
        "overall_aggregate": [
            {
                "candidate_id": sf.LOW_RANK,
                "equal_dataset_mean_regret_pp": 0.3,
                "equal_dataset_exact_best_rate": 0.8,
                "equal_dataset_within_one_pp_rate": 0.9,
                "equal_dataset_mean_spearman": 0.75,
            }
        ],
        "dataset_aggregate": [],
        "regret_contrasts": [],
        "speed_tradeoff": [],
    }
    report = sf.render_report(summary)
    assert sf.LOW_RANK in report
    assert "Runtime relative to full Fisher" in report
    assert "No promotion" in report
