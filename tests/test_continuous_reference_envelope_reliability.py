"""Focused checks for the frozen reference-envelope reliability audit."""

from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from experiments.continuous_reference_envelope_reliability import run


def test_protocol_sidecar_and_frozen_grid():
    protocol = run.load_protocol()

    assert run.sha256_path(run.PROTOCOL_PATH) == run.PROTOCOL_SHA256_PATH.read_text(
        encoding="utf-8"
    ).strip()
    assert protocol["scope"]["method_promotion_allowed"] is False
    assert protocol["backbones"]["reserved_evaluated"] == []
    assert protocol["expected_grid"]["archive_panel_count"] == 60
    assert protocol["expected_grid"]["budget_new_sklearn_estimator_fit_count"] == 18880
    assert protocol["evaluation_bootstrap"]["archive_100_status"] == (
        "unavailable_archived_predictions_not_stored"
    )
    assert protocol["budget_curve"]["selection_status"].startswith("source-selected")


def test_source_chain_and_archive_audit_are_complete_without_refits():
    protocol = run.load_protocol()
    raw, summary, manifest, cohort = run.validate_sources(protocol)
    archive = run.archive_audit(raw, protocol)

    assert summary["status"] == manifest["status"] == raw["status"]
    assert len(cohort["datasets"]) == 5
    assert archive["panel_count"] == 60
    assert archive["candidate_count"] == 300
    assert archive["evaluation_predictions_available"] is False
    assert len(archive["target_level_seed_stability"]) == 20
    assert len(archive["target_weighted_group_summaries"]) == 4
    assert set(archive["head_confusion"]) == set(run.ARCHIVE_GROUPS)
    assert archive["head_confusion"]["all"]["total"] == 300


def test_archive_cluster_intervals_use_target_and_dataset_units():
    protocol = run.load_protocol()
    raw, _summary, _manifest, _cohort = run.validate_sources(protocol)
    archive = run.archive_audit(raw, protocol)
    intervals = run._archive_cluster_intervals(archive, protocol)

    assert intervals["all"]["target_count"] == 20
    assert intervals["all"]["dataset_count"] == 5
    assert intervals["all"]["bootstrap_unit"].startswith("target block")
    assert set(intervals["all"]["target_block_intervals"]) >= {
        "spearman",
        "top1_regret_r2",
    }
    assert set(intervals["all"]["dataset_cluster_intervals"]) >= {
        "spearman",
        "top1_regret_r2",
    }


def test_budget_row_order_and_sizes_are_frozen():
    protocol = run.load_protocol()
    _raw, _summary, _manifest, cohort = run.validate_sources(protocol)

    for budget in protocol["budget_curve"]["budgets"]:
        target = protocol["budget_curve"]["targets"][0]
        train, evaluation, y_train, y_eval, train_ids, eval_ids = run._budget_rows(
            cohort, target, budget["training_replicate_seeds"]
        )
        assert len(train) == budget["rows"]
        assert len(evaluation) == 200
        assert y_train.shape == (budget["rows"],)
        assert y_eval.shape == (200,)
        assert len(train_ids) == len(set(train_ids))
        assert len(eval_ids) == len(set(eval_ids))
        assert not set(train_ids).intersection(eval_ids)
        target_rows = cohort["datasets"][target.split("::", 1)[0]]["target_cohorts"][target.split("::", 1)[1]]
        expected_ids = [
            str(row["sample_id"])
            for seed in run.REPLICATE_SEEDS
            if seed in tuple(int(value) for value in budget["training_replicate_seeds"])
            for row in target_rows
            if row["split"] == "training" and int(row["replicate_seed"]) == seed
        ]
        assert train_ids == expected_ids


def test_ranking_metric_ties_and_material_pairs_are_deterministic():
    truth = np.asarray([0.9, 0.8, 0.7, 0.4, 0.1])
    exact = run.ranking_metrics(truth, truth)
    reverse = run.ranking_metrics(truth, -truth)

    assert exact["spearman"] == pytest.approx(1.0)
    assert exact["top1_regret_r2"] == pytest.approx(0.0)
    assert exact["top3_opportunity_regret_r2"] == pytest.approx(0.0)
    assert exact["material_pair_accuracy"] == pytest.approx(1.0)
    assert reverse["spearman"] == pytest.approx(-1.0)
    assert reverse["exact_best"] is False


def test_material_pair_threshold_includes_exactly_point_zero_one():
    truth = np.asarray([0.0, 0.01, 0.03, 0.07, 0.2], dtype=np.float64)
    expected_inclusive = sum(
        abs(float(truth[left] - truth[right])) >= run.MATERIAL_ATOL
        for left in range(len(truth))
        for right in range(left + 1, len(truth))
    )
    expected_strict = sum(
        abs(float(truth[left] - truth[right])) > run.MATERIAL_ATOL
        for left in range(len(truth))
        for right in range(left + 1, len(truth))
    )
    metrics = run.ranking_metrics(truth, truth)

    assert run.MATERIAL_ATOL == 0.01
    assert expected_inclusive > expected_strict
    assert metrics["material_pair_count"] == expected_inclusive


def test_new_budget_head_evaluation_retains_predictions_after_all_fits():
    rng = np.random.default_rng(9137)
    X_train = rng.normal(size=(72, 8)).astype(np.float32)
    y_train = (X_train[:, 0] * X_train[:, 1] + 0.2 * X_train[:, 2]).astype(np.float64)
    X_eval = rng.normal(size=(40, 8)).astype(np.float32)
    y_eval = (X_eval[:, 0] * X_eval[:, 1] + 0.2 * X_eval[:, 2]).astype(np.float64)

    heads = run._evaluate_heads_with_predictions(X_train, y_train, X_eval, y_eval, 7821)

    assert tuple(heads) == run.HEADS
    for head in run.HEADS:
        assert len(heads[head]["test_predictions"]) == len(y_eval)
        assert np.isfinite(heads[head]["test_predictions"]).all()
        assert np.isfinite(heads[head]["test_r2"])


def test_new_budget_head_evaluation_fits_all_heads_before_eval_access(monkeypatch):
    events = []
    clock = [0.0]

    class FakeSearch:
        created = 0

        def __init__(self, estimator, **_kwargs):
            self.head_position = FakeSearch.created
            FakeSearch.created += 1

        def fit(self, _X, _y):
            events.append(("fit", self.head_position))
            clock[0] += 1.0
            self.best_score_ = 0.25
            self.best_params_ = {}
            return self

        def predict(self, X):
            events.append(("predict", self.head_position))
            clock[0] += 0.1
            return np.ones(len(X), dtype=np.float64)

    class Probe:
        def __init__(self, values, label):
            self.values = np.asarray(values)
            self.label = label

        def __array__(self, dtype=None):
            events.append((self.label,))
            return np.asarray(self.values, dtype=dtype)

    monkeypatch.setattr(
        run.residual,
        "_head_estimators",
        lambda: {head: (Ridge(), {}) for head in run.HEADS},
    )
    monkeypatch.setattr(run, "GridSearchCV", FakeSearch)
    monkeypatch.setattr(run.time, "perf_counter", lambda: clock[0])
    X_train = np.ones((12, 3), dtype=np.float32)
    y_train = np.arange(12, dtype=np.float64)
    X_eval = Probe(np.ones((5, 3), dtype=np.float32), "eval_X")
    y_eval = Probe(np.arange(5, dtype=np.float64), "eval_y")

    heads = run._evaluate_heads_with_predictions(X_train, y_train, X_eval, y_eval, 7821)

    assert events[: len(run.HEADS)] == [("fit", position) for position in range(len(run.HEADS))]
    first_eval_access = next(index for index, event in enumerate(events) if event[0] == "eval_X")
    assert first_eval_access >= len(run.HEADS)
    assert all(event[0] in {"eval_X", "eval_y"} for event in events[len(run.HEADS) : first_eval_access + 2])
    assert all(event[0] == "predict" for event in events[first_eval_access + 2 :])
    assert [heads[head]["wall_seconds"] for head in run.HEADS] == pytest.approx([1.1] * len(run.HEADS))


def _synthetic_gate_inputs():
    protocol = run.load_protocol()
    archive = {
        "target_weighted_group_summaries": [
            {"group": group, "mean_spearman": 1.0} for group in run.ARCHIVE_GROUPS
        ],
        "target_level_seed_stability": [
            {"heldout_envelope_stability": {"mean_pair_spearman": 1.0}}
        ],
    }
    new_budget_values = {
        "target-a": {
            "n200_u4103_4201": 0.00,
            "n200_u4103_4303": 0.00,
            "n200_u4201_4303": 0.00,
            "n300_u4103_4201_4303": 0.20,
        },
        "target-b": {
            "n200_u4103_4201": 0.06,
            "n200_u4103_4303": 0.06,
            "n200_u4201_4303": 0.06,
            "n300_u4103_4201_4303": 0.06,
        },
    }
    budget = {
        "group_summaries": [
            {
                "budget_id": "n300_u4103_4201_4303",
                "group": "all",
                "mean_spearman": 1.0,
                "mean_top1_regret_r2": 0.06,
            }
        ],
        "metric_rows": [
            {
                "target": target,
                "budget_id": budget_id,
                "stratum": "recoverable_lp_dominated_control",
                "top1_regret_r2": values[budget_id],
            }
            for target, values in new_budget_values.items()
            for budget_id in values
        ],
        "budget_reference_comparison_summaries": {
            "heldout_envelope": [
                {
                    "from_budget_id": budget_id,
                    "group": "all",
                    "mean_spearman": 1.0,
                }
                for budget_id in (
                    "n100_s4103",
                    "n100_s4201",
                    "n100_s4303",
                    "n200_u4103_4201",
                    "n200_u4103_4303",
                    "n200_u4201_4303",
                )
            ]
        },
    }
    return protocol, archive, budget


def test_declared_budget_top1_regret_gate_is_evaluated_and_can_fail():
    protocol, archive, budget = _synthetic_gate_inputs()

    result = run._diagnostic_gates(archive, budget, protocol)

    gate = next(row for row in result["checks"] if row["id"] == "budget_new_mean_top1_regret_r2")
    assert gate["estimate"] == pytest.approx(0.055)
    assert gate["maximum"] == pytest.approx(0.05)
    assert gate["weighting"] == (
        "equal target weight after equal weighting of the three fixed n200 unions and fixed n300 union"
    )
    assert gate["status"] == "fail"
    assert result["all_point_checks_passed"] is False
    assert result["bound_thresholds"] == protocol["diagnostic_gates"]["bound_thresholds"]


def _valid_checkpoint_fixture():
    protocol = run.load_protocol()
    _raw, _summary, _manifest, cohort = run.validate_sources(protocol)
    target = str(protocol["budget_curve"]["targets"][0])
    budget = next(
        row for row in protocol["budget_curve"]["budgets"] if row["id"] == "n200_u4103_4201"
    )
    _train, _evaluation, _y_train, y_eval, train_ids, eval_ids = run._budget_rows(
        cohort, target, budget["training_replicate_seeds"]
    )
    matrix_rows = int(max(np.max(_train), np.max(_evaluation))) + 1
    dataset_id, _target_id = target.split("::", 1)
    dimensions = {
        backbone: int(run.source_extract.SPEC_BY_ID[backbone]["dimension"])
        for backbone in run.BACKBONES
    }
    source_protocol = run.source_run.load_protocol()
    dataset_position = tuple(source_protocol["datasets"]).index(dataset_id)
    target_position = tuple(source_protocol["datasets"][dataset_id]["targets"]).index(target.split("::", 1)[1])
    datasets = {
        dataset_id: {
            "matrices": {
                backbone: np.ones((matrix_rows, dimension), dtype=np.float32)
                for backbone, dimension in dimensions.items()
            }
        }
    }
    params = {
        "linear_ridge": {"model__alpha": 0.01},
        "quadratic_ridge": {"model__alpha": 0.01},
        "knn": {"model__n_neighbors": 3, "model__weights": "distance"},
        "rbf_svr": {"model__C": 1.0, "model__gamma": "scale", "model__epsilon": 0.05},
    }
    candidates = []
    for backbone_position, backbone in enumerate(run.BACKBONES):
        heads = {}
        for head_position, head in enumerate(run.HEADS):
            heads[head] = {
                "test_r2": 1.0,
                "inner_best_r2": 0.1 + 0.1 * head_position,
                "best_params": params[head],
                "wall_seconds": 0.1,
                "test_predictions": y_eval.tolist(),
            }
            heads[head]["head_record_sha256"] = run._head_record_sha256(heads[head])
        pca_seed = run._budget_geometry_seed(dataset_position, target_position, 3, backbone_position)
        pca_metadata = run.geometry.pca_geometry(
            datasets[dataset_id]["matrices"][backbone],
            _train,
            _evaluation,
            pca_seed,
        )[2]
        candidates.append(
            {
                "candidate_id": backbone,
                "pca": pca_metadata,
                "pca_seed": pca_seed,
                "fit_seed": run._budget_seed(dataset_position, target_position, 3, backbone_position),
                "pca_wall_seconds": 0.1,
                "heads": heads,
                "cv_envelope_r2": 0.4,
                "heldout_envelope_r2": 1.0,
                "best_head_ids": list(run.HEADS),
                "evaluation_predictions_available": True,
            }
        )
    dataset_id, target_id = target.split("::", 1)
    panel = {
        "dataset_id": dataset_id,
        "target_id": target_id,
        "target": target,
        "stratum": run._stratum_by_target(protocol)[target],
        "budget_id": budget["id"],
        "budget_position": 3,
        "training_replicate_seeds": list(budget["training_replicate_seeds"]),
        "training_rows": int(budget["rows"]),
        "evaluation_rows": 200,
        "training_sample_ids": train_ids,
        "evaluation_sample_ids": eval_ids,
        "evaluation_targets": y_eval.tolist(),
        "candidate_count": len(run.BACKBONES),
        "candidates": candidates,
        "evaluation_predictions_available": True,
        "source_archived": False,
        "shared_evaluation_dependence": True,
        "panel_wall_seconds": 1.0,
    }
    checkpoint = {
        "schema_version": 1,
        "status": "completed_reference_envelope_budget_checkpoint",
        "source_hashes": run.source_hashes(),
        "panel": panel,
    }
    return checkpoint, protocol, target, budget, cohort, datasets


def test_checkpoint_resume_rejects_source_and_protocol_identity_mismatch():
    checkpoint, protocol, target, budget, cohort, datasets = _valid_checkpoint_fixture()
    bad_hashes = dict(checkpoint["source_hashes"])
    bad_hashes["source_raw_results"] = "0" * 64
    bad_checkpoint = deepcopy(checkpoint)
    bad_checkpoint["source_hashes"] = bad_hashes

    with pytest.raises(ValueError, match="source identity"):
        run._validate_budget_checkpoint(
            bad_checkpoint,
            protocol,
            target,
            budget,
            checkpoint["source_hashes"],
            cohort,
            datasets,
        )

    bad_protocol = deepcopy(protocol)
    first_target = bad_protocol["budget_curve"]["targets"][0]
    bad_protocol["strata"]["recoverable_lp_dominated_control"].remove(first_target)
    bad_protocol["strata"]["rank_changing_nonlinear_opportunity"].append(first_target)
    with pytest.raises(ValueError, match="reference envelope budget checkpoint identity changed"):
        run._validate_budget_checkpoint(
            checkpoint,
            bad_protocol,
            target,
            budget,
            checkpoint["source_hashes"],
            cohort,
            datasets,
        )


@pytest.mark.parametrize(
    "tamper, message",
    [
        (lambda panel: panel["training_sample_ids"].__setitem__(0, "tampered"), "identity changed"),
        (lambda panel: panel["evaluation_targets"].__setitem__(0, panel["evaluation_targets"][0] + 1.0), "identity changed"),
        (lambda panel: panel["candidates"][0]["pca"].__setitem__("fit_rows", 201), "PCA metadata changed"),
        (lambda panel: panel["candidates"][0]["pca"].__setitem__("explained_variance_ratio_sum", 0.1), "PCA metadata changed"),
        (lambda panel: panel["candidates"][0]["heads"]["linear_ridge"]["test_predictions"].__setitem__(0, 123.456), "prediction-score consistency"),
        (lambda panel: panel["candidates"][0]["heads"]["knn"]["best_params"].__setitem__("model__n_neighbors", 99), "head parameters changed"),
        (lambda panel: panel["candidates"][0]["heads"]["linear_ridge"].__setitem__("inner_best_r2", 0.35), "head record digest changed"),
    ],
)
def test_checkpoint_resume_rejects_tampered_canonical_content(tamper, message):
    checkpoint, protocol, target, budget, cohort, datasets = _valid_checkpoint_fixture()
    tampered = deepcopy(checkpoint)
    tamper(tampered["panel"])

    with pytest.raises(ValueError, match=message):
        run._validate_budget_checkpoint(
            tampered,
            protocol,
            target,
            budget,
            checkpoint["source_hashes"],
            cohort,
            datasets,
        )


def test_protocol_is_json_serializable_and_unknown_budget_fails():
    protocol = run.load_protocol()
    json.dumps(protocol)
    with pytest.raises(ValueError, match="budget is unknown"):
        run._budget_spec(protocol, "unknown")
