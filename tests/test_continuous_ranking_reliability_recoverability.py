import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from experiments.continuous_ranking_reliability_recoverability import acquire, prepare, run


def _records(training=1500, evaluation=400):
    rows = []
    for split, count in (("training", training), ("evaluation", evaluation)):
        for position in range(count):
            rows.append(
                {
                    "sample_id": f"{split}/{position:05d}",
                    "source_index": position,
                    "source_split": split,
                    "targets": {"target": float(position) / max(1, count - 1)},
                }
            )
    return rows


def test_protocol_sidecar_and_contract_match():
    payload = prepare.PROTOCOL_PATH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == prepare.PROTOCOL_SHA256_PATH.read_text().strip()
    protocol = run.load_protocol()
    assert protocol["scope"]["overlapindex_evaluated"] is False
    assert protocol["scope"]["reserved_backbones_evaluated"] is False
    assert protocol["recoverability"]["truth"].startswith("fixed 600-label ranking target")
    assert protocol["ranking_target"]["training_row_overrides"] == {
        "lsp_2000_pose_geometry::wrist_span_over_torso_length": 590
    }
    assert protocol["expected_grid"]["recoverability_target_budget_count"] == 18
    assert acquire._protocol()["cohort"]["training_rows_per_replicate"] == 200


def test_nested_selection_is_balanced_deterministic_and_disjoint():
    first = prepare.select_nested_target("dataset", "target", _records(), 1200)
    second = prepare.select_nested_target("dataset", "target", list(reversed(_records())), 1200)
    assert first == second
    training = [row for row in first if row["split"] == "training"]
    evaluation = [row for row in first if row["split"] == "evaluation"]
    assert len(training) == 1200
    assert len(evaluation) == 200
    assert not {row["sample_id"] for row in training} & {row["sample_id"] for row in evaluation}
    for budget in (300, 600, 1200):
        prefix = training[:budget]
        assert [sum(row["target_bin"] == target_bin for row in prefix) for target_bin in range(10)] == [budget // 10] * 10


def test_nested_selection_fails_closed_on_shortfall():
    with pytest.raises(ValueError, match="required"):
        prepare.select_nested_target("dataset", "target", _records(training=599), 600)


def test_prepared_cohort_accepts_canonical_json_dataset_order(monkeypatch, tmp_path):
    datasets = {
        dataset_id: {
            "rows": [],
            "target_cohorts": {
                target_id: [
                    *[
                        {
                            "sample_id": f"{target_id}/training/{position}",
                            "split": "training",
                            "budget_rank": position,
                        }
                        for position in range(prepare.MAX_TRAIN_BY_TARGET[f"{dataset_id}::{target_id}"])
                    ],
                    *[
                        {
                            "sample_id": f"{target_id}/evaluation/{position}",
                            "split": "evaluation",
                            "budget_rank": None,
                        }
                        for position in range(prepare.EVALUATION_ROWS)
                    ],
                ]
                for target_id in prepare.TARGETS_BY_DATASET[dataset_id]
            },
        }
        for dataset_id in reversed(prepare.DATASETS)
    }
    cohort = {
        "protocol_sha256": prepare.sha256_path(prepare.PROTOCOL_PATH),
        "datasets": datasets,
    }
    cohort_path = tmp_path / "cohort.json"
    manifest_path = tmp_path / "manifest.json"
    prepare._atomic_json(cohort_path, cohort)
    prepare._atomic_json(
        manifest_path,
        {
            "status": "completed_continuous_ranking_reliability_recoverability_preparation",
            "protocol_sha256": prepare.sha256_path(prepare.PROTOCOL_PATH),
            "cohort_sha256": prepare.sha256_path(cohort_path),
            "cohort_identity_sha256": prepare.sha256_payload(cohort),
            "overlapindex_fit_count": 0,
            "reserved_backbone_count": 0,
        },
    )
    assert set(prepare.load_cohort(tmp_path)["datasets"]) == set(prepare.DATASETS)


def test_ranking_metrics_preserve_regret_and_material_pairs():
    truth = [0.50, 0.40, 0.30, 0.20, 0.10]
    perfect = run.ranking_metrics(truth, truth)
    reversed_metrics = run.ranking_metrics(truth, list(reversed(truth)))
    assert perfect["spearman"] == pytest.approx(1.0)
    assert perfect["top1_regret_r2"] == 0.0
    assert perfect["material_pair_accuracy"] == 1.0
    assert reversed_metrics["spearman"] == pytest.approx(-1.0)
    assert reversed_metrics["top1_regret_r2"] == pytest.approx(0.4)
    assert reversed_metrics["material_pair_accuracy"] == 0.0


def test_ranking_metrics_use_null_for_constant_score():
    metrics = run.ranking_metrics([0.5, 0.4, 0.3, 0.2, 0.1], [1.0] * 5)
    assert metrics["spearman"] is None
    assert metrics["material_pair_accuracy"] == 0.5


def test_evaluate_heads_retains_predictions_only_when_requested(monkeypatch):
    monkeypatch.setattr(
        run,
        "_head_estimators",
        lambda: {head: (Ridge(), {"alpha": [1.0]}) for head in run.HEADS},
    )
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 4))
    y = X[:, 0] + 0.1 * rng.normal(size=60)
    retained = run.evaluate_heads(X[:40], y[:40], X[40:], y[40:], 11, retain_predictions=True)
    omitted = run.evaluate_heads(X[:40], y[:40], X[40:], y[40:], 11, retain_predictions=False)
    assert all(len(retained[head]["prediction"]) == 20 for head in run.HEADS)
    assert all("prediction" not in omitted[head] for head in run.HEADS)
    assert {head: retained[head]["prediction_sha256"] for head in run.HEADS} == {
        head: omitted[head]["prediction_sha256"] for head in run.HEADS
    }


def test_nested_candidate_uses_common_outer_rows_and_honest_oof(monkeypatch):
    monkeypatch.setattr(
        run,
        "_head_estimators",
        lambda: {head: (Ridge(), {"alpha": [1.0]}) for head in run.HEADS},
    )

    def identity_geometry(matrix, train, holdout, seed):
        return np.asarray(matrix[train]), np.asarray(matrix[holdout]), {
            "rank": int(matrix.shape[1]), "input_dimension": int(matrix.shape[1]),
            "fit_rows": len(train), "seed": seed,
        }

    monkeypatch.setattr(run.geometry, "pca_geometry", identity_geometry)
    rng = np.random.default_rng(13)
    matrix = rng.normal(size=(60, 5)).astype(np.float32)
    target = matrix[:, 0] - matrix[:, 1]
    rows = [
        {"sample_id": f"row/{position}", "matrix_index": position, "target": float(target[position])}
        for position in range(60)
    ]
    result = run.nested_candidate_scores(matrix, rows, "dataset::target", run.BACKBONES[0], 60)
    assert len(result["folds"]) == 3
    assert set(result["scores"]) == set(run.METHODS)
    assert result["scores"]["four_head_nested_cv_envelope"] >= result["scores"]["linear_nested_cv"]
    assert all(set(fold["heads"]) == set(run.HEADS) for fold in result["folds"])


def test_synthetic_panel_is_deterministic_and_has_five_aligned_candidates():
    first = run.synthetic_panel("quadratic", 17)
    second = run.synthetic_panel("quadratic", 17)
    assert np.array_equal(first["y"], second["y"])
    assert len(first["candidates"]) == len(run.BACKBONES)
    assert all(candidate["matrix"].shape == (3200, 32) for candidate in first["candidates"])
    assert all(
        np.array_equal(left["matrix"], right["matrix"])
        for left, right in zip(first["candidates"], second["candidates"])
    )
    with pytest.raises(ValueError, match="unknown synthetic family"):
        run.synthetic_panel("unknown", 17)


def test_reference_uncertainty_reports_stable_well_separated_ranking(monkeypatch):
    monkeypatch.setattr(run, "UNCERTAINTY_RESAMPLES", 100)
    y = np.linspace(-1.0, 1.0, 200)
    candidates = []
    for candidate_position, backbone in enumerate(run.BACKBONES):
        noise = 0.03 + 0.08 * candidate_position
        repeats = []
        scores = []
        for repeat in range(3):
            prediction = y + noise * np.sin(np.arange(200) + repeat)
            score = float(1.0 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2))
            scores.append(score)
            repeats.append({"heads": {head: {"prediction": prediction.tolist()} for head in run.HEADS}})
        candidates.append(
            {
                "candidate_id": backbone,
                "repeats": repeats,
                "envelope_r2": float(np.mean(scores)),
            }
        )
    protocol = json.loads(prepare.PROTOCOL_PATH.read_text())
    panel = {"target_key": "synthetic::stable", "evaluation_target": y.tolist(), "candidates": candidates}
    result = run.reference_uncertainty(panel, protocol)
    assert result["median_bootstrap_rank_spearman"] >= 0.9
    assert result["point_top1_probability"] >= 0.9
    assert result["stable"] is True


def test_new_experiment_does_not_modify_public_overlapindex_sources():
    package_files = sorted(Path(prepare.PACKAGE_DIR).glob("*.py"))
    assert {path.name for path in package_files} == {"__init__.py", "acquire.py", "extract.py", "prepare.py", "run.py"}
