from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.scalable_relevance_kmeans import speed_analysis as analysis
from experiments.scalable_relevance_kmeans import speed_experiment as experiment


def _small_panel(seed: int = 41) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.repeat(np.asarray(["a", "b", "c"], dtype=object), 15)
    signal = np.repeat(np.asarray([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.0]]), 15, axis=0)
    X = np.concatenate(
        [signal + rng.normal(scale=0.4, size=(45, 2)), rng.normal(scale=2.0, size=(45, 6))],
        axis=1,
    ).astype(np.float32)
    return X, y


def test_frozen_candidate_table_and_protocol_match() -> None:
    protocol_sha, protocol = experiment.validate_protocol()
    assert len(protocol_sha) == 64
    assert protocol["variant_specs"] == experiment.VARIANT_SPECS
    assert experiment.PROMOTION_PRECEDENCE == ("FAST", "CAP32", "NOLLOYD")
    assert experiment.DEVELOPMENT_METHODS == (
        "V1",
        "F32",
        "FAST",
        "CAP32",
        "NOLLOYD",
        "LP-FULL",
    )


def test_development_order_is_exactly_position_balanced() -> None:
    counts = {method: [0] * len(experiment.DEVELOPMENT_METHODS) for method in experiment.DEVELOPMENT_METHODS}
    for panel in range(30):
        for position, method in enumerate(experiment._ordered(experiment.DEVELOPMENT_METHODS, panel)):
            counts[method][position] += 1
    assert all(positions == [5] * 6 for positions in counts.values())


def test_candidate_factory_uses_fresh_closed_configuration() -> None:
    k = {"a": 2, "b": 2}
    first = experiment._build_selector("CAP32", k, 7)
    second = experiment._build_selector("CAP32", k, 7)
    assert first is not second
    assert first.kmeans_kwargs is not second.kmeans_kwargs
    assert first.margin_rows_per_class == 32
    assert first.fast_scoring is True
    assert first.lloyd_update is True
    assert experiment._build_selector("NOLLOYD", k, 7).lloyd_update is False


def test_cross_fitted_variant_is_deterministic_and_records_five_folds() -> None:
    X, y = _small_panel()
    first = experiment.cross_fitted_variant(X, y, candidate_id="CAP32", seed=9)
    second = experiment.cross_fitted_variant(X, y, candidate_id="CAP32", seed=9)
    assert first["score"] == second["score"]
    assert len(first["folds"]) == 5
    assert [row["candidate_config_sha256"] for row in first["folds"]] == [
        row["candidate_config_sha256"] for row in second["folds"]
    ]
    assert all(
        row["structural_diagnostics"]["input_dtype"] == "float32"
        for row in first["folds"]
    )


def test_development_gates_require_every_arm_runtime() -> None:
    candidate = "FAST"
    metrics = []
    for arm in experiment.ARMS:
        for head in analysis.HEADS:
            for method, regret in ((candidate, 0.2), ("LP-FULL", 0.1), ("V1", 0.2)):
                metrics.append(
                    {
                        "replicate": experiment.DEVELOPMENT_REPLICATE,
                        "arm": arm,
                        "budget": experiment.DEVELOPMENT_BUDGET,
                        "head": head,
                        "candidate_id": method,
                        "regret_pp": regret,
                    }
                )
    rows = []
    for model in experiment.MODELS:
        for arm in experiment.ARMS:
            rows.extend(
                [
                    {"model": model, "arm": arm, "candidate_id": candidate, "total_wall_seconds": 0.8},
                    {"model": model, "arm": arm, "candidate_id": "LP-FULL", "total_wall_seconds": 1.0},
                ]
            )
    gates = analysis._development_candidate_gates({"selector_rows": rows}, metrics, candidate)
    assert len([row for row in gates if row["gate"] == "runtime"]) == 3
    assert all(row["status"] == "pass" for row in gates)
    for row in rows:
        if row["arm"] == "nuisance_full" and row["candidate_id"] == candidate:
            row["total_wall_seconds"] = 1.4
    gates = analysis._development_candidate_gates({"selector_rows": rows}, metrics, candidate)
    assert next(
        row for row in gates if row["gate"] == "runtime" and row["arm"] == "nuisance_full"
    )["status"] == "fail"


def test_lock_validation_rejects_hash_or_source_mismatch(tmp_path: Path) -> None:
    protocol_sha, _ = experiment.validate_protocol()
    sources = experiment.source_identity()
    body = {
        "schema_version": 1,
        "status": "locked_for_full_evaluation",
        "selected_candidate": "FAST",
        "eligible_candidates": ["FAST"],
        "promotion_precedence": list(experiment.PROMOTION_PRECEDENCE),
        "runner_up_reselection_allowed": False,
        "protocol_sha256": protocol_sha,
        "source_identity": sources,
        "development_raw_sha256": "0" * 64,
        "development_manifest_sha256": "1" * 64,
        "candidate_gates": {},
        "retrospective_development_only": True,
    }
    body["lock_sha256"] = experiment.hashlib.sha256(
        experiment.canonical_json(body).encode("utf-8")
    ).hexdigest()
    path = tmp_path / "lock.json"
    path.write_text(experiment.canonical_json(body) + "\n", encoding="utf-8")
    assert experiment._validate_lock(path, protocol_sha, sources)["selected_candidate"] == "FAST"
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["selected_candidate"] = "CAP32"
    path.write_text(experiment.canonical_json(changed) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        experiment._validate_lock(path, protocol_sha, sources)


def test_output_artifacts_do_not_change_source_identity(tmp_path: Path) -> None:
    before = experiment.source_identity()
    (tmp_path / "raw_results.json").write_text("{}\n", encoding="utf-8")
    assert experiment.source_identity() == before
