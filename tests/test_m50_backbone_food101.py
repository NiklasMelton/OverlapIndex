"""Focused Food-101 runner tests; no archived outcome run is performed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.m50_backbone_ranking import food101
from experiments.m50_backbone_ranking import manifest
from experiments.m50_backbone_ranking import candidates


def test_stratification_encoding_preserves_original_scalar_values() -> None:
    labels = np.asarray([1, "one", 1, "one", 3, 3], dtype=object)
    encoded, classes = food101._stratification_encoding(labels)
    assert encoded.dtype == np.int64
    assert encoded.tolist() == [0, 1, 0, 1, 2, 2]
    assert classes == (1, "one", 3)
    first = food101._stratified_folds(
        np.repeat(labels, 5), n_splits=5, seed=17
    )
    second = food101._stratified_folds(
        np.repeat(labels, 5), n_splits=5, seed=17
    )
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(first, second))


def test_oi_kwargs_are_exact_archived_recipe_and_deep_copied() -> None:
    first = food101._oi_kwargs({"cat": 2, "dog": 2}, 41)
    second = food101._oi_kwargs({"cat": 2, "dog": 2}, 41)
    assert first == {
        "model_type": "MiniBatchKMeans",
        "kmeans_k": {"cat": 2, "dog": 2},
        "kmeans_kwargs": {"random_state": 41},
    }
    assert first is not second and first["kmeans_kwargs"] is not second["kmeans_kwargs"]
    first["kmeans_kwargs"]["random_state"] = 99
    assert second["kmeans_kwargs"]["random_state"] == 41


def test_full_selection_rejects_subsets_and_smoke_is_declared() -> None:
    with pytest.raises(ValueError, match="refuses subsets"):
        food101.validate_run_selection(
            food101.MODELS[:-1], food101.REPLICATES, food101.BUDGETS,
            tuple(name for name, _, _ in food101.ARMS), food101.METHODS,
            smoke=False,
        )
    assert food101.validate_run_selection(
        food101.SMOKE_MODELS,
        food101.SMOKE_REPLICATES,
        food101.SMOKE_BUDGETS,
        food101.SMOKE_ARMS,
        food101.METHODS,
        smoke=True,
    )[0] == food101.SMOKE_MODELS


def test_first_determinism_panels_follow_frozen_schedule() -> None:
    panels = tuple(
        manifest.FoodPanel(model, replicate, arm, budget)
        for model in ("m0", "m1")
        for replicate in (0,)
        for arm in ("baseline",)
        for budget in (64,)
    )
    methods = food101.METHODS
    schedule = manifest.planned_execution_order(panels, methods)
    first = food101._first_determinism_panel_ids(panels, schedule, methods)
    assert set(first) == set(methods) | {"LP-CAPPED-2048"}
    assert set(first.values()) == {panels[0].panel_id}


def test_determinism_checkpoint_identity_cannot_move_to_later_panel() -> None:
    panels = (
        manifest.FoodPanel("m0", 0, "baseline", 64),
        manifest.FoodPanel("m0", 0, "baseline", 68),
    )
    methods = food101.METHODS
    schedule = manifest.planned_execution_order(panels, methods)
    first = food101._first_determinism_panel_ids(panels, schedule, methods)
    record = food101._determinism_record(
        {
            "exact": True,
            "first_signature_sha256": "a" * 64,
            "second_signature_sha256": "a" * 64,
            "runtime_fields_excluded": True,
        },
        candidate_id="A",
        panel=panels[0],
    )
    food101._validate_determinism_records(
        {"A": record},
        expected_ids=("A",),
        expected_panel_ids={"A": first["A"]},
        panels=panels,
    )
    moved = dict(record)
    moved["panel_identity"] = {
        **record["panel_identity"],
        "budget": 68,
    }
    with pytest.raises(RuntimeError, match="panel identity"):
        food101._validate_determinism_records(
            {"A": moved},
            expected_ids=("A",),
            expected_panel_ids={"A": first["A"]},
            panels=panels,
        )


def test_food_environment_uses_canonical_manifest_surface() -> None:
    provenance = {"experiment_commit": "e" * 40, "git_dirty": True}
    observed = food101._environment(provenance)
    expected = manifest.environment(provenance)
    assert observed["packages"] == expected["packages"]
    assert observed["project"] == {"name": "overlapindex", "version": "0.1.4"}
    assert observed["thread_environment"] == {
        name: food101.os.environ.get(name) for name in food101._THREAD_ENVIRONMENT
    }


def test_fold_recipe_identity_is_resolved_and_hash_bound() -> None:
    oi_kwargs = food101._oi_kwargs({"a": 2, "b": 2}, 43)
    oi_identity = candidates.candidate_config_identity("M1-SW", oi_kwargs)
    oi_hash = candidates.candidate_config_sha256("M1-SW", oi_kwargs)
    assert oi_identity["overlap_index_kwargs"] == oi_kwargs
    assert len(oi_hash) == 64
    changed = dict(oi_kwargs)
    changed["kmeans_kwargs"] = {"random_state": 44}
    assert candidates.candidate_config_sha256("M1-SW", changed) != oi_hash

    probe_identity = food101._probe_config_identity(
        "LP-CAPPED-2048", fold=2, seed=43, cap_rows=food101.CAP_ROWS
    )
    assert probe_identity["selection"]["cap_rows"] == food101.CAP_ROWS
    assert probe_identity["logistic_regression"]["random_state"] == 43
    assert food101._config_sha256(probe_identity) != food101._config_sha256(
        {**probe_identity, "fold": 3}
    )


def test_smoke_surface_redacts_outcomes_and_timing_values() -> None:
    row = {
        "model": "m",
        "replicate": 0,
        "arm": "baseline",
        "budget": 64,
        "candidate_id": "M1-SW",
        "score": 0.7,
        "total_wall_seconds": 1.2,
        "prototype_refinement_enabled": True,
        "prototype_refinement": {"applied_count": 2},
        "conditioning_diagnostics": {"condition_after": 3.0},
        "status": "ok",
        "folds": [{"score": 0.7}],
    }
    redacted = food101._smoke_redact_row(row)
    assert redacted["score"] == {"present": True, "type": "float", "finite": True}
    assert redacted["prototype_refinement"]["type"] == "mapping"
    encoded = json.dumps(redacted)
    assert "0.7" not in encoded and "1.2" not in encoded and "applied_count" not in encoded


def test_lp_full_parity_uses_archived_score_only_not_timing() -> None:
    prior = {
        "selector_rows": [
            {
                "backbone": "dinov2-small",
                "replicate": 0,
                "arm": "baseline",
                "budget": 64,
                "method": "linear_probe_oof",
                "score": 0.75,
                "wall_seconds": 999.0,
            }
        ]
    }
    row = {
        "candidate_id": "LP-FULL",
        "backbone": "dinov2-small",
        "replicate": 0,
        "arm": "baseline",
        "budget": 64,
        "score": 0.75,
        "total_wall_seconds": 0.001,
    }
    parity = food101._probe_parity_rows([row], prior)
    assert len(parity) == 1 and parity[0]["exact"] is True
    assert parity[0]["candidate_id"] == "LP-FULL"


def test_baseline_parity_emits_exactly_one_row_per_archived_control() -> None:
    prior = {
        "selector_rows": [
            {
                "backbone": "dinov2-small",
                "replicate": 0,
                "arm": "baseline",
                "budget": 64,
                "method": "overlap_unrefined_cross_fitted",
                "score": 0.70,
            },
            {
                "backbone": "dinov2-small",
                "replicate": 0,
                "arm": "baseline",
                "budget": 64,
                "method": "overlap_refined_cross_fitted",
                "score": 0.80,
            },
        ]
    }
    rows = [
        {
            "candidate_id": "A",
            "model": "dinov2-small",
            "replicate": 0,
            "arm": "baseline",
            "budget": 64,
            "score": 0.70,
        },
        {
            "candidate_id": "B",
            "model": "dinov2-small",
            "replicate": 0,
            "arm": "baseline",
            "budget": 64,
            "score": 0.80,
        },
    ]
    parity = food101._baseline_parity_rows(rows, prior)
    assert len(parity) == 2
    assert [row["candidate_id"] for row in parity] == ["A", "B"]
    assert all(row["exact"] is True for row in parity)
    assert len({
        tuple(row[field] for field in ("model", "replicate", "arm", "budget", "candidate_id"))
        for row in parity
    }) == 2


def test_capped_probe_determinism_signature_excludes_timing() -> None:
    first = {
        "score": 0.8,
        "n_rows": 2048,
        "wall_seconds": 4.0,
        "cpu_seconds": 3.0,
        "folds": [{"fold": 0, "score": 0.8, "fit_wall_seconds": 1.0}],
    }
    second = {
        **first,
        "wall_seconds": 5.0,
        "cpu_seconds": 2.0,
        "folds": [{"fold": 0, "score": 0.8, "fit_wall_seconds": 9.0}],
    }
    assert food101._capped_determinism_comparison(first, second)["exact"] is True
    second["score"] = 0.7
    assert food101._capped_determinism_comparison(first, second)["exact"] is False


def test_cache_manifest_hash_mismatch_fails_before_matrix_load(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    path.write_text(
        json.dumps({
            "model": "dinov2-small",
            "sample_ids_sha256": food101.SAMPLE_IDS_SHA256,
            "path": "missing.npy",
            "sha256": "0" * 64,
            "shape": [28480, 2],
        }),
        encoding="utf-8",
    )
    with pytest.raises((ValueError, FileNotFoundError), match="matrix|cache"):
        food101.validate_cache_manifest(
            path,
            model="dinov2-small",
            expected_sample_ids_sha256=food101.SAMPLE_IDS_SHA256,
        )


def test_nonempty_output_without_manifest_is_not_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "food"
    output.mkdir()
    (output / "orphan.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-empty"):
        food101.run_food101(
            output=output,
            models=food101.SMOKE_MODELS,
            replicates=food101.SMOKE_REPLICATES,
            budgets=food101.SMOKE_BUDGETS,
            arms=food101.SMOKE_ARMS,
            methods=food101.METHODS,
            smoke=True,
        )


def test_final_bundle_interruption_leaves_running_manifest_and_is_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "food"
    output.mkdir()
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "models": food101.SMOKE_MODELS,
            "replicates": food101.SMOKE_REPLICATES,
            "budgets": food101.SMOKE_BUDGETS,
            "arms": food101.SMOKE_ARMS,
            "methods": food101.METHODS,
            "smoke": False,
        },
    )()
    provenance = {
        "experiment_commit": "e" * 40,
        "git_dirty": True,
        "starting_commit": manifest.EXPECTED_STARTING_COMMIT,
        "original_develop_base": manifest.EXPECTED_ORIGINAL_DEVELOP_BASE,
        "code_identity_sha256": "c" * 64,
    }
    source_hashes = {"source": (source, "a" * 64)}
    running = food101._running_manifest(
        args=args,
        provenance=provenance,
        protocol_hash="p" * 64,
        source_hashes=source_hashes,
    )
    food101._atomic_write_json(output / "manifest.json", running)
    real_write_json = food101._atomic_write_json

    def fail_terminal_manifest(path: Path, payload: object) -> None:
        if path.name == "manifest.json" and path.parent.name == food101._FINALIZATION_DIRECTORY:
            raise OSError("simulated interruption during final publication")
        real_write_json(path, payload)  # type: ignore[arg-type]

    monkeypatch.setattr(food101, "_atomic_write_json", fail_terminal_manifest)
    with pytest.raises(OSError, match="simulated interruption"):
        food101._write_bundle(
            output,
            artifact_status="stopped",
            args=args,
            provenance=provenance,
            source_hashes=source_hashes,
            cache_identity={},
            rows=(),
            reference_rows=(),
            parity_rows=(),
            probe_rows=(),
            capped_probe_rows=(),
            deterministic={},
            stop_reason="simulated",
            stop_error="interrupted",
        )
    assert food101._read_json(output / "manifest.json")["artifact_status"] == "running"
    assert (output / food101._FINALIZATION_DIRECTORY).is_dir()
    assert not (output / "raw_results.json").exists()

    monkeypatch.setattr(food101, "_atomic_write_json", real_write_json)
    food101._clear_transient_bundle_surfaces(output)
    assert not (output / food101._FINALIZATION_DIRECTORY).exists()
    assert food101._validate_running_manifest(running, running) is None

    changed = json.loads(json.dumps(running))
    changed["configuration"]["budgets"] = [68]
    with pytest.raises(RuntimeError, match="configuration"):
        food101._validate_running_manifest(changed, running)

    changed_environment = json.loads(json.dumps(running))
    changed_environment["environment"]["thread_environment"]["OMP_NUM_THREADS"] = "2"
    with pytest.raises(RuntimeError, match="environment"):
        food101._validate_running_manifest(changed_environment, running)


def test_food_run_fails_closed_when_thread_controls_are_not_preconfigured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    with pytest.raises(RuntimeError, match="one-thread environment"):
        food101.run_food101(
            output=tmp_path / "food",
            models=food101.SMOKE_MODELS,
            replicates=food101.SMOKE_REPLICATES,
            budgets=food101.SMOKE_BUDGETS,
            arms=food101.SMOKE_ARMS,
            methods=food101.METHODS,
            smoke=True,
        )


def test_checkpoint_requires_complete_probe_capped_and_determinism_surfaces(
    tmp_path: Path,
) -> None:
    methods = food101.METHODS
    panel = {"model": "m", "replicate": 0, "arm": "baseline", "budget": 64}
    schedule = [
        {"method_id": method, "execution_position": position}
        for position, method in enumerate(methods)
    ]
    identity = {"panel": panel, "schedule": schedule}

    def row(candidate_id: str) -> dict[str, object]:
        folds = []
        for fold in range(food101.FOLDS):
            if candidate_id in food101.OI_METHODS:
                kwargs = food101._oi_kwargs({"a": 1, "b": 1}, 42 + fold)
                identity = candidates.candidate_config_identity(candidate_id, kwargs)
                digest = candidates.candidate_config_sha256(candidate_id, kwargs)
            else:
                identity = food101._probe_config_identity(
                    candidate_id,
                    fold=fold,
                    seed=42,
                    cap_rows=food101.CAP_ROWS if candidate_id == "LP-CAPPED-2048" else None,
                )
                digest = food101._config_sha256(identity)
            folds.append(
                {
                    "fold": fold,
                    "split_seed": 42,
                    "candidate_config_identity": identity,
                    "candidate_config_sha256": digest,
                }
            )
        return {
            "model": "m",
            "replicate": 0,
            "arm": "baseline",
            "budget": 64,
            "candidate_id": candidate_id,
            "prototype_refinement_enabled": candidate_id in {"B", "M1-SW", "M1-CB"},
            "execution_position": methods.index(candidate_id) if candidate_id in methods else None,
            "execution_order": list(methods),
            "status": "ok",
            "score": 0.5,
            "fit_wall_seconds": 0.1,
            "fit_cpu_seconds": 0.1,
            "score_fixed_wall_seconds": 0.1,
            "score_fixed_cpu_seconds": 0.1,
            "total_wall_seconds": 0.2,
            "total_cpu_seconds": 0.2,
            "folds": folds,
        }

    def determinism_record(candidate_id: str) -> dict[str, object]:
        return {
            "panel_identity": {
                "model": "m",
                "replicate": 0,
                "arm": "baseline",
                "budget": 64,
                "candidate_id": candidate_id,
            },
            "exact": True,
            "first_signature_sha256": "a" * 64,
            "second_signature_sha256": "a" * 64,
            "runtime_fields_excluded": True,
        }

    payload = {
        "identity": identity,
        "artifact_status": "completed",
        "rows": [row(method) for method in methods],
        "probe_rows": [
            {
                **row("LP-FULL"),
                "prototype_refinement_enabled": False,
            }
        ],
        "capped_probe_rows": [
            {
                **row("LP-CAPPED-2048"),
                "prototype_refinement_enabled": False,
            }
        ],
        "determinism_verification": {
            **{method: determinism_record(method) for method in methods},
            "LP-CAPPED-2048": determinism_record("LP-CAPPED-2048"),
        },
    }
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(food101.canonical_json(payload), encoding="utf-8")
    assert food101._validate_checkpoint(checkpoint, identity, methods)["artifact_status"] == "completed"
    invalid = dict(payload)
    invalid["determinism_verification"] = {
        **{method: True for method in methods},
        "LP-CAPPED-2048": True,
    }
    checkpoint.write_text(food101.canonical_json(invalid), encoding="utf-8")
    with pytest.raises(RuntimeError, match="determinism"):
        food101._validate_checkpoint(checkpoint, identity, methods)
