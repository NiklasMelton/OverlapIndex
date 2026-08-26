"""Focused tests for the private Food-101 conditioned replay plumbing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.nuisance_conditioned_distance import food101


def _patch_food_run_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    candidate_ids: str,
    prior_rows: list[dict[str, object]] | None = None,
) -> tuple[object, Path]:
    """Install tiny deterministic stand-ins for the frozen Food inputs."""

    output = tmp_path / "food101"
    driver_path = tmp_path / "driver.py"
    result_path = tmp_path / "source_result.json"
    cohort_path = tmp_path / "source_cohort.json"
    prior_path = tmp_path / "prior.json"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    driver_path.write_text("# test driver\n", encoding="utf-8")
    result_path.write_text(
        json.dumps({"artifact_status": "completed", "reference_rows": []}),
        encoding="utf-8",
    )
    sample_ids = [f"split/{index:04d}/class-{index % 2}" for index in range(20)]
    cohort_path.write_text(
        json.dumps(
            {
                "configuration_hash": food101.SOURCE_CONFIGURATION_HASH,
                "extracted_sample_ids": sample_ids,
                "roles": {"0": {"selector": list(range(20))}},
            }
        ),
        encoding="utf-8",
    )
    prior_path.write_text(
        json.dumps({"selector_rows": prior_rows or []}), encoding="utf-8"
    )

    args = food101._parser().parse_args(
        [
            "--output",
            str(output),
            "--driver",
            str(driver_path),
            "--source-result",
            str(result_path),
            "--source-cohort",
            str(cohort_path),
            "--prior-replay",
            str(prior_path),
            "--cache-dir",
            str(cache_dir),
            "--models",
            food101.MODELS[0],
            "--replicates",
            "0",
            "--budgets",
            "64",
            "--arms",
            "nuisance_full",
            "--candidates",
            candidate_ids,
            "--smoke",
        ]
    )

    class _Driver:
        @staticmethod
        def _paired_split_banks(raw: np.ndarray, target: np.ndarray, *, seed: int) -> dict[str, object]:
            del target, seed
            return {
                "donor": np.arange(raw.shape[0]),
                "mode": np.arange(raw.shape[0]),
                "nuisance": np.arange(raw.shape[0]),
            }

        @staticmethod
        def _nested_stratified_indices(
            target: np.ndarray, budgets: tuple[int, ...], *, seed: int
        ) -> dict[int, np.ndarray]:
            del seed
            return {int(budget): np.arange(min(int(budget), target.size)) for budget in budgets}

        @staticmethod
        def _bridge_transform(
            raw: np.ndarray, *, donor: object, mode: object, nuisance: object,
            q: float, lam: float, nu: float,
        ) -> np.ndarray:
            del donor, mode, nuisance, q, lam, nu
            return raw

    matrix = np.arange(80, dtype=np.float32).reshape(20, 4)
    monkeypatch.setattr(food101, "_repository_provenance", lambda: {
        "branch": food101.EXPECTED_BRANCH,
        "starting_commit": food101.STARTING_COMMIT,
        "develop_commit_at_execution": food101.STARTING_COMMIT,
        "merge_base_with_develop": food101.STARTING_COMMIT,
        "first_protocol_commit": food101.FIRST_PROTOCOL_COMMIT,
        "first_protocol_parent": food101.STARTING_COMMIT,
        "protocol_commit": food101.STARTING_COMMIT,
        "experiment_commit": "test-commit",
        "git_dirty": True,
        "git_status_porcelain": ["?? test"],
        "experiment_source_sha256": {},
        "code_identity_sha256": "test-code-identity",
    })
    monkeypatch.setattr(food101, "_validate_sources", lambda result, cohort: None)
    monkeypatch.setattr(food101, "_load_driver", lambda path: _Driver())
    monkeypatch.setattr(
        food101,
        "_load_cache",
        lambda cache_dir, model, expected_sample_hash, *, verify_hash: (
            matrix,
            {
                "model": model,
                "path": "matrix.npy",
                "sha256": "matrix-sha",
                "shape": list(matrix.shape),
                "sample_ids_sha256": expected_sample_hash,
            },
        ),
    )

    hashes = {
        driver_path.resolve(): food101.DRIVER_SHA256,
        result_path.resolve(): food101.RESULT_SHA256,
        cohort_path.resolve(): food101.COHORT_SHA256,
        prior_path.resolve(): food101.PRIOR_SHA256,
    }
    monkeypatch.setattr(
        food101,
        "_sha256",
        lambda path: hashes.get(Path(path).resolve(), "protocol-sha"),
    )
    return args, output


def test_oi_kwargs_are_fresh_nested_mappings_with_paired_seed() -> None:
    first = food101._oi_kwargs({0: 2, 1: 2}, 41)
    second = food101._oi_kwargs({0: 2, 1: 2}, 42)

    assert first is not second
    assert first["kmeans_kwargs"] is not second["kmeans_kwargs"]
    assert first["kmeans_k"] is not second["kmeans_k"]
    assert first["kmeans_kwargs"]["random_state"] == 41
    assert second["kmeans_kwargs"]["random_state"] == 42
    first["kmeans_kwargs"]["random_state"] = -1
    assert second["kmeans_kwargs"]["random_state"] == 42


def test_stratified_cap_is_deterministic_and_balanced() -> None:
    labels = np.repeat(np.arange(4), 100)
    first = food101._stratified_cap(labels, 203, seed=17)
    second = food101._stratified_cap(labels, 203, seed=17)

    np.testing.assert_array_equal(first, second)
    assert len(first) == 203
    counts = np.unique(labels[first], return_counts=True)[1]
    assert int(np.max(counts) - np.min(counts)) <= 1


def test_repository_code_identity_ignores_generated_output(tmp_path) -> None:
    before = food101._repository_provenance()["code_identity_sha256"]
    output = tmp_path / "results" / "food101"
    output.mkdir(parents=True)
    (output / "raw_results.json").write_text('{"generated": true}\n', encoding="utf-8")
    (output / "checkpoint.json").write_text('{"resume": true}\n', encoding="utf-8")
    after = food101._repository_provenance()["code_identity_sha256"]
    assert after == before


def test_guardrail_trigger_is_panel_wide_and_uses_capped_probe() -> None:
    selector_rows = []
    probe_rows = []
    for model, b_score, e_score, refinement_rate, probe_score in (
        ("model-a", 0.90, 0.88, 0.10, 0.72),
        ("model-b", 0.80, 0.73, 0.20, 0.81),
    ):
        for candidate, score in (("B", b_score), ("C", e_score + 0.01), ("E", e_score)):
            selector_rows.append(
                {
                    "status": "ok",
                    "candidate_id": candidate,
                    "model": model,
                    "replicate": 0,
                    "arm": "nuisance_full",
                    "budget": 64,
                    "score": score,
                    "refinement": {"applied_rate": refinement_rate},
                    "outer_wall_seconds": 1.0,
                    "outer_cpu_seconds": 0.5,
                }
            )
        probe_rows.append(
            {
                "model": model,
                "replicate": 0,
                "arm": "nuisance_full",
                "budget": 64,
                "score": probe_score,
                "wall_seconds": 2.0,
                "cpu_seconds": 1.0,
            }
        )

    rows = food101._guardrail_policy_rows(
        selector_rows, probe_rows, promoted_candidate="C"
    )
    assert len(rows) == 2
    assert all(row["panel_triggered"] is True for row in rows)
    assert all(row["selected_score_source"] == "capped_linear_probe" for row in rows)
    assert {row["score"] for row in rows} == {0.72, 0.81}
    assert all(row["policy_wall_seconds"] == 4.0 for row in rows)


def test_guardrail_untriggered_panel_uses_promoted_oi() -> None:
    selector_rows = [
        {
            "status": "ok",
            "candidate_id": candidate,
            "model": "model-a",
            "replicate": 1,
            "arm": "baseline",
            "budget": 80,
            "score": score,
            "refinement": {"applied_rate": 0.10},
            "outer_wall_seconds": 1.0,
            "outer_cpu_seconds": 0.5,
        }
        for candidate, score in (("B", 0.90), ("D", 0.89), ("E", 0.88))
    ]
    probe_rows = [
        {
            "model": "model-a",
            "replicate": 1,
            "arm": "baseline",
            "budget": 80,
            "score": 0.75,
            "wall_seconds": 2.0,
            "cpu_seconds": 1.0,
        }
    ]
    rows = food101._guardrail_policy_rows(
        selector_rows, probe_rows, promoted_candidate="D"
    )
    assert rows[0]["panel_triggered"] is False
    assert rows[0]["selected_score_source"] == "D"
    assert rows[0]["score"] == 0.89
    assert rows[0]["policy_wall_seconds"] == 3.0


def test_candidate_refinement_is_strict_bool() -> None:
    generator = np.random.default_rng(2)
    matrix = np.vstack(
        [generator.normal(-1.0, 0.1, size=(15, 3)), generator.normal(1.0, 0.1, size=(15, 3))]
    ).astype(np.float32)
    labels = np.repeat([0, 1], 15)
    bad = ("C", "bad", "global_isotropy", np.bool_(True), False)

    with pytest.raises(TypeError, match="strict Python bool"):
        food101._cross_fitted_score(matrix, labels, candidate=bad, seed=3)


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([0] * 10 + [1] * 10, dtype=object),
        np.asarray(["class-b"] * 10 + ["class-a"] * 10, dtype=object),
    ],
)
def test_object_labels_use_deterministic_integer_stratification_only(
    labels: np.ndarray,
) -> None:
    encoded, classes = food101._stratification_encoding(labels)
    assert encoded.dtype == np.int64
    assert classes == (labels[0], labels[10])

    first = food101._stratified_folds(labels, n_splits=5, seed=19)
    second = food101._stratified_folds(labels.copy(), n_splits=5, seed=19)
    assert len(first) == len(second) == 5
    for (train_a, holdout_a), (train_b, holdout_b) in zip(first, second):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(holdout_a, holdout_b)
        assert set(encoded[holdout_a].tolist()) == {0, 1}


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([0] * 20 + [1] * 20, dtype=object),
        np.asarray(["first"] * 20 + ["second"] * 20, dtype=object),
    ],
)
def test_direct_cross_fit_accepts_archived_object_label_forms(
    labels: np.ndarray,
) -> None:
    generator = np.random.default_rng(13)
    matrix = np.vstack(
        [
            generator.normal(-1.0, 0.2, size=(20, 4)),
            generator.normal(1.0, 0.2, size=(20, 4)),
        ]
    ).astype(np.float32)

    first = food101._cross_fitted_score(
        matrix, labels, candidate=food101.CANDIDATES[0], seed=23
    )
    second = food101._cross_fitted_score(
        matrix, labels.copy(), candidate=food101.CANDIDATES[0], seed=23
    )
    assert first["score"] == second["score"]
    assert [fold["seed"] for fold in first["folds"]] == [23, 24, 25, 26, 27]


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([0] * 20 + [1] * 20, dtype=object),
        np.asarray(["first"] * 20 + ["second"] * 20, dtype=object),
    ],
)
def test_capped_probe_accepts_object_label_forms(labels: np.ndarray) -> None:
    generator = np.random.default_rng(31)
    matrix = np.vstack(
        [
            generator.normal(-1.0, 0.25, size=(20, 4)),
            generator.normal(1.0, 0.25, size=(20, 4)),
        ]
    ).astype(np.float32)
    first = food101._capped_probe(matrix, labels, seed=37)
    second = food101._capped_probe(matrix, labels.copy(), seed=37)
    assert first["score"] == second["score"]
    assert first["n_rows"] == second["n_rows"] == 40


def test_direct_a_cross_fit_is_deterministic() -> None:
    generator = np.random.default_rng(3)
    matrix = np.vstack(
        [generator.normal(-1.0, 0.2, size=(20, 4)), generator.normal(1.0, 0.2, size=(20, 4))]
    ).astype(np.float32)
    labels = np.repeat([0, 1], 20)
    candidate = food101.CANDIDATES[0]

    first = food101._cross_fitted_score(matrix, labels, candidate=candidate, seed=7)
    second = food101._cross_fitted_score(matrix, labels, candidate=candidate, seed=7)

    assert first["score"] == second["score"]
    assert first["refinement"] == second["refinement"]
    assert first["candidate_id"] == "A"
    assert first["prototype_refinement"] is False


def test_food_determinism_signature_excludes_clocks_but_not_state() -> None:
    base = {
        "candidate_id": "C",
        "candidate_name": "conditioned",
        "mode": "global_isotropy",
        "prototype_refinement": True,
        "score": 0.8,
        "refinement": {"applied_count": 2},
        "conditioning": [{"state_sha256": "abc"}],
        "fit_wall_seconds": 1.0,
        "folds": [
            {
                "fold": 0,
                "seed": 42,
                "train_size": 8,
                "holdout_size": 2,
                "score": 0.75,
                "fit_wall_seconds": 0.5,
                "score_fixed_cpu_seconds": 0.1,
                "refinement": {"applied_count": 1},
                "conditioning": {"state_sha256": "fold"},
                "conditioning_runtime": {
                    "conditioning_fit_wall_seconds": 0.2
                },
            }
        ],
    }
    repeated = {
        **base,
        "fit_wall_seconds": 9.0,
        "folds": [{**base["folds"][0], "fit_wall_seconds": 8.0}],
    }
    comparison = food101._cross_fit_determinism_comparison(base, repeated)
    assert comparison["exact"] is True
    assert comparison["runtime_fields_excluded"] is True

    changed = {**repeated, "score": 0.7}
    assert food101._cross_fit_determinism_comparison(base, changed)["exact"] is False


def test_candidate_failure_emits_top_level_stopped_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, output = _patch_food_run_dependencies(
        monkeypatch, tmp_path, candidate_ids="A,B"
    )
    calls = {"A": 0}

    def failing_score(
        matrix: np.ndarray,
        labels: np.ndarray,
        *,
        candidate: tuple[str, str, str, bool, bool],
        seed: int,
    ) -> dict[str, object]:
        del matrix, labels, seed
        candidate_id, name, mode, refined, _direct = candidate
        calls[candidate_id] = calls.get(candidate_id, 0) + 1
        if candidate_id == "A" and calls[candidate_id] == 2:
            raise ValueError("synthetic candidate failure")
        return {
            "candidate_id": candidate_id,
            "candidate_name": name,
            "mode": mode,
            "prototype_refinement": refined,
            "score": 0.5,
            "refinement": {},
            "conditioning": [],
            "folds": [],
        }

    monkeypatch.setattr(food101, "_cross_fitted_score", failing_score)
    with pytest.raises(RuntimeError, match="candidate cell failed"):
        food101._run(args)

    raw = json.loads((output / "raw_results.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert raw["artifact_status"] == "stopped"
    assert raw["stop_reason"] == "candidate_failure"
    assert "synthetic candidate failure" in raw["stop_error"]
    assert raw["selector_rows"] and raw["selector_rows"][0]["status"] == "error"
    assert raw["repository_provenance"]["code_identity_sha256"] == "test-code-identity"
    assert raw["configuration"]["candidates"] == ["A", "B"]
    assert manifest["artifact_status"] == "stopped"
    assert manifest["stop_reason"] == "candidate_failure"
    for name in (
        "selector_rows.csv",
        "capped_probe_rows.csv",
        "guardrail_rows.csv",
        "baseline_parity_rows.csv",
    ):
        assert (output / name).is_file()

    calls_before_resume = dict(calls)
    args.resume = True
    with pytest.raises(RuntimeError, match="resume.*checkpoint"):
        food101._run(args)
    assert calls == calls_before_resume
    resumed = json.loads((output / "raw_results.json").read_text(encoding="utf-8"))
    assert resumed["artifact_status"] == "stopped"
    assert resumed["stop_reason"] == "resume_stopped_checkpoint"


def test_baseline_parity_failure_emits_top_level_stopped_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior_rows = [
        {
            "backbone": food101.MODELS[0],
            "replicate": 0,
            "arm": "nuisance_full",
            "budget": 64,
            "method": "overlap_unrefined_cross_fitted",
            "score": 0.4,
        },
        {
            "backbone": food101.MODELS[0],
            "replicate": 0,
            "arm": "nuisance_full",
            "budget": 64,
            "method": "overlap_refined_cross_fitted",
            "score": 0.6,
        },
    ]
    args, output = _patch_food_run_dependencies(
        monkeypatch, tmp_path, candidate_ids="A,B", prior_rows=prior_rows
    )

    def scoring(
        matrix: np.ndarray,
        labels: np.ndarray,
        *,
        candidate: tuple[str, str, str, bool, bool],
        seed: int,
    ) -> dict[str, object]:
        del matrix, labels, seed
        candidate_id, name, mode, refined, _direct = candidate
        return {
            "candidate_id": candidate_id,
            "candidate_name": name,
            "mode": mode,
            "prototype_refinement": refined,
            "score": {"A": 0.5, "B": 0.6}[candidate_id],
            "refinement": {},
            "conditioning": [],
            "folds": [],
        }

    monkeypatch.setattr(food101, "_cross_fitted_score", scoring)
    with pytest.raises(RuntimeError, match="parity mismatch"):
        food101._run(args)

    raw = json.loads((output / "raw_results.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert raw["artifact_status"] == "stopped"
    assert raw["stop_reason"] == "baseline_parity_failure"
    assert "Frozen A/B parity mismatch" in raw["stop_error"]
    assert len(raw["selector_rows"]) == 2
    assert len(raw["baseline_parity_rows"]) == 2
    assert raw["baseline_parity"]["exact"] is False
    assert raw["capped_probe_rows"] == []
    assert manifest["artifact_status"] == "stopped"
    assert manifest["stop_reason"] == "baseline_parity_failure"

    calls_before_resume = {"count": 0}

    def should_not_run(
        matrix: np.ndarray,
        labels: np.ndarray,
        *,
        candidate: tuple[str, str, str, bool, bool],
        seed: int,
    ) -> dict[str, object]:
        del matrix, labels, candidate, seed
        calls_before_resume["count"] += 1
        raise AssertionError("resume must not perform later candidate work")

    monkeypatch.setattr(food101, "_cross_fitted_score", should_not_run)
    args.resume = True
    with pytest.raises(RuntimeError, match="resume.*checkpoint"):
        food101._run(args)
    assert calls_before_resume["count"] == 0
    resumed = json.loads((output / "raw_results.json").read_text(encoding="utf-8"))
    assert resumed["artifact_status"] == "stopped"
    assert resumed["stop_reason"] == "resume_stopped_checkpoint"


def test_resume_rejects_checkpoint_local_partial_determinism_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior_rows = [
        {
            "backbone": food101.MODELS[0],
            "replicate": 0,
            "arm": "nuisance_full",
            "budget": 64,
            "method": "overlap_unrefined_cross_fitted",
            "score": 0.5,
        },
        {
            "backbone": food101.MODELS[0],
            "replicate": 0,
            "arm": "nuisance_full",
            "budget": 64,
            "method": "overlap_refined_cross_fitted",
            "score": 0.5,
        },
    ]
    args, output = _patch_food_run_dependencies(
        monkeypatch, tmp_path, candidate_ids="A,B", prior_rows=prior_rows
    )
    calls = {"count": 0}

    def scoring(
        matrix: np.ndarray,
        labels: np.ndarray,
        *,
        candidate: tuple[str, str, str, bool, bool],
        seed: int,
    ) -> dict[str, object]:
        del matrix, labels, seed
        calls["count"] += 1
        candidate_id, name, mode, refined, _direct = candidate
        return {
            "candidate_id": candidate_id,
            "candidate_name": name,
            "mode": mode,
            "prototype_refinement": refined,
            "score": 0.5,
            "refinement": {},
            "conditioning": [],
            "folds": [],
        }

    monkeypatch.setattr(food101, "_cross_fitted_score", scoring)
    food101._run(args)
    assert calls["count"] == 4  # two candidates, each warm-up plus measurement

    first_checkpoint = next((output / "checkpoints").glob("*.json"))
    first_payload = json.loads(first_checkpoint.read_text(encoding="utf-8"))
    second_model = food101.MODELS[1]
    second_checkpoint = first_checkpoint.with_name(
        first_checkpoint.name.replace(food101.MODELS[0], second_model, 1)
    )
    second_payload = {
        **first_payload,
        "identity": {
            **first_payload["identity"],
            "model": second_model,
        },
        # This is a structurally completed cell whose local determinism
        # record is intentionally incomplete.  The first checkpoint's full
        # record must not mask the omission when resuming the second cell.
        "determinism_verification": {
            "A": first_payload["determinism_verification"]["A"]
        },
        "rows": [
            {**row, "model": second_model, "backbone": second_model}
            for row in first_payload["rows"]
        ],
        "probe_rows": [
            {**row, "model": second_model, "backbone": second_model}
            for row in first_payload["probe_rows"]
        ],
    }
    second_checkpoint.write_text(
        json.dumps(second_payload), encoding="utf-8"
    )

    def should_not_run(
        matrix: np.ndarray,
        labels: np.ndarray,
        *,
        candidate: tuple[str, str, str, bool, bool],
        seed: int,
    ) -> dict[str, object]:
        del matrix, labels, candidate, seed
        raise AssertionError("resume must stop before later candidate work")

    monkeypatch.setattr(food101, "_cross_fitted_score", should_not_run)
    args.models = second_model
    args.resume = True
    with pytest.raises(RuntimeError, match="incomplete_determinism"):
        food101._run(args)
    assert calls["count"] == 4
    resumed = json.loads((output / "raw_results.json").read_text(encoding="utf-8"))
    assert resumed["artifact_status"] == "stopped"
    assert resumed["stop_reason"] == "resume_checkpoint_incomplete_determinism"
