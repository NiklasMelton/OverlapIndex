"""Focused runtime contract tests; the 750-panel/4500-row benchmark is never run here."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.m50_backbone_ranking import food101, manifest, runtime_benchmark as runtime


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.canonical_json(payload) + "\n", encoding="utf-8")


def _provisional_artifact(root: Path) -> Path:
    protocol_hash = "p" * 64
    code_hash = "c" * 64
    decision = {
        "stage": "development_provisional",
        "status": "provisional_runtime_pending",
        "protocol_sha256": protocol_hash,
        "code_identity_sha256": code_hash,
        "selected_candidate": "M1-SW",
        "selected_chain": None,
        "runtime_candidate_ids": list(runtime.RUNTIME_PRIMITIVE_METHODS),
        "runtime_pending": True,
        "runner_up_allowed": False,
    }
    decision_path = root / "development_lock.json"
    _write_json(decision_path, decision)
    decision_hash = manifest.sha256_path(decision_path)
    _write_json(
        root / "analysis_manifest.json",
        {
            "schema_version": 1,
            "stage": "development_provisional",
            "protocol_sha256": protocol_hash,
            "code_identity_sha256": code_hash,
            "development_lock_sha256": decision_hash,
            "files": {"development_lock.json": {"sha256": decision_hash}},
        },
    )
    return root


def test_lp_full_uses_all_rows_while_capped_probe_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []

    def fake_score(
        values: np.ndarray,
        labels: np.ndarray,
        seed: int,
        *,
        candidate_id: str,
        cap_rows: int | None,
    ) -> dict[str, object]:
        del labels, seed, candidate_id, cap_rows
        seen.append(values.shape[0])
        return {"score": 0.5, "fit_wall_seconds": 1.0, "fit_cpu_seconds": 1.0, "score_fixed_wall_seconds": 1.0, "score_fixed_cpu_seconds": 1.0, "folds": []}

    monkeypatch.setattr(runtime.food101, "_lp_fold_score", fake_score)
    values = np.ones((food101.CAP_ROWS + 1, 2), dtype=np.float32)
    labels = np.asarray([0, 1] * ((food101.CAP_ROWS + 1) // 2 + 1), dtype=object)[: values.shape[0]]
    runtime._direct_probe(values, labels, 17, capped=False)
    runtime._direct_probe(values, labels, 17, capped=True)
    assert seen == [food101.CAP_ROWS + 1, food101.CAP_ROWS]


def test_runtime_method_ids_are_locked_and_derived_methods_are_not_estimators(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="derived"):
        runtime.measure_runtime(
            np.ones((20, 2), dtype=np.float32),
            np.asarray([0, 1] * 10),
            method_id="F",
            seed=1,
            k_per_class={0: 1, 1: 1},
            warmup=False,
        )
    decision = _provisional_artifact(tmp_path / "analysis")
    assert manifest.validate_provisional_runtime_decision(
        decision, protocol_hash="p" * 64, code_identity_hash="c" * 64
    )["selected_candidate"] == "M1-SW"
    assert runtime._validate_methods(runtime.RUNTIME_PRIMITIVE_METHODS) == runtime.RUNTIME_PRIMITIVE_METHODS
    with pytest.raises(ValueError, match="exactly match"):
        runtime._validate_methods(("B", "M1-SW", "LP-FULL"))


def test_runtime_crossfit_fake_estimator_has_separate_fold_fit_and_score() -> None:
    values = np.arange(40, dtype=np.float32).reshape(20, 2)
    labels = np.asarray([0] * 10 + [1] * 10, dtype=object)
    calls: list[tuple[str, int, int]] = []

    class Fake:
        prototype_refinement_ = {}
        conditioning_diagnostics_ = {}
        runtime_diagnostics_ = {
            "conditioning_fit_wall_seconds": 0.01,
            "conditioning_fit_cpu_seconds": 0.01,
        }

        def fit(self, train: np.ndarray, target: np.ndarray) -> "Fake":
            calls.append(("fit", train.shape[0], target.shape[0]))
            return self

        def score_fixed(self, holdout: np.ndarray, target: np.ndarray) -> float:
            calls.append(("score", holdout.shape[0], target.shape[0]))
            return 0.5

    result = runtime.measure_runtime(
        values,
        labels,
        method_id="B",
        seed=11,
        k_per_class={0: 1, 1: 1},
        estimator_factory=lambda method, seed, k: Fake(),
        warmup=False,
    )
    assert result["candidate_id"] == "B"
    assert result["warmup_excluded"] is True
    assert [kind for kind, _, _ in calls] == ["fit", "score"] * 5
    assert all(train != holdout for (_, train, _), (_, holdout, _) in zip(calls[::2], calls[1::2]))


def test_runtime_oi_normalizes_once_before_both_fit_and_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.arange(40, dtype=np.float32).reshape(20, 2) + 1.0
    labels = np.asarray([0] * 10 + [1] * 10, dtype=object)
    normalization_calls: list[np.ndarray] = []
    fit_values: list[np.ndarray] = []
    score_values: list[np.ndarray] = []
    original_row_l2 = food101._row_l2

    def spy_row_l2(matrix: np.ndarray) -> np.ndarray:
        result = original_row_l2(matrix)
        normalization_calls.append(np.asarray(matrix).copy())
        return result

    monkeypatch.setattr(runtime.food101, "_row_l2", spy_row_l2)

    class Fake:
        prototype_refinement_ = {}
        conditioning_diagnostics_ = {}
        runtime_diagnostics_ = {}

        def fit(self, train: np.ndarray, target: np.ndarray) -> "Fake":
            del target
            fit_values.append(np.asarray(train).copy())
            return self

        def score_fixed(self, holdout: np.ndarray, target: np.ndarray) -> float:
            del target
            score_values.append(np.asarray(holdout).copy())
            return 0.5

    result = runtime.measure_runtime(
        values,
        labels,
        method_id="B",
        seed=11,
        k_per_class={0: 1, 1: 1},
        estimator_factory=lambda method, seed, k: Fake(),
        warmup=True,
    )
    assert result["normalization_in_clock"] == "outer_crossfit_before_fold_clocks"
    assert len(normalization_calls) == 2
    expected = original_row_l2(values)
    folds = food101._stratified_folds(labels, n_splits=food101.FOLDS, seed=11)
    assert all(
        np.allclose(observed, expected[train])
        for observed, (train, _holdout) in zip(fit_values, folds * 2)
    )
    assert all(
        np.allclose(observed, expected[holdout])
        for observed, (_train, holdout) in zip(score_values, folds * 2)
    )


def _runtime_identity() -> dict[str, object]:
    return runtime._runtime_identity(
        code_identity_sha256="c" * 64,
        protocol_sha256="p" * 64,
        provisional_decision_sha256="d" * 64,
        provisional_analysis_manifest_sha256="a" * 64,
        methods=runtime.RUNTIME_PRIMITIVE_METHODS,
    )


def _runtime_fold_recipes(candidate_id: str, seed: int) -> list[dict[str, object]]:
    folds: list[dict[str, object]] = []
    for fold in range(food101.FOLDS):
        if candidate_id in food101.OI_METHODS:
            kwargs = food101._oi_kwargs({0: 1, 1: 1}, seed + fold)
            identity = runtime.candidates.candidate_config_identity(candidate_id, kwargs)
            digest = runtime.candidates.candidate_config_sha256(candidate_id, kwargs)
        else:
            identity = food101._probe_config_identity(
                candidate_id,
                fold=fold,
                seed=seed,
                cap_rows=food101.CAP_ROWS if candidate_id == "LP-CAPPED-2048" else None,
            )
            digest = food101._config_sha256(identity)
        folds.append(
            {
                "fold": fold,
                "split_seed": seed,
                "candidate_config_identity": identity,
                "candidate_config_sha256": digest,
            }
        )
    return folds


def test_runtime_checkpoint_requires_rotated_order_positions_and_call_seed() -> None:
    methods = runtime.RUNTIME_PRIMITIVE_METHODS
    model, arm, budget, repeat = manifest.runtime_panels()[0]
    schedule = tuple(
        row
        for row in manifest.runtime_execution_order(methods)
        if row["model"] == model
        and row["arm"] == arm
        and int(row["budget"]) == budget
        and int(row["repeat"]) == repeat
    )
    ordered = tuple(sorted(schedule, key=lambda row: int(row["execution_position"])))
    values = np.ones((20, 2), dtype=np.float32)
    labels = np.asarray([0] * 10 + [1] * 10, dtype=object)
    identity = runtime._runtime_block_identity(
        runtime_identity=_runtime_identity(),
        model=model,
        arm=arm,
        budget=budget,
        repeat=repeat,
        schedule=ordered,
        values=values,
        labels=labels,
    )
    order = [str(row["method_id"]) for row in ordered]
    dataset_identity = {
        "values_sha256": identity["values_sha256"],
        "labels_sha256": identity["labels_sha256"],
        "row_count": identity["row_count"],
        "feature_count": identity["feature_count"],
        "metadata": identity["dataset_metadata"],
    }
    rows = [
        {
                "candidate_id": str(row["method_id"]),
                "backbone": model,
                "arm": arm,
            "budget": budget,
            "repeat": repeat,
            "execution_position": int(row["execution_position"]),
            "execution_order": order,
            "schedule_seed": manifest.SCHEDULE_SEED,
            "call_seed": 42 + budget,
            "warmup_excluded": True,
            "status": "ok",
            "dataset_identity": dataset_identity,
            "total_wall_seconds": 0.1,
            "total_cpu_seconds": 0.1,
            "peak_memory_mb": 1.0,
            "candidate_score": 0.5,
            "folds": _runtime_fold_recipes(str(row["method_id"]), 42 + budget),
        }
        for row in ordered
    ]
    payload = {"artifact_status": "completed", "identity": identity, "rows": rows}
    assert len(runtime._validate_runtime_block(payload, identity=identity, methods=methods)) == 6
    invalid = [dict(row) for row in rows]
    invalid[1]["execution_position"] = invalid[0]["execution_position"]
    with pytest.raises(RuntimeError, match="positions"):
        runtime._validate_runtime_block(
            {"artifact_status": "completed", "identity": identity, "rows": invalid},
            identity=identity,
            methods=methods,
        )


def test_runtime_bundle_binds_raw_bytes_and_canonical_table(tmp_path: Path) -> None:
    rows = [
        {
            "candidate_id": "A",
            "backbone": "dinov2-small",
            "budget": 64,
            "repeat": 0,
            "status": "ok",
            "warmup_excluded": True,
            "total_wall_seconds": 0.1,
            "total_cpu_seconds": 0.1,
            "peak_memory_mb": 1.0,
        }
    ]
    runtime._write_runtime_bundle(
        tmp_path,
        artifact_status="stopped",
        runtime_identity=_runtime_identity(),
        rows=rows,
        methods=runtime.RUNTIME_PRIMITIVE_METHODS,
        stop_reason="test",
        stop_error="test",
    )
    raw_path = tmp_path / "raw_results.json"
    manifest_path = tmp_path / "manifest.json"
    raw_bytes = raw_path.read_bytes()
    raw = json.loads(raw_bytes)
    artifact_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert artifact_manifest["raw_results"] == {
        "sha256": manifest.sha256_path(raw_path),
        "size_bytes": len(raw_bytes),
    }
    table_bytes = (tmp_path / "runtime_rows.jsonl").read_bytes()
    assert artifact_manifest["tables"]["runtime_rows"]["row_count"] == 1
    assert artifact_manifest["tables"]["runtime_rows"]["sha256"] == manifest.sha256_bytes(table_bytes)
    assert raw["runtime_rows"] == rows


def test_runtime_repeat_determinism_excludes_only_timing_and_memory() -> None:
    rows: list[dict[str, object]] = []
    for repeat in runtime.RUNTIME_REPEATS:
        rows.append(
            {
                "candidate_id": "B",
                "backbone": "m",
                "arm": "baseline",
                "budget": 64,
                "repeat": repeat,
                "candidate_score": 0.75,
                "prototype_refinement": {"applied_count": 2},
                "conditioning_diagnostics": {"state_sha256": "a" * 64},
                "folds": [{"fold": 0, "score": 0.75, "candidate_config_sha256": "a" * 64}],
                "total_wall_seconds": float(repeat + 1),
                "total_cpu_seconds": float(repeat + 1),
                "peak_memory_mb": float(repeat + 2),
            }
        )
    runtime._validate_runtime_repeat_determinism(rows, ("B",))
    rows[-1]["candidate_score"] = 0.5
    with pytest.raises(RuntimeError, match="determinism"):
        runtime._validate_runtime_repeat_determinism(rows, ("B",))
    rows[-1]["candidate_score"] = 0.75
    rows[-1]["folds"][0]["candidate_config_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="determinism"):
        runtime._validate_runtime_repeat_determinism(rows, ("B",))


def test_food_runtime_factory_is_paired_nested_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runtime factory uses one raw max panel and evicts prior arrays."""

    class FrozenDriver:
        calls: list[tuple[str, int]] = []

        @staticmethod
        def _paired_split_banks(values: np.ndarray, labels: np.ndarray, *, seed: int) -> dict[str, np.ndarray]:
            FrozenDriver.calls.append(("banks", int(seed)))
            return {
                "donor": np.asarray(values).copy(),
                "mode": np.ones(values.shape[0], dtype=np.float32),
                "nuisance": np.asarray(values).copy(),
            }

        @staticmethod
        def _bridge_transform(
            values: np.ndarray,
            *,
            donor: np.ndarray,
            mode: np.ndarray,
            nuisance: np.ndarray,
            q: float,
            lam: float,
            nu: float,
        ) -> np.ndarray:
            del donor, mode, nuisance, q
            return np.asarray(values, dtype=np.float32) + np.float32(lam + nu)

    sample_ids = [
        f"train/{class_index:02d}/class-{class_index:02d}/row-{row_index:04d}.jpg"
        for class_index in range(runtime.RUNTIME_CLASSES)
        for row_index in range(runtime.RUNTIME_ROWS_PER_CLASS)
    ]
    cohort = {
        "extracted_sample_ids": sample_ids,
        "extracted_train_rows": runtime.RUNTIME_COHORT_ROWS,
    }

    runtime_source = tmp_path / "frozen_runtime_driver.py"
    runtime_source.write_text(
        "import numpy as np\n"
        "def build_nested_indices(labels, budgets, *, seed=42):\n"
        "    del seed\n"
        "    result = {}\n"
        "    for budget in budgets:\n"
        "        result[int(budget)] = np.sort(np.concatenate([\n"
        "            np.flatnonzero(np.asarray(labels, dtype=object) == label)[:int(budget)]\n"
        "            for label in list(dict.fromkeys(np.asarray(labels, dtype=object).tolist()))\n"
        "        ]).astype(np.int64))\n"
        "    return result\n",
        encoding="utf-8",
    )
    runtime_hash = manifest.sha256_path(runtime_source)
    monkeypatch.setattr(runtime.food101, "DEFAULT_RUNTIME_SOURCE", runtime_source)
    monkeypatch.setattr(runtime.food101, "RUNTIME_SOURCE_SHA256", runtime_hash)
    monkeypatch.setattr(
        runtime.food101,
        "_load_archived_inputs",
        lambda *args, **kwargs: (FrozenDriver, {}, cohort, {}, {}),
    )

    raw_by_model = {
        model: np.arange(runtime.RUNTIME_COHORT_ROWS * 2, dtype=np.float32).reshape(
            runtime.RUNTIME_COHORT_ROWS, 2
        )
        + np.float32(model_index * 1_000_000)
        for model_index, model in enumerate(("model-a", "model-b"))
    }

    def fake_load_cache(cache_dir: Path, model: str, expected_sample_ids_sha256: str):
        del cache_dir, expected_sample_ids_sha256
        return raw_by_model[model], {
            "model": model,
            "sha256": f"{(1 + len(model)):064x}"[-64:],
        }

    monkeypatch.setattr(runtime.food101, "_load_cache", fake_load_cache)
    factory = runtime.food_dataset_factory(tmp_path / "cache")

    baseline_640, labels_640 = factory("model-a", "baseline", 640)
    baseline_128, labels_128 = factory("model-a", "baseline", 128)
    assert baseline_640.shape == (runtime.RUNTIME_CLASSES * 640, 2)
    assert baseline_128.shape == (runtime.RUNTIME_CLASSES * 128, 2)
    assert set(np.unique(labels_640, return_counts=True)[1].tolist()) == {640}
    assert set(np.unique(labels_128, return_counts=True)[1].tolist()) == {128}
    # The small panel consists of the same max-panel rows, in the exact
    # nested-index order; compare by unique encoded row values rather than
    # assuming class blocks are contiguous after sorting.
    assert set(map(tuple, baseline_128.tolist())).issubset(
        set(map(tuple, baseline_640.tolist()))
    )

    conditioned, _ = factory("model-a", "nuisance_full", 128)
    assert conditioned.shape == baseline_128.shape
    assert not np.array_equal(conditioned, baseline_128)
    metadata = factory._m50_runtime_metadata
    assert metadata["nested_indices_sha256"]
    assert metadata["nested_index_seed"] == runtime.RUNTIME_NESTED_INDEX_SEED
    assert metadata["bridge_bank_seed"] == runtime.RUNTIME_BRIDGE_BANK_SEED
    assert {
        "cache_matrix_sha256",
        "max_panel_values_sha256",
        "max_panel_labels_sha256",
        "max_panel_index_sha256",
    }.issubset(metadata["models"]["model-a"])

    factory("model-b", "baseline", 64)
    snapshot = factory._m50_runtime_cache_snapshot()
    assert snapshot["active_key"] == ("model-b", "baseline")
    assert snapshot["active_matrix_nbytes"] == (
        runtime.RUNTIME_CLASSES
        * max(runtime.RUNTIME_BUDGETS)
        * raw_by_model["model-b"].shape[1]
        * np.dtype(np.float32).itemsize
    )
    # Metadata is intentionally retained, but only the current transformed
    # matrix is retained by the closure.
    assert snapshot["retained_model_metadata"] == ("model-a", "model-b")


def test_run_runtime_single_block_preflight_and_schedule_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real runner path with a reduced structural grid."""

    for name in runtime._THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(manifest, "runtime_panels", lambda: (("m", "baseline", 64, 0),))
    monkeypatch.setattr(manifest, "RUNTIME_ARMS", ("baseline",))
    monkeypatch.setattr(runtime, "RUNTIME_REPEATS", (0,))
    monkeypatch.setattr(manifest, "verify_protocol_hash", lambda: "p" * 64)
    monkeypatch.setattr(
        manifest,
        "code_identity_sha256",
        lambda *args, **kwargs: "c" * 64,
    )
    monkeypatch.setattr(
        manifest,
        "repository_provenance",
        lambda **kwargs: {
            "branch": "test",
            "experiment_commit": "e" * 40,
            "git_dirty": True,
            "source_hashes": {},
            "code_identity_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(manifest, "verify_protocol_source_evidence", lambda: {})
    monkeypatch.setattr(manifest, "environment", lambda provenance: {"packages": {}})
    monkeypatch.setattr(runtime.food101, "_require_thread_environment", lambda: {})

    values = np.ones((runtime.RUNTIME_CLASSES * 64, 2), dtype=np.float32)
    labels = np.repeat(np.arange(runtime.RUNTIME_CLASSES), 64).astype(object)
    digest = "a" * 64

    def factory(model: str, arm: str, budget: int):
        assert (model, arm, budget) == ("m", "baseline", 64)
        return values, labels

    factory._m50_runtime_metadata = {
        "models": {
            "m": {
                "cache_matrix_sha256": digest,
                "max_panel_values_sha256": digest,
                "max_panel_labels_sha256": digest,
                "max_panel_index_sha256": digest,
                "bridge_bank_seed": runtime.RUNTIME_BRIDGE_BANK_SEED,
                "max_panel_budget": max(runtime.RUNTIME_BUDGETS),
                "nested_index_seed": runtime.RUNTIME_NESTED_INDEX_SEED,
                "nested_indices_sha256": digest,
            }
        }
    }

    calls: list[str] = []

    def fake_measure(values, labels, *, method_id, seed, k_per_class, **kwargs):
        del values, labels, seed, k_per_class, kwargs
        calls.append(method_id)
        return {
            "candidate_id": method_id,
            "candidate_score": 0.5,
            "fit_wall_seconds": 0.1,
            "fit_cpu_seconds": 0.1,
            "conditioning_fit_wall_seconds": 0.0,
            "conditioning_fit_cpu_seconds": 0.0,
            "score_fixed_wall_seconds": 0.1,
            "score_fixed_cpu_seconds": 0.1,
            "total_wall_seconds": 0.2,
            "total_cpu_seconds": 0.2,
            "prototype_refinement": {},
                "conditioning_diagnostics": {},
                "peak_memory_mb": 1.0,
                "warmup_excluded": True,
                "folds": _runtime_fold_recipes(method_id, 42 + 64),
            }

    monkeypatch.setattr(runtime, "measure_runtime", fake_measure)
    result = runtime.run_runtime(
        output=tmp_path / "runtime",
        development_decision=_provisional_artifact(tmp_path / "analysis"),
        protocol_hash="p" * 64,
        code_identity_hash="c" * 64,
        dataset_factory=factory,
        use_fresh_process=False,
    )
    assert result["artifact_status"] == "completed"
    assert set(calls) == set(runtime.RUNTIME_PRIMITIVE_METHODS)
    assert len(calls) == len(runtime.RUNTIME_PRIMITIVE_METHODS)
    assert result["environment"]["thread_environment"] == {
        name: "1" for name in runtime._THREAD_ENVIRONMENT
    }
