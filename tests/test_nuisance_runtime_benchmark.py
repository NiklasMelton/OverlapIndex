"""Focused tests for the private large-budget runtime benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.nuisance_conditioned_distance import runtime_benchmark as runtime


def _small_panel(labels_kind: str = "int") -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(123)
    matrix = np.vstack(
        [
            generator.normal(-1.0, 0.2, size=(20, 4)),
            generator.normal(1.0, 0.2, size=(20, 4)),
        ]
    ).astype(np.float32)
    if labels_kind == "string":
        labels = np.asarray(["left"] * 20 + ["right"] * 20, dtype=object)
    else:
        labels = np.asarray([0] * 20 + [1] * 20, dtype=object)
    return matrix, labels


def test_frozen_counts_and_locked_primitive_surfaces() -> None:
    assert len(runtime.MODELS) == 10
    assert runtime.BUDGETS == (64, 128, 256, 512, 640)
    assert runtime.REPEATS == tuple(range(5))
    assert runtime.N_CLASSES == 40
    assert runtime.FOLDS == 5
    assert runtime.K == 10
    assert runtime.primitive_method_ids("E") == (
        "A",
        "B",
        "E",
        "full_probe",
        "capped_probe_component",
    )
    assert runtime.primitive_method_ids("C") == (
        "A",
        "B",
        "E",
        "C",
        "full_probe",
        "capped_probe_component",
    )
    assert runtime.primitive_method_ids("D")[3] == "D"
    assert runtime.primitive_method_ids(None, all_candidates=True) == (
        "A",
        "B",
        "C",
        "D",
        "E",
        "full_probe",
        "capped_probe_component",
    )


def test_nested_indices_are_hash_seeded_balanced_and_nested_for_strings() -> None:
    labels = np.asarray(["b"] * 20 + ["a"] * 20, dtype=object)
    first = runtime.build_nested_indices(labels, (5, 10), seed=42)
    second = runtime.build_nested_indices(labels.copy(), (5, 10), seed=42)
    for budget in (5, 10):
        np.testing.assert_array_equal(first[budget], second[budget])
        assert len(first[budget]) == 2 * budget
        assert {str(value): int(np.sum(labels[first[budget]] == value)) for value in ("a", "b")} == {
            "a": budget,
            "b": budget,
        }
    assert set(first[5]).issubset(set(first[10]))


def test_counterbalanced_order_is_deterministic_and_position_balanced() -> None:
    methods = ("A", "B", "E", "full_probe", "capped_probe_component")
    orders = [
        runtime.counterbalanced_method_order(
            methods, model_index=model, budget_index=budget, repeat=repeat
        )
        for model in range(10)
        for budget in range(5)
        for repeat in range(5)
    ]
    assert len(orders) == 250
    assert all(set(order) == set(methods) for order in orders)
    for position in range(len(methods)):
        counts = {method: sum(order[position] == method for order in orders) for method in methods}
        assert max(counts.values()) - min(counts.values()) <= 1


def test_candidate_factories_use_fresh_kwargs_and_strict_bool(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeOI:
        def __init__(self, **kwargs: object) -> None:
            calls.append(("direct", kwargs))

    class FakeConditioned:
        def __init__(self, **kwargs: object) -> None:
            calls.append(("conditioned", kwargs))

    monkeypatch.setattr(runtime, "OverlapIndex", FakeOI)
    monkeypatch.setattr(runtime, "ConditionedOverlapIndex", FakeConditioned)
    first = runtime.build_oi_candidate("A", seed=41)
    second = runtime.build_oi_candidate("A", seed=42)
    runtime.build_oi_candidate("C", seed=43)
    assert first is not second
    assert len(calls) == 3
    assert calls[0][1]["prototype_refinement"] is False
    assert calls[1][1]["prototype_refinement"] is False
    first_kwargs = calls[0][1]["kmeans_kwargs"]
    second_kwargs = calls[1][1]["kmeans_kwargs"]
    assert first_kwargs is not second_kwargs
    assert first_kwargs["random_state"] == 41
    assert second_kwargs["random_state"] == 42
    first_kwargs["random_state"] = -1
    assert second_kwargs["random_state"] == 42
    assert calls[2][0] == "conditioned"
    assert calls[2][1]["prototype_refinement"] is True

    with pytest.raises(ValueError, match="unknown OI candidate"):
        runtime.build_oi_candidate("unknown", seed=1)


@pytest.mark.parametrize("labels_kind", ["int", "string"])
def test_full_probe_uses_archived_recipe_and_scalar_label_encoding(labels_kind: str) -> None:
    matrix, labels = _small_panel(labels_kind)
    result = runtime._timed_probe(matrix, labels, seed=42, folds=5)
    assert 0.0 <= result["score"] <= 1.0
    assert result["n_rows"] == 40
    assert result["total_wall_seconds"] >= 0.0
    assert result["total_cpu_seconds"] >= 0.0
    probe = runtime.build_full_probe(42)
    assert probe.named_steps["normalizer"].norm == "l2"
    logistic = probe.named_steps["logisticregression"]
    assert logistic.C == 1.0
    assert logistic.max_iter == 2_000
    assert logistic.n_jobs == 1
    capped = runtime.build_capped_probe(42)
    assert capped.named_steps["standardscaler"].with_mean is True
    assert capped.named_steps["logisticregression"].C == 1.0
    capped_result = runtime._timed_probe(
        matrix, labels, seed=42, folds=5, maximum_rows=10
    )
    assert capped_result["n_rows"] == 10
    assert capped_result["probe_rows_capped"] is True


def test_benchmark_excludes_one_warmup_per_method(monkeypatch: pytest.MonkeyPatch) -> None:
    matrix, labels = _small_panel()
    calls: list[str] = []

    def fake_oi(matrix: np.ndarray, labels: np.ndarray, *, candidate_id: str, seed: int, folds: int) -> dict[str, object]:
        calls.append(candidate_id)
        return {
            "candidate_id": candidate_id,
            "candidate_name": candidate_id,
            "score": 0.5,
            "fit_wall_seconds": 1.0,
            "fit_cpu_seconds": 1.0,
            "score_fixed_wall_seconds": 1.0,
            "score_fixed_cpu_seconds": 1.0,
            "total_wall_seconds": 2.0,
            "total_cpu_seconds": 2.0,
            "folds": [],
        }

    def fake_probe(matrix: np.ndarray, labels: np.ndarray, *, seed: int, folds: int, maximum_rows: int | None = None) -> dict[str, object]:
        method = "capped_probe_component" if maximum_rows is not None else "full_probe"
        calls.append(method)
        return {
            "candidate_id": method,
            "candidate_name": method,
            "score": 0.5,
            "total_wall_seconds": 2.0,
            "total_cpu_seconds": 2.0,
        }

    monkeypatch.setattr(runtime, "_timed_oi", fake_oi)
    monkeypatch.setattr(runtime, "_timed_probe", fake_probe)
    result = runtime.benchmark_matrix(
        matrix,
        labels,
        budgets=(5, 10),
        repeats=(0, 1),
        method_ids=("A", "full_probe"),
    )
    assert len(result["rows"]) == 2 * 2 * 2
    assert len(result["warmup"]) == 2
    assert sorted(result["warmed_methods"]) == ["A", "full_probe"]
    assert all(row["warmup_excluded"] is True for row in result["rows"])
    assert calls.count("A") == 1 + 8 // 2
    assert calls.count("full_probe") == 1 + 8 // 2


def test_oi_timing_uses_row_l2_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    matrix, labels = _small_panel()
    fit_rows: list[np.ndarray] = []

    class FakeOI:
        def fit(self, values: np.ndarray, target: np.ndarray) -> "FakeOI":
            fit_rows.append(np.asarray(values))
            return self

        def score_fixed(self, values: np.ndarray, target: np.ndarray) -> float:
            return 0.5

    monkeypatch.setattr(runtime, "build_oi_candidate", lambda candidate_id, seed: FakeOI())
    runtime._timed_oi(matrix, labels, candidate_id="A", seed=42, folds=5)
    assert fit_rows
    for values in fit_rows:
        np.testing.assert_allclose(
            np.linalg.norm(values, axis=1), 1.0, rtol=1e-6, atol=1e-6
        )


def test_policy_g_uses_b_plus_e_plus_cap_only_when_triggered() -> None:
    def row(method: str, score: float, *, applied: int = 0, before: int = 100) -> dict[str, object]:
        return {
            "model": "toy",
            "repeat": 0,
            "budget": 64,
            "method": method,
            "status": "ok",
            "score": score,
            "total_wall_seconds": 1.0,
            "total_cpu_seconds": 2.0,
            "folds": [{"refinement": {"applied_count": applied, "prototype_count_before": before}}],
        }

    untriggered = runtime._policy_rows(
        [row("B", 0.50), row("E", 0.50), row("C", 0.50), row("capped_probe_component", 0.2)],
        promoted_candidate="C",
    )[0]
    assert untriggered["policy_wall_seconds"] == 3.0
    assert untriggered["composed_from"] == ["B", "E", "C"]

    triggered = runtime._policy_rows(
        [row("B", 0.50, applied=60), row("E", 0.50), row("C", 0.50), row("capped_probe_component", 0.2)],
        promoted_candidate="C",
    )[0]
    assert triggered["panel_triggered"] is True
    assert triggered["policy_wall_seconds"] == 3.0
    assert triggered["composed_from"] == ["B", "E", "capped_probe_component"]


def test_warmup_and_first_measured_structural_signatures_are_compared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix, labels = _small_panel()
    calls = {"A": 0}

    def fake_oi(matrix: np.ndarray, labels: np.ndarray, *, candidate_id: str, seed: int, folds: int) -> dict[str, object]:
        calls[candidate_id] = calls.get(candidate_id, 0) + 1
        return {
            "candidate_id": candidate_id,
            "candidate_name": candidate_id,
            "score": float(calls[candidate_id]),
            "fit_wall_seconds": 1.0,
            "fit_cpu_seconds": 1.0,
            "score_fixed_wall_seconds": 1.0,
            "score_fixed_cpu_seconds": 1.0,
            "total_wall_seconds": 2.0,
            "total_cpu_seconds": 2.0,
            "folds": [],
        }

    monkeypatch.setattr(runtime, "_timed_oi", fake_oi)
    with pytest.raises(RuntimeError, match="nondeterminism"):
        runtime.benchmark_matrix(
            matrix,
            labels,
            budgets=(5,),
            repeats=(0,),
            method_ids=("A",),
        )


def test_partial_artifact_stops_on_candidate_error_and_code_hash_ignores_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = runtime.code_identity_sha256()
    output = tmp_path / "runtime"
    (output / "checkpoints").mkdir(parents=True)
    (output / "raw_results.json").write_text('{"generated": true}\n', encoding="utf-8")
    after = runtime.code_identity_sha256()
    assert before == after

    monkeypatch.setattr(runtime, "repository_provenance", lambda: {"code_identity_sha256": before})

    def failing_benchmark(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("synthetic candidate failure")

    monkeypatch.setattr(runtime, "benchmark_matrix", failing_benchmark)
    matrix, labels = _small_panel()
    with pytest.raises(RuntimeError, match="synthetic candidate failure"):
        runtime.run_runtime_benchmark(
            output=output,
            smoke=True,
            models=("toy",),
            budgets=(5,),
            repeats=(0,),
            matrices={"toy": matrix},
            labels=labels,
        )
    payload = json.loads((output / "raw_results.json").read_text(encoding="utf-8"))
    assert payload["artifact_status"] == "stopped_candidate_error"
    assert (output / "manifest.json").is_file()
    assert (output / "primitive_rows.csv").is_file()
    assert (output / "policy_rows.csv").is_file()


def test_runtime_payload_never_claims_parent_process_memory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    matrix, labels = _small_panel()
    monkeypatch.setattr(runtime, "repository_provenance", lambda: {"code_identity_sha256": "test"})
    monkeypatch.setattr(
        runtime,
        "benchmark_matrix",
        lambda *args, **kwargs: {"rows": [], "warmup": [], "determinism": {}},
    )
    payload = runtime.run_runtime_benchmark(
        output=tmp_path / "runtime",
        smoke=True,
        models=("toy",),
        budgets=(5,),
        repeats=(0,),
        matrices={"toy": matrix},
        labels=labels,
    )
    assert payload["memory"]["status"] == "unavailable"
    assert "peak_rss_bytes" not in payload["memory"]
