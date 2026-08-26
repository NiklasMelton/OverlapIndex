"""Outcome-free tests for heteroscedastic runner orchestration."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.heteroscedastic_distance_conditioning import fixtures
from experiments.heteroscedastic_distance_conditioning import runner


def _case(index: int = 0):
    return fixtures.smoke_cases()[index]


def test_prior_regression_is_owned_by_its_dedicated_runner(tmp_path) -> None:
    assert "prior_regression" not in runner.STAGES
    with pytest.raises(ValueError, match="stage must be one of"):
        runner.validate_stage("prior_regression")
    with pytest.raises(ValueError, match="stage must be one of"):
        runner.run_stage("prior_regression", tmp_path)


def test_candidate_table_is_exactly_the_frozen_ten_methods() -> None:
    assert runner.METHOD_IDS == (
        "A",
        "B",
        "L",
        "P25",
        "P50-SW",
        "P50-CB",
        "W50-SW",
        "W50-CB",
        "M50-SW",
        "M50-CB",
    )
    assert len(runner.CANDIDATE_SPECS) == 10
    assert runner.PROMOTABLE_CANDIDATES == (
        "P25",
        "P50-SW",
        "P50-CB",
        "W50-SW",
        "W50-CB",
        "M50-SW",
        "M50-CB",
    )


def test_a_and_b_are_direct_upstream_controls_with_fresh_nested_kwargs() -> None:
    first = runner.estimator_for(runner.CANDIDATE_BY_ID["A"], 2, 41)
    second = runner.estimator_for(runner.CANDIDATE_BY_ID["A"], 2, 41)
    assert first.__class__.__name__ == "OverlapIndex"
    assert first.prototype_refinement is False
    assert second.kmeans_kwargs is not first.kmeans_kwargs
    assert first.kmeans_kwargs == second.kmeans_kwargs
    second.kmeans_kwargs["random_state"] = 99
    assert first.kmeans_kwargs["random_state"] == 41
    refined = runner.estimator_for(runner.CANDIDATE_BY_ID["B"], 2, 41)
    assert refined.__class__.__name__ == "OverlapIndex"
    assert refined.prototype_refinement is True


def test_oi_kwargs_always_builds_a_new_nested_mapping() -> None:
    first = runner.oi_kwargs(8, 17)
    second = runner.oi_kwargs(8, 17)
    assert first == second
    assert first is not second
    assert first["kmeans_kwargs"] is not second["kmeans_kwargs"]
    first["kmeans_kwargs"]["random_state"] = 18
    assert second["kmeans_kwargs"]["random_state"] == 17


def test_stratification_encoding_is_first_observed_and_does_not_sort_labels() -> None:
    labels = np.asarray(["z", 2, "z", 2, "a"], dtype=object)
    np.testing.assert_array_equal(
        runner._first_observed_integer_encoding(labels),
        np.asarray([0, 1, 0, 1, 2]),
    )


def test_pair_sample_identity_is_shared_across_scenarios_in_a_base_cell() -> None:
    cases = fixtures.smoke_cases()
    first = cases[0]
    same_base = next(
        case
        for case in cases
        if case.scenario == "S1"
        and case.seed == first.seed
        and case.balance == first.balance
        and case.count_level == first.count_level
        and case.nuisance_dim == first.nuisance_dim
        and case.k == first.k
        and case.signal_state == first.signal_state
    )
    assert runner._pair_identity_key(first) == runner._pair_identity_key(same_base)
    assert runner._pair_indices(first, 160) == runner._pair_indices(same_base, 160)


def test_conditioner_only_fold_seam_never_constructs_upstream_oi(monkeypatch: pytest.MonkeyPatch) -> None:
    import experiments.heteroscedastic_distance_conditioning.conditioning_adapter as adapter

    calls = []

    class FakeFitted:
        transform_ = np.eye(2)

    def fake_fit(X, labels, **kwargs):
        calls.append((np.asarray(X).copy(), np.asarray(labels).copy(), kwargs))
        return FakeFitted()

    monkeypatch.setattr(adapter, "_fit_conditioner", fake_fit)
    seam = runner._ConditionerOnly("oas_diagonal", "sample_weighted_rows", 0.5)
    labels = np.asarray(["b", 1, "b", 1], dtype=object)
    seam.fit(np.arange(8, dtype=float).reshape(4, 2), labels)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][1], labels)
    assert seam.transform_.shape == (2, 2)


def test_smoke_redaction_contains_no_outcomes_or_timing() -> None:
    row = {
        "candidate_id": "A",
        "candidate_score": 0.4,
        "fit_score": 0.6,
        "reference_accuracies": {"linear": 1.0},
        "linear_probe": {"score": 1.0},
        "fit_wall_seconds": 1.0,
        "score_fixed_wall_seconds": 2.0,
        "total_wall_seconds": 3.0,
        "status": "ok",
    }
    redacted = runner._redact_smoke_row(row)
    assert redacted["outcomes_redacted"] is True
    for key in (
        "candidate_score",
        "fit_score",
        "reference_accuracies",
        "linear_probe",
        "fit_wall_seconds",
        "score_fixed_wall_seconds",
        "total_wall_seconds",
    ):
        assert key not in redacted


def test_normalized_tables_retain_axes_and_pair_truth() -> None:
    case = _case()
    checkpoint = {
        "case": case.identity(),
        "case_id": case.case_id,
        "rows": [
            {
                **case.identity(),
                "case_id": case.case_id,
                "candidate_id": "A",
                "candidate_name": "oi_unrefined_raw",
                "status": "ok",
                "truth": {
                    "truth_pair_overlap": [[0.0, 0.5], [0.5, 0.0]],
                    "truth_overlap_label": [[0, 1], [1, 0]],
                    "overlap_pairs": [[0, 1]],
                    "overlap_severity": 0.5,
                },
                "pairwise": {
                    "directional_pair_support_hits_evidence": [
                        {"source_label": 0, "target_label": 1, "support": 4, "hits": 2}
                    ]
                },
            }
        ],
    }
    tables = runner._table_rows([checkpoint])
    geometry = tables["geometry_rows.jsonl"]
    assert not geometry
    pair = tables["pair_rows.jsonl"][0]
    assert pair["seed"] == case.seed
    assert pair["scenario"] == case.scenario
    assert pair["truth_overlap"] == 0.5
    assert pair["truth_overlap_label"] == 1
    assert pair["designated_pair"] is True


def test_normalized_tables_use_frozen_case_method_and_direction_order() -> None:
    cases = fixtures.smoke_cases()[:2]
    checkpoints = []
    for case in reversed(cases):
        rows = []
        for rank, candidate_id in reversed(tuple(enumerate(runner.METHOD_IDS))):
            rows.append(
                {
                    **case.identity(),
                    "case_id": case.case_id,
                    "candidate_id": candidate_id,
                    "candidate_name": candidate_id,
                    "status": "ok",
                    "geometry": {"input_rank": rank},
                    "prototype": {"input_rank": rank},
                    "pairwise": {
                        "directional_pair_support_hits_evidence": [
                            {"source_label": 1, "target_label": 0, "support": 2, "hits": 1},
                            {"source_label": 0, "target_label": 1, "support": 3, "hits": 2},
                        ]
                    },
                    "candidate_score": float(rank),
                    "reference_accuracies": {"linear": float(rank)},
                    "linear_probe": {"score": float(rank)},
                }
            )
        checkpoints.append({"case": case.identity(), "case_id": case.case_id, "rows": rows})

    tables = runner._table_rows(checkpoints)
    expected_case_ids = [case.case_id for case in cases]
    for table_name in (
        "case_rows.jsonl",
        "geometry_rows.jsonl",
        "prototype_rows.jsonl",
        "selector_panels.jsonl",
    ):
        rows = tables[table_name]
        assert [row["case_id"] for row in rows] == [
            case_id for case_id in expected_case_ids for _ in runner.METHOD_IDS
        ]
        assert [row["candidate_id"] for row in rows] == list(runner.METHOD_IDS) * len(cases)

    pair_rows = tables["pair_rows.jsonl"]
    assert [row["case_id"] for row in pair_rows] == [
        case_id for case_id in expected_case_ids for _ in runner.METHOD_IDS for _ in range(2)
    ]
    assert [row["candidate_id"] for row in pair_rows] == [
        candidate_id for _case_id in expected_case_ids for candidate_id in runner.METHOD_IDS for _ in range(2)
    ]
    assert [(row["source_label"], row["target_label"]) for row in pair_rows[:2]] == [(0, 1), (1, 0)]


def test_normalized_table_assembly_is_byte_stable_under_input_permutation(tmp_path) -> None:
    case = _case()
    rows = []
    for candidate_id in reversed(runner.METHOD_IDS):
        rows.append(
            {
                **case.identity(),
                "case_id": case.case_id,
                "candidate_id": candidate_id,
                "status": "ok",
                "candidate_score": 0.5,
                "reference_accuracies": {"linear": 0.5},
                "linear_probe": {"score": 0.5},
            }
        )
    checkpoint = {"case": case.identity(), "case_id": case.case_id, "rows": rows}
    first = tmp_path / "first"
    second = tmp_path / "second"
    runner.assemble_tables(first, [checkpoint])
    runner.assemble_tables(second, [{**checkpoint, "rows": list(reversed(rows))}])
    for name in (
        "case_rows.jsonl",
        "geometry_rows.jsonl",
        "prototype_rows.jsonl",
        "pair_rows.jsonl",
        "selector_panels.jsonl",
        "resource_rows.jsonl",
    ):
        assert (first / "tables" / name).read_bytes() == (second / "tables" / name).read_bytes()


def test_resource_table_uses_frozen_resource_case_method_order(tmp_path) -> None:
    case = _case()
    checkpoint = {
        "case": case.identity(),
        "case_id": case.case_id,
        "rows": [
            {"candidate_id": candidate_id, "status": "ok"}
            for candidate_id in runner.METHOD_IDS
        ],
    }
    candidates = ("B", "L", *runner.PROMOTABLE_CANDIDATES)
    resource_rows = [
        {
            "resource_id": resource_id,
            "case_id": case.case_id,
            "candidate_id": candidate_id,
            "total_wall_seconds": 1.0,
        }
        for resource_id in reversed(("R0", "R1", "R2", "R3"))
        for candidate_id in reversed(candidates)
    ]
    first = runner._table_rows([checkpoint], resource_rows)["resource_rows.jsonl"]
    assert [row["resource_id"] for row in first] == [
        resource_id for resource_id in ("R0", "R1", "R2", "R3") for _ in candidates
    ]
    assert [row["candidate_id"] for row in first] == list(candidates) * 4

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    runner.assemble_tables(first_dir, [checkpoint], resource_rows)
    runner.assemble_tables(second_dir, [checkpoint], list(reversed(resource_rows)))
    assert (first_dir / "tables" / "resource_rows.jsonl").read_bytes() == (
        second_dir / "tables" / "resource_rows.jsonl"
    ).read_bytes()


def test_smoke_pair_rows_are_structural_identity_only() -> None:
    case = _case()
    checkpoint = {
        "case": case.identity(),
        "case_id": case.case_id,
        "rows": [
            {
                "candidate_id": "A",
                "status": "ok",
                "outcomes_redacted": True,
                "state_identity": {"exact_state_match": True},
                "smoke_pair_identities": [
                    {
                        "source_label": 0,
                        "target_label": 1,
                        "pair_structure": {
                            key: {"present": True, "type": "int" if key in {"support", "hits", "sparse_adj_hits"} else "float", "finite": True}
                            for key in ("support", "hits", "sparse_adj_hits", "pairwise_index", "evidence")
                        },
                    }
                ],
            }
        ],
    }
    pair = runner._table_rows([checkpoint])["pair_rows.jsonl"][0]
    allowed = set(case.identity()) | {
        "candidate_id",
        "status",
        "source_label",
        "target_label",
        "exact_state_match",
        "pair_structure",
        "outcomes_redacted",
    }
    assert set(pair) == allowed
    assert set(pair["pair_structure"]) == {
        "support",
        "hits",
        "sparse_adj_hits",
        "pairwise_index",
        "evidence",
    }
    for descriptor in pair["pair_structure"].values():
        assert set(descriptor) == {"present", "type", "finite"}
    for forbidden in ("support", "hits", "sparse_adj_hits", "pairwise_index", "evidence", "finite"):
        assert forbidden not in pair


def test_deterministic_signature_excludes_only_runtime_fields() -> None:
    first = {
        "candidate_id": "A",
        "status": "ok",
        "candidate_score": 0.5,
        "fit_state": {"state_sha256": "x"},
        "fit_wall_seconds": 1.0,
        "total_cpu_seconds": 1.1,
    }
    second = dict(first, fit_wall_seconds=99.0, total_cpu_seconds=99.0)
    assert runner.determinism_signature(first) == runner.determinism_signature(second)
    assert runner.verify_deterministic_rows([first], [second])
    second["candidate_score"] = 0.51
    with pytest.raises(AssertionError):
        runner.verify_deterministic_rows([first], [second])


def test_confirmation_structural_gates_carry_promotion_and_prior_sources() -> None:
    promotion = {
        "status": "locked",
        "locked_candidate": "P50-SW",
        "structural_gates": {"exact_parity": True},
    }
    prior = {
        "status": "pass",
        "locked_candidate": "P50-SW",
        "structural_gates": {"leakage_no_refit": True},
    }
    gates = runner._validated_structural_gates(
        "confirmation",
        promotion_decision=promotion,
        prior_regression_decision=prior,
    )
    assert set(gates["source_decisions"]) == {
        "promotion_decision",
        "prior_regression_decision",
    }
    assert gates["source_decisions"]["promotion_decision"]["status"] == "locked"
    assert gates["source_decisions"]["prior_regression_decision"]["status"] == "pass"
    assert gates["exact_parity"] == "pass"
    assert gates["leakage_no_refit"] == "pass"


def test_smoke_case_never_fits_downstream_references_or_emits_numeric_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    dataset = SimpleNamespace(
        case=case,
        train=SimpleNamespace(X=np.zeros((4, 2)), y=np.asarray([0, 1, 0, 1])),
        evaluation=SimpleNamespace(
            X=np.zeros((4, 2)),
            y=np.asarray([0, 1, 0, 1]),
        ),
    )

    def fail_reference(*_args, **_kwargs):
        raise AssertionError("smoke must not fit downstream references")

    monkeypatch.setattr(runner, "_reference_models", fail_reference)
    monkeypatch.setattr(runner, "_linear_probe_oof", fail_reference)

    def fake_candidate(spec, _case, _dataset, *_args, **kwargs):
        del kwargs
        return (
            {
                "candidate_id": spec.candidate_id,
                "candidate_score": 0.5,
                "fit_state": {"state_sha256": "same"},
                "score_fixed_state": {"state_sha256": "same"},
                "final_state": {"state_sha256": "same"},
                "geometry": {"pair_distance_spearman": 0.75},
                "pairwise": {
                    "directional_pair_support_hits_evidence": [
                        {"source_label": 0, "target_label": 1, "support": 3, "hits": 2}
                    ]
                },
                "status": "ok",
            },
            object(),
        )

    monkeypatch.setattr(runner, "_run_candidate", fake_candidate)
    result = runner._run_case(
        case,
        methods=("A",),
        expose_outcomes=False,
        include_reference_outcomes=False,
        dataset_factory=lambda _case: dataset,
    )
    assert result["outcomes_redacted"] is True
    assert result["_smoke_score_digest"]
    row = result["rows"][0]
    assert "candidate_score" not in row
    assert row["geometry"]["pair_distance_spearman"]["finite"] is True
    assert row["smoke_pair_identities"][0]["source_label"] == 0
    assert row["smoke_pair_identities"][0]["target_label"] == 1
    assert set(row["smoke_pair_identities"][0]["pair_structure"]) == {
        "support",
        "hits",
        "sparse_adj_hits",
        "pairwise_index",
        "evidence",
    }
    assert "pairwise" not in row


def test_prediction_signature_is_part_of_determinism() -> None:
    first = {
        "candidate_id": "A",
        "status": "ok",
        "prediction_signature": {"sha256": "a", "dtype": "int64", "shape": [2]},
    }
    second = dict(first, prediction_signature={"sha256": "b", "dtype": "int64", "shape": [2]})
    with pytest.raises(AssertionError, match="structural candidate rows"):
        runner.verify_deterministic_rows([first], [second])


def test_checkpoint_payload_hash_and_complete_method_block_are_required(tmp_path) -> None:
    identity = {
        "stage": "smoke",
        "case_id": "checkpoint-case",
        "methods": list(runner.METHOD_IDS),
    }
    payload = {
        "case_id": "checkpoint-case",
        "rows": [
            {
                "candidate_id": candidate_id,
                "status": "ok",
                "outcomes_redacted": True,
            }
            for candidate_id in runner.METHOD_IDS
        ],
        "outcomes_redacted": True,
    }

    runner._write_checkpoint(tmp_path, payload, identity)
    path = runner._checkpoint_path(tmp_path, "checkpoint-case")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["payload_sha256"] == runner.digest(record["payload"])
    assert runner._read_checkpoint(tmp_path, "checkpoint-case", identity) == payload

    # A payload mutation with an unchanged record hash must fail before a
    # resumed case can reach fixture generation or candidate fitting.
    record["payload"]["rows"][0]["status"] = "error"
    runner._atomic_write_bytes(path, (runner.canonical_json(record) + "\n").encode("utf-8"))
    with pytest.raises(RuntimeError, match="payload hash"):
        runner._read_checkpoint(tmp_path, "checkpoint-case", identity)

    # Recomputing the payload hash does not make an incomplete/error block
    # reusable: every canonical method must occur exactly once, in order, and
    # have status=ok.
    invalid_payloads = (
        dict(payload, rows=payload["rows"][:-1]),
        dict(
            payload,
            rows=[
                *payload["rows"][:1],
                dict(payload["rows"][1], candidate_id="A"),
                *payload["rows"][2:],
            ],
        ),
        dict(payload, rows=[dict(payload["rows"][0], status="error"), *payload["rows"][1:]]),
        dict(payload, stopped=True),
        dict(payload, outcomes_redacted=False),
    )
    for invalid in invalid_payloads:
        runner._write_checkpoint(tmp_path, invalid, identity)
        with pytest.raises(RuntimeError):
            runner._read_checkpoint(tmp_path, "checkpoint-case", identity)


def test_smoke_redaction_drops_numeric_and_arbitrary_diagnostic_strings() -> None:
    row = {
        "candidate_id": "P50-SW",
        "status": "ok",
        "geometry": {
            "pair_distance_spearman": 0.912345,
            "status": "candidate-specific outcome",
            "nested": [7.25, "numeric-looking 0.88"],
        },
        "refinement": {
            "applied_count": 99,
            "reason": "material candidate advantage",
        },
        "truth": {"pair_overlap": [[0.0, 0.5]]},
        "fit_state": {"state_sha256": "a"},
        "score_fixed_state": {"state_sha256": "a"},
        "final_state": {"state_sha256": "a"},
        "pairwise": {
            "directional_pair_support_hits_evidence": [
                {"source_label": 0, "target_label": 1, "support": 12, "hits": 11}
            ]
        },
    }
    redacted = runner._redact_smoke_row(row)
    serialized = runner.canonical_json(redacted)
    assert "0.912345" not in serialized
    assert "numeric-looking 0.88" not in serialized
    assert "material candidate advantage" not in serialized
    assert '"support":12' not in serialized
    assert redacted["state_identity"]["exact_state_match"] is True


def test_smoke_stage_writes_final_table_manifest_and_marks_limited_run_partial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    case = _case()
    provenance = {
        "code_identity_sha256": "code",
        "source_hashes": {"runner.py": "runner"},
    }
    monkeypatch.setattr(runner, "_provenance", lambda: provenance)
    monkeypatch.setattr(runner, "_authorize_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_generator_bank_hashes", lambda *_args, **_kwargs: {"bank": "hash"})
    monkeypatch.setattr(runner.manifest, "verify_protocol_hash", lambda: "protocol")

    def fake_case(*_args, **_kwargs):
        rows = [
            {
                **case.identity(),
                "case_id": case.case_id,
                "candidate_id": candidate,
                "status": "ok",
                "outcomes_redacted": True,
                "state_identity": {"exact_state_match": True},
                "smoke_pair_identities": [],
            }
            for candidate in runner.METHOD_IDS
        ]
        return {
            "case": case.identity(),
            "case_id": case.case_id,
            "rows": rows,
            "references": {},
            "linear_probe": {"status": "redacted"},
            "outcomes_redacted": True,
            "_smoke_score_digest": "scores",
        }

    monkeypatch.setattr(runner, "_run_case", fake_case)
    paths = runner.run_stage(
        "smoke",
        tmp_path,
        cases=fixtures.smoke_cases(),
        limit=1,
        run_resources=False,
    )
    manifest = __import__("json").loads(paths["manifest"].read_text())
    assert manifest["artifact_status"] == "partial"
    assert set(manifest["table_manifest"]) == {
        "case_rows.jsonl",
        "geometry_rows.jsonl",
        "prototype_rows.jsonl",
        "pair_rows.jsonl",
        "selector_panels.jsonl",
        "resource_rows.jsonl",
    }
    for name, spec in manifest["table_manifest"].items():
        path = tmp_path / "tables" / name
        assert path.is_file()
        assert spec["row_count"] == len(path.read_text().splitlines())
    raw = json.loads(paths["raw"].read_text())
    assert raw["manifest_sha256"] == hashlib.sha256(
        paths["manifest"].read_bytes()
    ).hexdigest()
    assert manifest["summary_sha256"] == hashlib.sha256(
        (tmp_path / "summary.json").read_bytes()
    ).hexdigest()
    assert raw["summary_sha256"] == manifest["summary_sha256"]


def test_completed_output_is_read_only_and_identity_mismatch_is_prewrite_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    case = _case()
    provenance = {
        "code_identity_sha256": "code",
        "source_hashes": {"runner.py": "runner"},
    }
    monkeypatch.setattr(runner, "_provenance", lambda: provenance)
    monkeypatch.setattr(runner, "_authorize_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_generator_bank_hashes", lambda *_args, **_kwargs: {"bank": "hash"})
    monkeypatch.setattr(runner.manifest, "verify_protocol_hash", lambda: "protocol")

    def fake_case(*_args, **_kwargs):
        rows = [
            {
                "candidate_id": candidate_id,
                "status": "ok",
                "outcomes_redacted": True,
                "state_identity": {"exact_state_match": True},
                "smoke_pair_identities": [],
            }
            for candidate_id in runner.METHOD_IDS
        ]
        return {
            "case": case.identity(),
            "case_id": case.case_id,
            "rows": rows,
            "outcomes_redacted": True,
            "_smoke_score_digest": "scores",
        }

    monkeypatch.setattr(runner, "_run_case", fake_case)
    paths = runner.run_stage(
        "smoke",
        tmp_path,
        cases=fixtures.smoke_cases(),
        limit=1,
        run_resources=False,
    )

    manifest_payload = json.loads(paths["manifest"].read_text())
    raw_payload = json.loads(paths["raw"].read_text())
    manifest_payload["artifact_status"] = "stopped"
    raw_payload["artifact_status"] = "stopped"
    raw_payload["manifest"]["artifact_status"] = "stopped"
    raw_payload["manifest_sha256"] = runner.sha256_bytes(
        (runner.canonical_json(raw_payload["manifest"]) + "\n").encode("utf-8")
    )
    runner._atomic_write_bytes(
        paths["manifest"], (runner.canonical_json(manifest_payload) + "\n").encode("utf-8")
    )
    runner._atomic_write_bytes(
        paths["raw"], (runner.canonical_json(raw_payload) + "\n").encode("utf-8")
    )
    stopped_before = {
        path.relative_to(tmp_path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    def fail_if_run(*_args, **_kwargs):
        raise AssertionError("same-identity completed output must be read-only")

    monkeypatch.setattr(runner, "_run_case", fail_if_run)
    returned = runner.run_stage(
        "smoke",
        tmp_path,
        cases=fixtures.smoke_cases(),
        limit=1,
        run_resources=False,
    )
    assert returned["manifest"] == paths["manifest"]
    stopped_after = {
        path.relative_to(tmp_path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert stopped_after == stopped_before

    # A frozen stop is terminal/read-only.  Convert the fixture to a valid
    # completed artifact only for the separate completed-output assertions.
    manifest_payload["artifact_status"] = "completed"
    raw_payload["artifact_status"] = "completed"
    raw_payload["manifest"]["artifact_status"] = "completed"
    raw_payload["manifest_sha256"] = runner.sha256_bytes(
        (runner.canonical_json(raw_payload["manifest"]) + "\n").encode("utf-8")
    )
    runner._atomic_write_bytes(
        paths["manifest"], (runner.canonical_json(manifest_payload) + "\n").encode("utf-8")
    )
    runner._atomic_write_bytes(
        paths["raw"], (runner.canonical_json(raw_payload) + "\n").encode("utf-8")
    )
    before = {
        path.relative_to(tmp_path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    table_path = tmp_path / "tables" / "case_rows.jsonl"
    table_bytes = table_path.read_bytes()
    runner._atomic_write_bytes(table_path, table_bytes + b"{}\n")
    tampered_before = {
        path.relative_to(tmp_path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    with pytest.raises(RuntimeError, match="table hash"):
        runner.run_stage(
            "smoke",
            tmp_path,
            cases=fixtures.smoke_cases(),
            limit=1,
            run_resources=False,
        )
    tampered_after = {
        path.relative_to(tmp_path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert tampered_after == tampered_before
    runner._atomic_write_bytes(table_path, table_bytes)

    mismatched = json.loads(paths["manifest"].read_text())
    mismatched["protocol_sha256"] = "different-protocol"
    runner._atomic_write_bytes(
        paths["manifest"], (runner.canonical_json(mismatched) + "\n").encode("utf-8")
    )
    mismatch_before = {
        path.relative_to(tmp_path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    with pytest.raises(RuntimeError, match="existing manifest identity mismatch"):
        runner.run_stage(
            "smoke",
            tmp_path,
            cases=fixtures.smoke_cases(),
            limit=1,
            run_resources=False,
        )
    mismatch_after = {
        path.relative_to(tmp_path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert mismatch_after == mismatch_before
