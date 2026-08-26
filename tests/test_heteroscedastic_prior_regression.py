"""Outcome-free contract tests for the bounded prior-regression adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.heteroscedastic_distance_conditioning import prior_regression as p
from experiments.nuisance_conditioned_distance import analysis as old_analysis
from experiments.nuisance_conditioned_distance import runner as old_runner


def _decision(candidate: str = "W50-CB") -> dict[str, str]:
    return {
        "status": "locked",
        "decision_stage": "development",
        "locked_candidate": candidate,
        "protocol_sha256": p.sha256_path(p.CURRENT_PROTOCOL_PATH),
        "code_identity_sha256": p.current_code_identity(),
    }


def _evidence() -> p.PriorSourceEvidence:
    return p.verify_prior_source_evidence()


def test_source_evidence_hash_failure_is_rejected_before_old_artifact_read(tmp_path: Path) -> None:
    payload = json.loads(p.CURRENT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    for entry in payload["source_evidence"]:
        if entry["path"].endswith("screen/raw_results.json"):
            entry["sha256"] = "0" * 64
            break
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(p.PriorEvidenceError, match="frozen value"):
        p.verify_prior_source_evidence(
            current_protocol_path=protocol,
            verify_implementation_sources=False,
        )


def test_unauthorized_lock_never_reaches_source_or_dataset_factory(tmp_path: Path) -> None:
    touched: list[str] = []

    def fail_if_called() -> None:
        touched.append("called")
        raise AssertionError("unauthorized prior regression touched the panel")

    decision = {
        "status": "locked",
        "decision_stage": "development",
        "locked_candidate": "A",
        "protocol_sha256": p.sha256_path(p.CURRENT_PROTOCOL_PATH),
        "code_identity_sha256": p.current_code_identity(),
    }
    with pytest.raises(p.PriorAuthorizationError):
        p.run_prior_regression(
            tmp_path,
            decision,
            case_factory=fail_if_called,
            dataset_factory=fail_if_called,
        )
    assert touched == []
    assert not tmp_path.exists() or not (tmp_path / "manifest.json").exists()


def test_exact_prior_grid_identity_generator_and_backend_contract() -> None:
    evidence = _evidence()
    panel = p.recover_prior_panel(evidence=evidence)
    assert len(panel) == p.PRIOR_SCREEN_CASE_COUNT == 720
    assert panel.raw_metadata["row_count"] == p.PRIOR_SCREEN_METHOD_ROW_COUNT == 4_320
    assert panel.expected_methods == ("A", "B", "C", "D", "E", "F")
    assert panel.backend == "MiniBatchKMeans"
    assert panel.fit_split == "train"
    assert panel.evaluation_split == "evaluation"
    assert panel.generator_sha256 == "277af6d6a6fd8fb8d613faeb4bf0bce845145b57c6a0862de05167efd5f0efb3"
    assert panel.case_grid_sha256 == p.prior_case_grid_sha256(panel.cases)
    assert p.prior_execution_order(panel.cases[0], "W50-CB") == ("A", "B", "L", "W50-CB")
    assert p.prior_backend_kwargs(8, 17) == old_runner.oi_kwargs(8, 17)


def test_prior_regression_method_set_is_exact_and_never_ranks_runner_up() -> None:
    assert p.prior_regression_methods("P50-SW") == ("A", "B", "L", "P50-SW")
    assert p.prior_regression_methods("W50-CB") == ("A", "B", "L", "W50-CB")
    with pytest.raises(p.PriorAuthorizationError):
        p.prior_regression_methods("A")
    with pytest.raises(p.PriorAuthorizationError):
        p.prior_regression_methods("runner-up")


def test_resume_identity_includes_old_hashes_lock_and_code(tmp_path: Path) -> None:
    evidence = _evidence()
    case = old_runner.stage_cases("screen")[0]
    authorization = p.validate_development_promotion_decision(_decision())
    methods = p.prior_regression_methods(authorization)
    identity = p.checkpoint_identity(
        case,
        authorization=authorization,
        source_evidence=evidence,
        code_identity_sha256=authorization.code_identity_sha256,
        protocol_sha256=authorization.protocol_sha256,
        methods=methods,
    )
    payload = {
        "case_id": case.case_id,
        "rows": [
            {"candidate_id": method, "status": "ok"}
            for method in methods
        ],
        "status": "ok",
        "methods": list(methods),
    }
    p.write_checkpoint(tmp_path, payload, identity)
    record = json.loads((tmp_path / "checkpoints" / f"{case.case_id}.json").read_text())
    assert record["payload_sha256"] == p.sha256_bytes(p.canonical_json(record["payload"]).encode())
    assert p.read_checkpoint(tmp_path, case.case_id, identity) == payload

    # Mutating an otherwise valid payload without updating its digest is
    # rejected before any case can be resumed.
    record["payload"]["rows"][0]["status"] = "error"
    (tmp_path / "checkpoints" / f"{case.case_id}.json").write_text(
        p.canonical_json(record) + "\n"
    )
    with pytest.raises(p.PriorIdentityError, match="payload hash"):
        p.read_checkpoint(tmp_path, case.case_id, identity)

    # A fresh digest cannot bless a partial, reordered, stopped, or errored
    # block.  These are all non-reusable resume records.
    invalid_payloads = (
        dict(payload, rows=payload["rows"][:-1]),
        dict(payload, rows=[payload["rows"][1], payload["rows"][0], *payload["rows"][2:]]),
        dict(payload, rows=[dict(payload["rows"][0], status="error"), *payload["rows"][1:]]),
        dict(payload, stopped=True),
        dict(payload, error="candidate failed"),
        dict(payload, rows=[dict(payload["rows"][0], stop_reason="stopped"), *payload["rows"][1:]]),
        dict(payload, case_id="different-case"),
    )
    for invalid in invalid_payloads:
        p.write_checkpoint(tmp_path, invalid, identity)
        with pytest.raises((p.PriorIdentityError, p.PriorStructuralError)):
            p.read_checkpoint(tmp_path, case.case_id, identity)

    changed = dict(identity)
    changed["promotion_decision_sha256"] = "different-lock"
    with pytest.raises(p.PriorIdentityError, match="identity mismatch"):
        p.read_checkpoint(tmp_path, case.case_id, changed)


def test_a_and_b_use_direct_upstream_construction_and_copy_backend_kwargs() -> None:
    calls: list[tuple[str, int, int, dict[str, object]]] = []

    def factory(candidate_id: str, k: int, seed: int, kwargs: dict[str, object]):
        calls.append((candidate_id, k, seed, kwargs))
        return {
            "candidate_id": candidate_id,
            "prototype_refinement": candidate_id == "B",
            "kwargs": kwargs,
        }

    assert p.assert_direct_control_construction(k=2, seed=3, estimator_factory=factory)
    assert [item[0] for item in calls] == ["A", "B"]
    assert calls[0][3] == p.prior_backend_kwargs(2, 3)
    assert calls[0][3] is not calls[1][3]
    calls[0][3]["kmeans_kwargs"]["random_state"] = 99  # type: ignore[index]
    assert calls[1][3]["kmeans_kwargs"]["random_state"] == 3  # type: ignore[index]


def test_legacy_l_adapter_is_new_adapter_with_frozen_full_whitening_controls() -> None:
    captured: dict[str, object] = {}

    def adapter(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return dict(kwargs)

    result = p.build_prior_estimator("L", 8, 17, adapter_factory=adapter)
    assert result["estimator"] == "oas_diagonal"
    assert result["weighting"] == "sample_weighted_rows"
    assert result["gamma"] == 1.0
    assert result["prototype_refinement"] is True
    assert result["overlap_index_kwargs"] == p.prior_backend_kwargs(8, 17)


def test_prior_case_serializes_heldout_pairwise_state_and_opaque_prediction_signature() -> None:
    case = old_runner.stage_cases("screen")[0]
    dataset = {
        "train": {"X": np.zeros((4, 2)), "y": np.asarray([0, 0, 1, 1])},
        "evaluation": {"X": np.zeros((4, 2)), "y": np.asarray([0, 0, 1, 1])},
    }

    class PairwiseModel:
        def __init__(self) -> None:
            self.pairwise_index: dict[tuple[int, int], float] = {}
            self.pairwise_cardinality: dict[tuple[int, int], int] = {}
            self._pairwise_hits: dict[tuple[int, int], int] = {}
            self.sparse_adj: dict[tuple[int, int], int] = {}
            self.index = 0.25

        def _set_pairwise(self, value: float, hits: int) -> None:
            self.pairwise_index = {(0, 1): value, (1, 0): value}
            self.pairwise_cardinality = {(0, 1): 10, (1, 0): 10}
            self._pairwise_hits = {(0, 1): hits, (1, 0): hits}
            self.sparse_adj = {(0, 1): hits, (1, 0): hits}

        def fit(self, _X: np.ndarray, _y: np.ndarray) -> "PairwiseModel":
            # This is deliberately different from score_fixed's held-out
            # state.  The regression must serialize the latter.
            self._set_pairwise(0.1, 9)
            return self

        def score_fixed(self, _X: np.ndarray, _y: np.ndarray) -> float:
            self._set_pairwise(0.7, 3)
            return 0.3

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.zeros(X.shape[0], dtype=np.int64)

    block = p.run_prior_case(
        case,
        "W50-CB",
        dataset_factory=lambda _case: dataset,
        estimator_factory=lambda *_args, **_kwargs: PairwiseModel(),
    )
    assert [row["candidate_id"] for row in block["rows"]] == ["A", "B", "L", "W50-CB"]
    for row in block["rows"]:
        assert row["pairwise"]["legacy"]["0->1"]["pairwise_index"] == 0.7
        assert row["pairwise"]["legacy"]["0->1"]["overlap_evidence"] == pytest.approx(0.3)
        assert row["prediction_count"] == 4
        assert set(row["prediction_signature"]) == {"sha256", "dtype", "shape"}
        assert "predictions" not in row


def test_promotion_decision_hash_matches_canonical_mapping_and_file_bytes(tmp_path: Path) -> None:
    decision = _decision("W50-CB")
    path = tmp_path / "promotion_decision.json"
    path.write_text(p.canonical_json(decision) + "\n", encoding="utf-8")
    assert p.decision_sha256(decision) == p.decision_sha256(path)
    assert p.validate_development_promotion_decision(path).promotion_decision_sha256 == p.decision_sha256(decision)


@pytest.mark.parametrize("gate_status", ["pass", "fail", "inconclusive"])
def test_production_historical_gate_surface_maps_exact_prior_panel_without_old_outcomes(
    gate_status: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the versioned old evaluator path on exactly four new arms.

    The fake archived evaluator is deliberately installed at the old analysis
    boundary.  This keeps the test outcome-free while proving that production
    mapping supplies all 720 newly generated case identities, maps L/E exactly,
    and does not let the old artifact-completeness marker veto gate evaluation.
    """

    panel = p.recover_prior_panel(evidence=_evidence())
    locked = "W50-CB"
    methods = p.prior_regression_methods(locked)
    rows: list[dict[str, object]] = []
    for case in panel.cases:
        identity = p.prior_case_identity(case)
        for method in methods:
            rows.append(
                {
                    **identity,
                    "case_id": case.case_id,
                    "candidate_id": method,
                    "status": "ok",
                }
            )

    captured: dict[str, object] = {}

    def fake_summary(records: list[dict[str, object]], **kwargs: object) -> dict[str, object]:
        captured["records"] = records
        captured["summary_kwargs"] = kwargs
        # This deliberately says the old six-arm artifact is incomplete: the
        # prior regression has only A/B/L/locked, and the gate call must still
        # evaluate the complete *new* panel rather than reject this mapping.
        return {"artifact_completeness": {"status": "fail"}, "candidates": {}}

    def fake_gates(
        summary: dict[str, object],
        *,
        protocol: dict[str, object],
        candidate_ids: tuple[str, ...],
    ) -> dict[str, object]:
        captured["gate_summary"] = summary
        captured["gate_candidate_ids"] = candidate_ids
        captured["gate_protocol"] = protocol
        return {"E": {"status": gate_status, "gates": [{"status": gate_status}]}}

    monkeypatch.setattr(old_analysis, "build_analysis_summary", fake_summary)
    monkeypatch.setattr(old_analysis, "evaluate_historical_gates", fake_gates)
    result = p.evaluate_prior_historical_gates(
        rows,
        locked_candidate=locked,
        panel=panel,
    )

    records = captured["records"]
    assert isinstance(records, list)
    assert len(records) == p.PRIOR_SCREEN_CASE_COUNT * len(methods) == 2_880
    assert captured["gate_candidate_ids"] == ("E",)
    assert result["status"] == gate_status
    assert result["locked_candidate"] == locked
    assert result["runner_up_after_lock"] is False
    assert captured["gate_summary"] == {"artifact_completeness": {"status": "fail"}, "candidates": {}}

    by_case: dict[str, list[dict[str, object]]] = {}
    for record in records:
        assert "candidate_score" not in record
        assert "evaluation_score" not in record
        by_case.setdefault(str(record["case_id"]), []).append(record)
    assert tuple(by_case) == tuple(case.case_id for case in panel.cases)
    assert set(by_case) == {case.case_id for case in panel.cases}
    assert all(
        tuple(record["candidate_id"] for record in case_records) == ("A", "B", "D", "E")
        for case_records in by_case.values()
    )
    assert all(
        record["candidate_id"] not in {"L", locked}
        for record in records
    )


@pytest.mark.parametrize(
    ("gate_status", "expected"),
    [("pass", "pass"), ("fail", "fail"), ("inconclusive", "inconclusive")],
)
def test_historical_gate_status_is_explicit_and_never_reopens_lock(
    gate_status: str, expected: str
) -> None:
    rows = [{"candidate_id": "W50-CB", "case_id": "case-0"}]

    def evaluator(
        _rows: list[dict[str, object]],
        locked_candidate: str,
        panel: object = None,
    ) -> dict[str, object]:
        return {
            "status": gate_status,
            "locked_candidate": locked_candidate,
            "runner_up_after_lock": False,
        }

    result = p.evaluate_prior_historical_gates(
        rows,
        locked_candidate="W50-CB",
        evaluator=evaluator,
    )
    assert result["status"] == expected
    assert result["locked_candidate"] == "W50-CB"
    assert result["runner_up_after_lock"] is False


def test_atomic_stop_emits_stopped_artifact_without_partial_checkpoint(tmp_path: Path) -> None:
    evidence = _evidence()
    case = old_runner.stage_cases("screen")[0]
    panel = p.PriorPanel(
        cases=(case,),
        source_evidence=evidence,
        manifest={},
        raw_metadata={},
        case_grid_sha256=p.prior_case_grid_sha256((case,)),
        generator_path="archived-generator",
        generator_sha256="generator-sha",
        backend="MiniBatchKMeans",
        fit_split="train",
        evaluation_split="evaluation",
    )
    dataset = {
        "train": {"X": np.zeros((4, 2)), "y": np.asarray([0, 0, 1, 1])},
        "evaluation": {"X": np.zeros((4, 2)), "y": np.asarray([0, 0, 1, 1])},
    }

    def estimator_factory(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("intentional atomic candidate failure")

    paths = p.run_prior_regression(
        tmp_path,
        _decision("W50-CB"),
        evidence=evidence,
        panel=panel,
        dataset_factory=lambda _case: dataset,
        estimator_factory=estimator_factory,
        warmup=False,
        resume=False,
    )
    assert paths["stopped"].is_file()
    assert json.loads(paths["summary"].read_text(encoding="utf-8"))["status"] == "stopped"
    assert json.loads(paths["raw"].read_text(encoding="utf-8"))["artifact_status"] == "stopped"
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    decision_path = paths["decision"]
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    expected_methods = ["A", "B", "L", "W50-CB"]
    assert manifest["method_ids"] == expected_methods
    assert manifest["promotion_decision_sha256"] == p.decision_sha256(_decision("W50-CB"))
    assert manifest["protocol_sha256"] == p.sha256_path(p.CURRENT_PROTOCOL_PATH)
    assert manifest["code_identity_sha256"] == p.current_code_identity()
    assert manifest["prior_source_evidence"] == evidence.as_dict()
    assert decision["locked_candidate"] == "W50-CB"
    assert decision["selected_candidate"] is None
    assert decision["status"] == "rejected"
    assert decision["method_ids"] == expected_methods
    assert decision["promotion_decision_sha256"] == p.decision_sha256(_decision("W50-CB"))
    assert decision["prior_source_evidence"] == evidence.as_dict()
    assert decision["runner_up_after_lock"] is False
    table_manifest = manifest["table_manifest"]
    assert set(table_manifest) == {
        "case_rows.jsonl",
        "geometry_rows.jsonl",
        "prototype_rows.jsonl",
        "pair_rows.jsonl",
        "selector_panels.jsonl",
        "resource_rows.jsonl",
    }
    for name, table in table_manifest.items():
        table_path = tmp_path / "tables" / name
        assert table_path.is_file()
        assert table["sha256"] == p.sha256_path(table_path)
        assert table["row_count"] == 0

    checkpoint = tmp_path / "checkpoints" / f"{case.case_id}.json"
    assert checkpoint.is_file()
    checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))["payload"]
    assert checkpoint_payload["status"] == "error"
    assert checkpoint_payload["rows"] == []
    assert not list(tmp_path.rglob(".partial-*"))

    # A final decision is immutable: a second invocation fails in preflight,
    # before touching the manifest, tables, raw result, or decision bytes.
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (paths["manifest"], paths["raw"], paths["summary"], decision_path)
    }
    with pytest.raises(FileExistsError, match="immutable"):
        p.run_prior_regression(
            tmp_path,
            _decision("W50-CB"),
            evidence=evidence,
            panel=panel,
            dataset_factory=lambda _case: dataset,
            estimator_factory=estimator_factory,
            warmup=False,
            resume=False,
        )
    for path, (payload_before, mtime_before) in before.items():
        assert path.read_bytes() == payload_before
        assert path.stat().st_mtime_ns == mtime_before


def test_existing_manifest_identity_mismatch_is_fail_closed_before_generation(tmp_path: Path) -> None:
    evidence = _evidence()
    case = old_runner.stage_cases("screen")[0]
    panel = p.PriorPanel(
        cases=(case,),
        source_evidence=evidence,
        manifest={},
        raw_metadata={},
        case_grid_sha256=p.prior_case_grid_sha256((case,)),
        generator_path="archived-generator",
        generator_sha256="generator-sha",
        backend="MiniBatchKMeans",
        fit_split="train",
        evaluation_split="evaluation",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        p.canonical_json(
            {
                "schema_version": 1,
                "stage": "prior_regression",
                "locked_candidate": "stale-lock",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    before = (manifest_path.read_bytes(), manifest_path.stat().st_mtime_ns)
    touched: list[str] = []

    def fail_if_called(_case: object) -> object:
        touched.append("dataset")
        raise AssertionError("identity mismatch reached dataset generation")

    with pytest.raises(p.PriorIdentityError, match="identity"):
        p.run_prior_regression(
            tmp_path,
            _decision("W50-CB"),
            evidence=evidence,
            panel=panel,
            dataset_factory=fail_if_called,
            warmup=False,
            resume=False,
        )
    assert touched == []
    assert manifest_path.read_bytes() == before[0]
    assert manifest_path.stat().st_mtime_ns == before[1]
