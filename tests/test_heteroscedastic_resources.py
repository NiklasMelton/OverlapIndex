"""Outcome-free fresh-process resource orchestration tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.heteroscedastic_distance_conditioning import fixtures
from experiments.heteroscedastic_distance_conditioning import resource_benchmark as resources
from experiments.heteroscedastic_distance_conditioning import resource_worker
from experiments.heteroscedastic_distance_conditioning import runner


PROMOTABLE_RESOURCE_METHODS = (
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


def _development_resource_cases():
    # Planning the 5,184 identities does not allocate any fixture banks.
    return fixtures.development_cases()


def test_resource_request_count_and_cyclic_positions_are_exact() -> None:
    requests = resources.resource_requests(
        _development_resource_cases(), PROMOTABLE_RESOURCE_METHODS
    )
    assert len(requests) == 12 * 4 * 9
    assert {row["resource_id"] for row in requests} == {"R0", "R1", "R2", "R3"}
    resources.validate_resource_requests(
        requests, candidate_ids=PROMOTABLE_RESOURCE_METHODS
    )
    for resource_id, *_ in resources.RESOURCE_CELLS:
        rows = [row for row in requests if row["resource_id"] == resource_id]
        assert len(rows) == 12 * 9
        for case_id in {row["case"]["case_id"] for row in rows}:
            block = [row for row in rows if row["case"]["case_id"] == case_id]
            assert {row["execution_position"] for row in block} == set(range(9))


def test_resource_request_method_set_cannot_be_truncated() -> None:
    with pytest.raises(ValueError, match="exactly"):
        resources.resource_requests(_development_resource_cases(), ("B",))


def test_confirmation_resource_requests_require_lock_and_have_exact_count() -> None:
    token = fixtures._make_confirmation_authorization("promotion", "prior")
    requests = resources.resource_requests(
        fixtures.confirmation_cases(),
        ("B", "P50-SW"),
        stage="confirmation",
        locked_candidate="P50-SW",
        authorization=token,
    )
    assert len(requests) == 24 * 4 * 2
    assert all(row["stage"] == "confirmation" for row in requests)
    assert all(row["confirmation_authorization"] is not None for row in requests)
    resources.validate_resource_requests(
        requests,
        candidate_ids=("B", "P50-SW"),
        stage="confirmation",
        cases=fixtures.confirmation_cases(),
    )
    with pytest.raises(PermissionError):
        resources.resource_requests(
            fixtures.confirmation_cases(),
            ("B", "P50-SW"),
            stage="confirmation",
            locked_candidate="P50-SW",
        )


def test_resource_worker_response_uses_worker_memory_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Completed:
        returncode = 0
        stdout = json.dumps(
            {
                "fresh_process": True,
                "memory_source": "worker_ru_maxrss",
                "parent_peak_rss_bytes": None,
                "status": "ok",
            }
        )
        stderr = ""

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return Completed()

    request = {
        "stage": "development",
        "resource_id": "R0",
        "case": {"case_id": "case"},
        "candidate_id": "B",
    }
    response = resources.run_worker(request, subprocess_runner=fake_run)
    assert response["fresh_process"] is True
    assert response["memory_source"] == "worker_ru_maxrss"
    assert response["parent_peak_rss_bytes"] is None
    assert calls and calls[0][0][0][1:3] == ["-m", "experiments.heteroscedastic_distance_conditioning.resource_worker"]


def test_measure_request_does_not_substitute_parent_rss() -> None:
    case = fixtures.development_cases()[0]

    class FakeEstimator:
        runtime_diagnostics_ = {
            "conditioning_fit_wall_seconds": 0.1,
            "conditioning_fit_cpu_seconds": 0.01,
            "oi_fit_wall_seconds": 0.2,
            "oi_fit_cpu_seconds": 0.02,
        }

        def fit(self, X, y):
            return self

        def score_fixed(self, X, y):
            return 0.5

    dataset = SimpleNamespace(
        train=SimpleNamespace(X=np.zeros((4, 2)), y=np.asarray([0, 1, 0, 1])),
        evaluation=SimpleNamespace(X=np.zeros((4, 2)), y=np.asarray([0, 1, 0, 1])),
    )
    request = {
        "stage": "development",
        "resource_id": "R0",
        "case": case.identity(),
        "candidate_id": "B",
    }
    result = resource_worker.measure_request(
        request,
        dataset_factory=lambda _case: dataset,
        estimator_factory=lambda _spec, _k, _seed: FakeEstimator(),
    )
    assert result["status"] == "ok"
    assert result["fresh_process"] is True
    assert result["memory_source"] == "worker_ru_maxrss"
    assert result["parent_peak_rss_bytes"] is None
    assert result["total_wall_seconds"] >= 0.0
    assert result["total_cpu_seconds"] >= 0.0


def test_resource_aggregation_requires_expected_request_identity() -> None:
    expected = [
        {"resource_id": "R0", "case": {"case_id": "a"}, "candidate_id": "B"},
        {"resource_id": "R0", "case": {"case_id": "a"}, "candidate_id": "L"},
    ]
    rows = [
        {
            "resource_id": "R0",
            "case_id": "a",
            "candidate_id": "B",
            "total_wall_seconds": 1.0,
            "total_cpu_seconds": 1.0,
            "score_fixed_wall_seconds": 1.0,
            "score_fixed_cpu_seconds": 1.0,
            "peak_rss_bytes": 100.0,
        }
    ]
    summary = resources.aggregate_resource_rows(rows, expected_requests=expected)
    assert summary["status"] == "inconclusive"
    assert summary["missing_expected"] == [["R0", "a", "L"]]


def test_resource_aggregation_uses_linear_p95_and_paired_baseline() -> None:
    rows = []
    expected = []
    for index in range(4):
        case_id = "case-{}".format(index)
        for candidate, multiplier in (("B", 1.0), ("L", 2.0)):
            rows.append(
                {
                    "resource_id": "R0",
                    "case_id": case_id,
                    "candidate_id": candidate,
                    "total_wall_seconds": multiplier,
                    "total_cpu_seconds": multiplier,
                    "score_fixed_wall_seconds": multiplier,
                    "score_fixed_cpu_seconds": multiplier,
                    "peak_rss_bytes": 100.0 * multiplier,
                }
            )
            expected.append({"resource_id": "R0", "case": {"case_id": case_id}, "candidate_id": candidate})
    summary = resources.aggregate_resource_rows(rows, expected_requests=expected)
    values = summary["candidates"]["L"]
    assert values["paired_count"] == 4
    assert values["total_wall_ratio_median"] == pytest.approx(2.0)
    assert values["total_wall_ratio_p95"] == pytest.approx(2.0)
    assert values["peak_memory_ratio_max"] == pytest.approx(2.0)
    assert resources.check_resource_gates(summary)["L"]["status"] == "fail"
