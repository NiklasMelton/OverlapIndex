"""Parent-side orchestration for fresh-process resource measurements."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from . import fixtures, runner


RESOURCE_CELLS: Tuple[Tuple[str, str, str, str, int, int, str], ...] = (
    ("R0", "S0", "balanced", "small", 4, 2, "linear_separated"),
    ("R1", "H2", "balanced", "large", 32, 8, "linear_separated"),
    ("R2", "C", "imbalanced", "large", 32, 8, "genuine_overlap_half"),
    ("R3", "ALL", "imbalanced", "large", 32, 8, "nonlinear_separated"),
)
RESOURCE_CELL_BY_ID = {row[0]: row for row in RESOURCE_CELLS}


def _resource_order_index(resource_id: Any) -> int:
    value = str(resource_id)
    for index, row in enumerate(RESOURCE_CELLS):
        if row[0] == value:
            return index
    raise ValueError("unknown resource cell {!r}".format(resource_id))


def _resource_cell_identity(resource_id: str) -> Dict[str, Any]:
    try:
        _, scenario, balance, count_level, nuisance_dim, k, signal_state = RESOURCE_CELL_BY_ID[str(resource_id)]
    except KeyError as exc:
        raise ValueError("unknown resource cell {!r}".format(resource_id)) from exc
    return {
        "resource_id": str(resource_id),
        "scenario": scenario,
        "balance": balance,
        "count_level": count_level,
        "nuisance_dim": int(nuisance_dim),
        "k": int(k),
        "signal_state": signal_state,
    }


def resource_cell_identity(resource_id: str) -> Dict[str, Any]:
    return dict(_resource_cell_identity(resource_id))


def resource_cases(cases: Sequence[Any], resource_id: str) -> Tuple[Any, ...]:
    identity = _resource_cell_identity(resource_id)
    selected = []
    for case in cases:
        if all(getattr(case, key, None) == value for key, value in identity.items() if key != "resource_id"):
            selected.append(case)
    if not selected:
        raise ValueError("no case matches resource cell {!r}".format(resource_id))
    return tuple(selected)


def _planned_resource_schedule(
    cases: Sequence[Any],
    candidates: Sequence[str],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Build one hash-ranked cyclic schedule over every resource/case block."""

    candidate_tuple = tuple(str(value) for value in candidates)
    blocks: List[Tuple[str, str]] = []
    for resource_id, *_ in RESOURCE_CELLS:
        for case in resource_cases(cases, resource_id):
            blocks.append((resource_id, runner.case_id(case)))
    ranked = sorted(
        blocks,
        key=lambda block: (
            hashlib.sha256(
                "{}\0resource\0{}\0{}".format(
                    runner.SCHEDULE_SEED,
                    block[0],
                    block[1],
                ).encode("utf-8")
            ).digest(),
            block,
        ),
    )
    result: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for rank, block in enumerate(ranked):
        offset = rank % len(candidate_tuple)
        order = tuple(
            candidate_tuple[(offset + position) % len(candidate_tuple)]
            for position in range(len(candidate_tuple))
        )
        result[block] = {
            "positions": {candidate: position for position, candidate in enumerate(order)},
            "order": order,
        }
    return result


def resource_requests(
    cases: Sequence[Any],
    candidate_ids: Sequence[str],
    *,
    stage: str = "development",
    locked_candidate: Optional[str] = None,
    authorization: Any = None,
    code_identity_sha256: Optional[str] = None,
    protocol_sha256: Optional[str] = None,
) -> Tuple[Dict[str, Any], ...]:
    """Build deterministic worker requests for all seed/cell/candidate rows."""

    stage_value = str(stage).lower()
    if stage_value not in {"development", "confirmation"}:
        raise PermissionError("resource requests are authorized only for development or confirmation")
    candidates = tuple(str(value) for value in candidate_ids)
    if stage_value == "development":
        expected_candidates = ("B", "L") + runner.PROMOTABLE_CANDIDATES
    else:
        if authorization is None:
            raise PermissionError("confirmation resource requests require validated authorization")
        try:
            from . import manifest

            manifest.validate_confirmation_authorization(authorization)
            fixtures._check_confirmation_authorization(authorization)
        except (ImportError, AttributeError) as exc:
            raise PermissionError("confirmation authorization cannot be validated") from exc
        if locked_candidate is None:
            raise PermissionError("confirmation resource requests require the locked candidate")
        if str(locked_candidate) not in runner.PROMOTABLE_CANDIDATES:
            raise ValueError("confirmation lock must name one promotable candidate")
        expected_candidates = ("B", str(locked_candidate))
    authorization_payload = None
    if stage_value == "confirmation":
        authorization_payload = {
            "promotion_decision_sha256": str(authorization.promotion_decision_sha256),
            "prior_regression_decision_sha256": str(authorization.prior_regression_decision_sha256),
            "protocol_sha256": str(authorization.protocol_sha256),
        }
    if candidates != tuple(expected_candidates):
        raise ValueError(
            "resource candidate set must be exactly {!r}; got {!r}".format(
                expected_candidates, candidates
            )
        )
    output: List[Dict[str, Any]] = []
    for resource_id, *_ in RESOURCE_CELLS:
        matched = resource_cases(cases, resource_id)
        for case in sorted(matched, key=lambda value: runner.case_id(value)):
            for candidate_id in candidates:
                output.append(
                    {
                        "schema_version": runner.SCHEMA_VERSION,
                        "stage": stage_value,
                        "resource_id": resource_id,
                        "case": runner._case_identity(case),
                        "candidate_id": candidate_id,
                        "locked_candidate": str(locked_candidate) if stage_value == "confirmation" else None,
                        "code_identity_sha256": code_identity_sha256,
                        "protocol_sha256": protocol_sha256,
                        "confirmation_authorization": authorization_payload,
                        "warmup_excluded": True,
                    }
                )
    # Counterbalance over the complete seed x resource-cell block set, rather
    # than reusing one case-only offset in all four resource cells.
    resource_schedule = _planned_resource_schedule(cases, candidates)
    for row in output:
        block = (str(row["resource_id"]), str(row["case"]["case_id"]))
        schedule = resource_schedule[block]
        row["execution_position"] = schedule["positions"][str(row["candidate_id"])]
        row["execution_order"] = list(schedule["order"])
        row["schedule_seed"] = runner.SCHEDULE_SEED
    output.sort(
        key=lambda row: (
            _resource_order_index(row["resource_id"]),
            str(row["case"]["case_id"]),
            int(row["execution_position"]),
            str(row["candidate_id"]),
        )
    )
    validate_resource_requests(
        output,
        candidate_ids=candidates,
        stage=stage_value,
        cases=cases,
    )
    return tuple(output)


def validate_resource_requests(
    requests: Sequence[Mapping[str, Any]],
    *,
    candidate_ids: Sequence[str],
    stage: str = "development",
    cases: Optional[Sequence[Any]] = None,
) -> bool:
    """Validate exact worker identities and per-case cyclic positions."""

    candidates = tuple(str(value) for value in candidate_ids)
    if str(stage).lower() not in {"development", "confirmation"}:
        raise ValueError("resource request stage must be development or confirmation")
    seen: set[Tuple[str, str, str]] = set()
    by_case: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for request in requests:
        if str(request.get("stage", stage)).lower() != str(stage).lower():
            raise ValueError("resource request stage mismatch")
        if str(request.get("resource_id")) not in RESOURCE_CELL_BY_ID:
            raise ValueError("unknown resource id in request")
        case = request.get("case", {})
        cid = str(case.get("case_id", request.get("case_id", "")))
        candidate = str(request.get("candidate_id", ""))
        key = (str(request.get("resource_id")), cid, candidate)
        if key in seen:
            raise ValueError("duplicate resource request identity: {}".format(key))
        seen.add(key)
        block = (str(request.get("resource_id")), cid)
        by_case.setdefault(block, []).append(request)
    for block, rows in by_case.items():
        cid = block[1]
        if {str(row.get("candidate_id")) for row in rows} != set(candidates):
            raise ValueError("resource candidate block is incomplete for {}".format(block))
        positions = [int(row.get("execution_position", -1)) for row in rows]
        if set(positions) != set(range(len(candidates))):
            raise ValueError("resource execution positions are not a complete cycle for {}".format(block))
        order = tuple(
            str(row.get("candidate_id"))
            for row in sorted(rows, key=lambda row: int(row.get("execution_position", -1)))
        )
        expected_order = tuple(rows[0].get("execution_order", ()))
        if order != expected_order:
            raise ValueError("resource execution order trace disagrees for {}".format(block))
    if cases is not None:
        expected: set[Tuple[str, str, str]] = set()
        expected_case_ids: set[str] = set()
        for resource_id, *_ in RESOURCE_CELLS:
            for case in resource_cases(cases, resource_id):
                expected_case_ids.add(runner.case_id(case))
                for candidate in candidates:
                    expected.add((resource_id, runner.case_id(case), candidate))
        observed = {
            (
                str(request.get("resource_id")),
                str(request.get("case", {}).get("case_id", request.get("case_id"))),
                str(request.get("candidate_id")),
            )
            for request in requests
        }
        if observed != expected:
            raise ValueError(
                "resource requests do not cover the exact planned cells: "
                "missing={!r}, unexpected={!r}".format(
                    sorted(expected - observed), sorted(observed - expected)
                )
            )
        if {key[1] for key in observed} != expected_case_ids:
            raise ValueError("resource requests contain an unexpected case identity")
        expected_schedule = _planned_resource_schedule(cases, candidates)
        for block, rows in by_case.items():
            planned = expected_schedule.get(block)
            if planned is None:
                raise ValueError("resource request block is outside the frozen schedule: {}".format(block))
            observed_positions = {
                str(row.get("candidate_id")): int(row.get("execution_position", -1))
                for row in rows
            }
            if observed_positions != planned["positions"]:
                raise ValueError("resource execution positions disagree with the frozen schedule for {}".format(block))
            observed_order = tuple(
                str(row.get("candidate_id"))
                for row in sorted(rows, key=lambda row: int(row.get("execution_position", -1)))
            )
            if observed_order != tuple(planned["order"]):
                raise ValueError("resource execution order disagrees with the frozen schedule for {}".format(block))
    return True


def run_worker(
    request: Mapping[str, Any],
    *,
    python_executable: Optional[str] = None,
    subprocess_runner: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Run one fresh worker and parse exactly one canonical response object."""

    executable = python_executable or sys.executable
    command = [executable, "-m", "experiments.heteroscedastic_distance_conditioning.resource_worker"]
    run = subprocess_runner or subprocess.run
    completed = run(
        command,
        input=runner.canonical_json(request) + "\n",
        text=True,
        capture_output=True,
        check=False,
        cwd=str(runner.ROOT),
        env=dict(os.environ),
    )
    if int(getattr(completed, "returncode", 0)) != 0:
        error = getattr(completed, "stderr", "") or "worker exited with nonzero status"
        return {
            "schema_version": runner.SCHEMA_VERSION,
            "stage": request.get("stage"),
            "resource_id": request.get("resource_id"),
            "case_id": request.get("case", {}).get("case_id"),
            "candidate_id": request.get("candidate_id"),
            "status": "error",
            "error": str(error).strip(),
            "fresh_process": True,
            "memory_source": "worker_ru_maxrss",
            "parent_peak_rss_bytes": None,
        }
    stdout = str(getattr(completed, "stdout", "")).strip().splitlines()
    if len(stdout) != 1:
        raise RuntimeError("resource worker must emit exactly one JSON response line")
    try:
        response = json.loads(stdout[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError("resource worker emitted invalid JSON") from exc
    if not isinstance(response, Mapping):
        raise RuntimeError("resource worker response must be a JSON object")
    response = dict(response)
    if response.get("fresh_process") is not True:
        raise RuntimeError("resource response did not identify a fresh process")
    if response.get("memory_source") != "worker_ru_maxrss":
        raise RuntimeError("resource response used a non-worker memory source")
    if response.get("parent_peak_rss_bytes") is not None:
        raise RuntimeError("resource response must not copy parent RSS")
    return runner._json_safe(response)


def run_resource_benchmark(
    cases: Sequence[Any],
    candidate_ids: Sequence[str],
    *,
    output: Optional[Union[str, Path]] = None,
    code_identity_sha256: Optional[str] = None,
    protocol_sha256: Optional[str] = None,
    worker_runner: Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]] = None,
    stage: str = "development",
    locked_candidate: Optional[str] = None,
    authorization: Any = None,
    warmup: bool = True,
) -> List[Dict[str, Any]]:
    """Collect resource rows in isolated processes, then optionally persist JSONL."""

    if not warmup:
        raise RuntimeError(
            "resource benchmarks require an excluded disposable warmup process"
        )

    requests = resource_requests(
        cases,
        candidate_ids,
        stage=stage,
        locked_candidate=locked_candidate,
        authorization=authorization,
        code_identity_sha256=code_identity_sha256,
        protocol_sha256=protocol_sha256,
    )
    rows: List[Dict[str, Any]] = []
    if warmup and requests:
        # A disposable fresh worker is used for warm-up.  Its response is
        # deliberately not appended to the measured resource table.
        warmup_request = dict(requests[0])
        warmup_request["warmup"] = True
        warmup_response = (
            worker_runner(warmup_request)
            if worker_runner is not None
            else run_worker(warmup_request)
        )
        _validate_worker_response(warmup_request, warmup_response, warmup=True)
    observed: set[Tuple[str, str, str]] = set()
    for request in requests:
        response = worker_runner(request) if worker_runner is not None else run_worker(request)
        _validate_worker_response(request, response, warmup=False)
        row = dict(response)
        case_identity = request.get("case", {})
        if isinstance(case_identity, Mapping):
            for axis, value in case_identity.items():
                # The request is the canonical identity source; a worker may
                # add timing fields but cannot rewrite case axes.
                if axis == "case_id":
                    continue
                row.setdefault(str(axis), runner._json_safe(value))
        row.setdefault("resource_id", request["resource_id"])
        row.setdefault("case_id", request["case"]["case_id"])
        row.setdefault("candidate_id", request["candidate_id"])
        row.setdefault("stage", request.get("stage"))
        row.setdefault("execution_position", request.get("execution_position"))
        row.setdefault("execution_order", request.get("execution_order"))
        row.setdefault("schedule_seed", request.get("schedule_seed"))
        row.setdefault("request_case_identity", runner._json_safe(case_identity))
        row.setdefault("code_identity_sha256", request.get("code_identity_sha256"))
        row.setdefault("protocol_sha256", request.get("protocol_sha256"))
        row["request_identity_sha256"] = runner.digest(request)
        row["warmup_excluded"] = True
        identity = (
            str(row.get("resource_id")),
            str(row.get("case_id")),
            str(row.get("candidate_id")),
        )
        if identity in observed:
            raise RuntimeError("duplicate measured resource response: {}".format(identity))
        observed.add(identity)
        rows.append(runner._json_safe(row))
    expected = {
        (
            str(request.get("resource_id")),
            str(request.get("case", {}).get("case_id")),
            str(request.get("candidate_id")),
        )
        for request in requests
    }
    if observed != expected:
        raise RuntimeError(
            "resource response identities do not match the planned requests: "
            "missing={!r}, unexpected={!r}".format(
                sorted(expected - observed), sorted(observed - expected)
            )
        )
    rows.sort(
        key=lambda row: (
            _resource_order_index(row.get("resource_id")),
            str(row.get("case_id")),
            int(row.get("execution_position", 0)),
            str(row.get("candidate_id")),
        )
    )
    if output is not None:
        path = Path(output)
        payload = b"".join((runner.canonical_json(row) + "\n").encode("utf-8") for row in rows)
        runner._atomic_write_bytes(path, payload)
    return rows


def _validate_worker_response(
    request: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    warmup: bool,
) -> None:
    """Validate one fresh-worker response before it can enter a table."""

    if not isinstance(response, Mapping):
        raise RuntimeError("resource worker response must be a JSON object")
    expected_identity = {
        "stage": str(request.get("stage")),
        "resource_id": str(request.get("resource_id")),
        "case_id": str(request.get("case", {}).get("case_id")),
        "candidate_id": str(request.get("candidate_id")),
    }
    for key, expected in expected_identity.items():
        observed = str(response.get(key))
        if observed != expected:
            raise RuntimeError(
                "resource worker identity mismatch for {}: expected {!r}, got {!r}".format(
                    key, expected, response.get(key)
                )
            )
    if response.get("status") != "ok":
        kind = "warmup" if warmup else "measured"
        raise RuntimeError(
            "{} resource worker failed for {}: {}".format(
                kind,
                expected_identity["case_id"],
                response.get("error"),
            )
        )
    if response.get("fresh_process") is not True:
        raise RuntimeError("resource worker did not report a fresh process")
    if response.get("memory_source") != "worker_ru_maxrss":
        raise RuntimeError("resource worker did not report worker_ru_maxrss")
    if response.get("parent_peak_rss_bytes") is not None:
        raise RuntimeError("resource worker response contains parent RSS")
    required = (
        "total_wall_seconds",
        "total_cpu_seconds",
        "score_fixed_wall_seconds",
        "score_fixed_cpu_seconds",
        "peak_rss_bytes",
    )
    for field in required:
        try:
            value = float(response[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("resource response missing {}".format(field)) from exc
        if not np.isfinite(value) or value < 0.0:
            raise RuntimeError("resource response has non-finite {}".format(field))
    if float(response["peak_rss_bytes"]) <= 0.0:
        raise RuntimeError("resource response peak_rss_bytes must be positive")


def _quantile(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    try:
        return float(np.quantile(np.asarray(values, dtype=float), probability, method="linear"))
    except TypeError:  # NumPy < 1.22 compatibility for the archived env.
        return float(np.quantile(np.asarray(values, dtype=float), probability, interpolation="linear"))


def aggregate_resource_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_candidate: str = "B",
    expected_requests: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Aggregate paired candidate/B timing and memory ratios.

    Pairing is exact within ``(resource_id, case_id)``.  Missing or non-finite
    mandatory values are reported as inconclusive rather than silently dropped
    from a passing denominator.
    """

    by_key: Dict[Tuple[str, str], Dict[str, Mapping[str, Any]]] = {}
    duplicate_rows: List[Dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("resource_id")), str(row.get("case_id")))
        candidate = str(row.get("candidate_id"))
        target = by_key.setdefault(key, {})
        if candidate in target:
            duplicate_rows.append(
                {"resource_id": key[0], "case_id": key[1], "candidate_id": candidate}
            )
        target[candidate] = row
    expected_keys: Optional[set[Tuple[str, str, str]]] = None
    if expected_requests is not None:
        expected_keys = set()
        for request in expected_requests:
            case_value = request.get("case", {})
            expected_case_id = request.get("case_id", case_value.get("case_id"))
            expected_keys.add(
                (
                    str(request.get("resource_id")),
                    str(expected_case_id),
                    str(request.get("candidate_id")),
                )
            )
        observed_keys = {
            (key[0], key[1], candidate)
            for key, mapping in by_key.items()
            for candidate in mapping
        }
        missing_expected = sorted(expected_keys - observed_keys)
        unexpected_observed = sorted(observed_keys - expected_keys)
        if missing_expected or unexpected_observed or duplicate_rows:
            return runner._json_safe(
                {
                    "schema_version": runner.SCHEMA_VERSION,
                    "baseline_candidate": baseline_candidate,
                    "candidate_count": 0,
                    "candidates": {},
                    "status": "inconclusive",
                    "missing_expected": [list(value) for value in missing_expected],
                    "unexpected_observed": [list(value) for value in unexpected_observed],
                    "duplicate_rows": duplicate_rows,
                }
            )
    identity_failures: List[Dict[str, Any]] = []
    for key, mapping in by_key.items():
        for axis in ("seed", "scenario", "balance", "count_level", "nuisance_dim", "k", "signal_state"):
            observed_values = {
                runner.canonical_json(row.get(axis))
                for row in mapping.values()
                if row.get(axis) is not None
            }
            if len(observed_values) > 1:
                identity_failures.append(
                    {"resource_id": key[0], "case_id": key[1], "axis": axis}
                )
    if identity_failures:
        return runner._json_safe(
            {
                "schema_version": runner.SCHEMA_VERSION,
                "baseline_candidate": baseline_candidate,
                "candidate_count": 0,
                "candidates": {},
                "status": "inconclusive",
                "identity_failures": identity_failures,
                "duplicate_rows": duplicate_rows,
            }
        )
    candidates = sorted({str(row.get("candidate_id")) for row in rows if str(row.get("candidate_id")) != baseline_candidate})
    output: Dict[str, Any] = {
        "schema_version": runner.SCHEMA_VERSION,
        "baseline_candidate": baseline_candidate,
        "candidate_count": len(candidates),
        "status": "pass" if not duplicate_rows else "inconclusive",
        "duplicate_rows": duplicate_rows,
        "identity_failures": [],
        "candidates": {},
    }
    fields = {
        "total_wall_ratio": "total_wall_seconds",
        "total_cpu_ratio": "total_cpu_seconds",
        "score_fixed_wall_ratio": "score_fixed_wall_seconds",
        "score_fixed_cpu_ratio": "score_fixed_cpu_seconds",
        "peak_memory_ratio": "peak_rss_bytes",
    }
    for candidate in candidates:
        ratios: Dict[str, List[float]] = {name: [] for name in fields}
        missing: List[Dict[str, Any]] = []
        for key, mapping in sorted(by_key.items()):
            base = mapping.get(baseline_candidate)
            other = mapping.get(candidate)
            if base is None or other is None:
                missing.append({"resource_id": key[0], "case_id": key[1], "reason": "missing_pair"})
                continue
            for ratio_name, field in fields.items():
                try:
                    numerator = float(other[field])
                    denominator = float(base[field])
                except (KeyError, TypeError, ValueError):
                    missing.append({"resource_id": key[0], "case_id": key[1], "field": field, "reason": "missing_or_nonfinite"})
                    continue
                if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
                    missing.append({"resource_id": key[0], "case_id": key[1], "field": field, "reason": "missing_or_nonfinite"})
                    continue
                ratios[ratio_name].append(numerator / denominator)
        output["candidates"][candidate] = {
            "paired_count": len(ratios["total_wall_ratio"]),
            "missing_count": len(missing),
            "missing": missing,
            "total_wall_ratio_median": _quantile(ratios["total_wall_ratio"], 0.50),
            "total_wall_ratio_p95": _quantile(ratios["total_wall_ratio"], 0.95),
            "total_cpu_ratio_median": _quantile(ratios["total_cpu_ratio"], 0.50),
            "total_cpu_ratio_p95": _quantile(ratios["total_cpu_ratio"], 0.95),
            "score_fixed_wall_ratio_median": _quantile(ratios["score_fixed_wall_ratio"], 0.50),
            "score_fixed_cpu_ratio_median": _quantile(ratios["score_fixed_cpu_ratio"], 0.50),
            "peak_memory_ratio_median": _quantile(ratios["peak_memory_ratio"], 0.50),
            "peak_memory_ratio_p95": _quantile(ratios["peak_memory_ratio"], 0.95),
            "peak_memory_ratio_max": max(ratios["peak_memory_ratio"], default=None),
            "inconclusive": bool(missing),
        }
    return runner._json_safe(output)


def check_resource_gates(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply frozen resource thresholds without selecting a candidate."""

    decisions: Dict[str, Any] = {}
    for candidate, values in dict(summary.get("candidates", {})).items():
        if values.get("inconclusive"):
            decisions[candidate] = {"status": "inconclusive", "reasons": ["missing_or_nonfinite"]}
            continue
        checks = {
            "total_wall_ratio_median": float(values["total_wall_ratio_median"]) <= 1.10,
            "total_wall_ratio_p95": float(values["total_wall_ratio_p95"]) <= 1.25,
            "score_fixed_wall_ratio_median": float(values["score_fixed_wall_ratio_median"]) <= 1.25,
            "peak_memory_ratio_median": float(values["peak_memory_ratio_median"]) <= 1.15,
            "peak_memory_ratio_p95": float(values["peak_memory_ratio_p95"]) <= 1.25,
            "peak_memory_ratio_max": float(values["peak_memory_ratio_max"]) <= 1.50,
        }
        decisions[candidate] = {
            "status": "pass" if all(checks.values()) else "fail",
            "checks": checks,
        }
    return runner._json_safe(decisions)


__all__ = [
    "RESOURCE_CELLS",
    "RESOURCE_CELL_BY_ID",
    "aggregate_resource_rows",
    "check_resource_gates",
    "resource_cases",
    "resource_cell_identity",
    "resource_requests",
    "validate_resource_requests",
    "run_resource_benchmark",
    "run_worker",
]
