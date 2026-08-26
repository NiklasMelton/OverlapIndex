"""Fresh-process resource worker for the heteroscedastic experiment.

The parent runner starts one worker per resource/case/candidate measurement.
Peak RSS is collected *inside this process* using ``resource.getrusage``; no
value is inherited from, or substituted by, the parent process.  Requests and
responses are canonical JSON objects on stdin/stdout so the worker is also a
small, deterministic test seam.
"""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
import platform
import resource as _resource
import sys
import time
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from . import fixtures, runner


RESOURCE_IDS = ("R0", "R1", "R2", "R3")


def _authorization_from_request(request: Mapping[str, Any]) -> Any:
    """Reconstitute the validated confirmation token inside a fresh worker.

    The parent validates the opaque token before dispatching requests.  Only
    its prerequisite hashes cross the JSON process boundary; the worker
    recreates the process-local marker so fixture generation retains its own
    guard rather than accepting an arbitrary object.
    """

    payload = request.get("confirmation_authorization")
    if not isinstance(payload, Mapping):
        raise PermissionError(
            "confirmation resource requests require validated authorization hashes"
        )
    promotion = str(payload.get("promotion_decision_sha256", ""))
    prior = str(payload.get("prior_regression_decision_sha256", ""))
    protocol_hash = str(payload.get("protocol_sha256", request.get("protocol_sha256", "")))
    if not promotion or not prior or protocol_hash != fixtures.PROTOCOL_SHA256:
        raise PermissionError("confirmation authorization does not match the frozen protocol")
    token = fixtures._make_confirmation_authorization(promotion, prior, protocol_hash)
    fixtures._check_confirmation_authorization(token)
    return token


def _worker_peak_rss_bytes() -> Optional[int]:
    """Return this fresh worker's peak RSS in bytes."""

    try:
        value = int(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None
    # macOS reports bytes, Linux/BSD generally report KiB.  The conversion is
    # local to the worker and is explicitly recorded in the response.
    if sys.platform == "darwin":
        return max(0, value)
    return max(0, value * 1024)


def _case_from_identity(identity: Mapping[str, Any]) -> Any:
    required = (
        "stage",
        "seed",
        "scenario",
        "balance",
        "count_level",
        "nuisance_dim",
        "k",
        "signal_state",
    )
    missing = [name for name in required if name not in identity]
    if missing:
        raise ValueError("resource case is missing identity fields: {}".format(missing))
    return fixtures.case_spec(
        stage=str(identity["stage"]),
        seed=int(identity["seed"]),
        scenario=str(identity["scenario"]),
        balance=str(identity["balance"]),
        count_level=str(identity["count_level"]),
        nuisance_dim=int(identity["nuisance_dim"]),
        k=int(identity["k"]),
        signal_state=str(identity["signal_state"]),
    )


def _validate_request(request: Mapping[str, Any]) -> Tuple[str, Any, runner.CandidateSpec]:
    if not isinstance(request, Mapping):
        raise TypeError("resource request must be a JSON object")
    stage = str(request.get("stage", "")).lower()
    if stage not in {"development", "confirmation"}:
        raise PermissionError(
            "fresh resource workers are authorized only for development or confirmation"
        )
    resource_id = str(request.get("resource_id", ""))
    if resource_id not in RESOURCE_IDS:
        raise ValueError("resource_id must be one of {!r}".format(RESOURCE_IDS))
    request_protocol = request.get("protocol_sha256")
    if request_protocol is not None and str(request_protocol) != fixtures.PROTOCOL_SHA256:
        raise PermissionError("resource request protocol hash mismatch")
    identity = request.get("case")
    if not isinstance(identity, Mapping):
        raise ValueError("resource request must include a case identity object")
    if str(identity.get("stage", "")).lower() != stage:
        raise ValueError("resource request stage and case stage must agree")
    case = _case_from_identity(identity)
    candidate_id = str(request.get("candidate_id", ""))
    try:
        spec = runner.CANDIDATE_BY_ID[candidate_id]
    except KeyError as exc:
        raise ValueError("unknown resource candidate {!r}".format(candidate_id)) from exc
    if stage == "confirmation":
        locked = str(request.get("locked_candidate", ""))
        if not locked or locked not in runner.PROMOTABLE_CANDIDATES or candidate_id not in {"B", locked}:
            raise PermissionError("confirmation worker candidate is outside B/locked set")
        _authorization_from_request(request)
    elif candidate_id not in {"B", "L"}.union(set(runner.PROMOTABLE_CANDIDATES)):
        raise ValueError("development resource candidate is not authorized")
    return resource_id, case, spec


def measure_request(
    request: Mapping[str, Any],
    *,
    dataset_factory: Optional[Callable[..., Any]] = None,
    estimator_factory: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Measure one case/candidate entirely within this process.

    ``dataset_factory`` and ``estimator_factory`` are optional dependency
    seams for focused tests.  Production calls use the frozen fixture and
    runner factories; neither seam may be populated from evaluation outcomes.
    """

    resource_id, case, spec = _validate_request(request)
    stage = str(request.get("stage", "")).lower()
    authorization = _authorization_from_request(request) if stage == "confirmation" else None
    row: Dict[str, Any] = {
        "schema_version": runner.SCHEMA_VERSION,
        "stage": stage,
        "resource_id": resource_id,
        "case_id": runner.case_id(case),
        "candidate_id": spec.candidate_id,
        "candidate_name": spec.name,
        "status": "ok",
        "error": None,
        "fresh_process": True,
        "worker_pid": int(os.getpid()),
        "worker_python": platform.python_version(),
        "worker_executable": sys.executable,
        "memory_source": "worker_ru_maxrss",
        "parent_peak_rss_bytes": None,
        "warmup_excluded": True,
    }
    try:
        generation_wall_start = time.perf_counter()
        generation_cpu_start = time.process_time()
        # Confirmation bank generation remains behind the validated token;
        # development/smoke-style resource cases use the ordinary generator.
        if dataset_factory is None:
            if stage == "confirmation":
                dataset = fixtures.generate_confirmation_dataset(
                    case,
                    authorization=authorization,
                )
            else:
                dataset = fixtures.generate_dataset(case)
        else:
            dataset = dataset_factory(case)
        row["dataset_generation_wall_seconds"] = max(
            0.0, float(time.perf_counter() - generation_wall_start)
        )
        row["dataset_generation_cpu_seconds"] = max(
            0.0, float(time.process_time() - generation_cpu_start)
        )
        # Start the production clock immediately before candidate construction
        # and stop after score_fixed.  Dataset generation above is common setup
        # and remains a separate diagnostic timing.
        total_wall_start = time.perf_counter()
        total_cpu_start = time.process_time()
        if estimator_factory is None:
            model = runner.estimator_for(spec, int(case.k), int(case.seed))
        else:
            model = estimator_factory(spec, int(case.k), int(case.seed))
        # The production total covers candidate construction, fit, and
        # score_fixed only.  Fixture generation is common setup and is timed
        # separately above so it cannot dilute candidate/B ratios.
        fit_wall_start = time.perf_counter()
        fit_cpu_start = time.process_time()
        model.fit(dataset.train.X, dataset.train.y)
        fit_wall = time.perf_counter() - fit_wall_start
        fit_cpu = time.process_time() - fit_cpu_start
        runtime = getattr(model, "runtime_diagnostics_", {})
        if not isinstance(runtime, Mapping):
            runtime = {}
        conditioning_wall = float(runtime.get("conditioning_fit_wall_seconds", 0.0))
        conditioning_cpu = float(runtime.get("conditioning_fit_cpu_seconds", 0.0))
        oi_wall = runtime.get("oi_fit_wall_seconds", fit_wall)
        oi_cpu = runtime.get("oi_fit_cpu_seconds", fit_cpu)
        score_wall_start = time.perf_counter()
        score_cpu_start = time.process_time()
        score = float(model.score_fixed(dataset.evaluation.X, dataset.evaluation.y))
        score_wall = time.perf_counter() - score_wall_start
        score_cpu = time.process_time() - score_cpu_start
        row.update(
            {
                "conditioning_fit_wall_seconds": max(0.0, conditioning_wall),
                "conditioning_fit_cpu_seconds": max(0.0, conditioning_cpu),
                "oi_fit_refinement_wall_seconds": max(0.0, float(oi_wall)),
                "oi_fit_refinement_cpu_seconds": max(0.0, float(oi_cpu)),
                "fit_wall_seconds": max(0.0, float(fit_wall)),
                "fit_cpu_seconds": max(0.0, float(fit_cpu)),
                "score_fixed_wall_seconds": max(0.0, float(score_wall)),
                "score_fixed_cpu_seconds": max(0.0, float(score_cpu)),
                "total_wall_seconds": max(0.0, float(time.perf_counter() - total_wall_start)),
                "total_cpu_seconds": max(0.0, float(time.process_time() - total_cpu_start)),
                "candidate_score": score,
                "peak_rss_bytes": _worker_peak_rss_bytes(),
            }
        )
    except Exception as exc:
        row.update(
            {
                "status": "error",
                "error": "{}: {}".format(type(exc).__name__, exc),
                "peak_rss_bytes": _worker_peak_rss_bytes(),
                "total_wall_seconds": None,
                "total_cpu_seconds": None,
            }
        )
    return runner._json_safe(row)


def worker_main(request: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    if request is None:
        text = sys.stdin.read()
        if not text.strip():
            raise ValueError("resource worker stdin is empty")
        request = json.loads(text)
    response = measure_request(request)
    sys.stdout.write(runner.canonical_json(response) + "\n")
    sys.stdout.flush()
    return response


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, help="JSON request file; stdin by default")
    args = parser.parse_args()
    if args.request is None:
        worker_main()
    else:
        worker_main(json.loads(args.request.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by parent subprocess
    raise SystemExit(_cli())


__all__ = [
    "RESOURCE_IDS",
    "measure_request",
    "worker_main",
    "_worker_peak_rss_bytes",
]
