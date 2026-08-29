"""Frozen development lock and full analysis for the speed-focused experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from experiments.scalable_relevance_kmeans import speed_experiment as experiment


HEADS = ("linear", "quadratic", "knn", "rbf")
NONLINEAR_HEADS = ("quadratic", "knn", "rbf")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_829


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object at {path}")
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, experiment.canonical_json(payload) + "\n")


def _verify_artifact(input_dir: Path, stage: str, lock: Mapping[str, Any] | None = None) -> dict[str, Any]:
    manifest = _read_json(input_dir / "manifest.json")
    raw_path = input_dir / "raw_results.json"
    raw = _read_json(raw_path)
    if manifest.get("artifact_status") != "completed" or raw.get("artifact_status") != "completed":
        raise RuntimeError("speed artifact is not completed")
    if manifest.get("stage") != stage or raw.get("stage") != stage:
        raise RuntimeError("speed artifact stage mismatch")
    if manifest.get("raw_results_sha256") != experiment.sha256_path(raw_path):
        raise RuntimeError("speed artifact raw hash mismatch")
    if manifest.get("run_identity") != raw.get("run_identity"):
        raise RuntimeError("speed manifest/raw identity mismatch")
    identity = raw.get("run_identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("speed run identity is malformed")
    identity_sha = hashlib.sha256(experiment.canonical_json(identity).encode("utf-8")).hexdigest()
    if manifest.get("run_identity_sha256") != identity_sha or raw.get("run_identity_sha256") != identity_sha:
        raise RuntimeError("speed run identity hash mismatch")
    protocol_sha, _ = experiment.validate_protocol()
    sources = experiment.source_identity()
    if identity.get("protocol_sha256") != protocol_sha or identity.get("source_identity") != sources:
        raise RuntimeError("speed artifact no longer matches frozen sources")
    if stage == "development":
        methods = experiment.DEVELOPMENT_METHODS
        expected_counts = {"panels": 30, "selector_rows": 180, "reference_rows": 120, "checkpoints": 30}
    else:
        if lock is None:
            raise RuntimeError("full artifact verification requires the development lock")
        methods = (str(lock["selected_candidate"]), "LP-FULL")
        expected_counts = {"panels": 600, "selector_rows": 1200, "reference_rows": 600, "checkpoints": 600}
        if identity.get("development_lock_sha256") != lock.get("lock_sha256"):
            raise RuntimeError("full artifact development-lock hash mismatch")
    if tuple(identity.get("methods", ())) != methods or manifest.get("counts") != expected_counts:
        raise RuntimeError("speed artifact grid/count mismatch")
    rows = raw.get("selector_rows")
    references = raw.get("reference_rows")
    if not isinstance(rows, list) or len(rows) != expected_counts["selector_rows"]:
        raise RuntimeError("speed selector grid is incomplete")
    if not isinstance(references, list) or len(references) != expected_counts["reference_rows"]:
        raise RuntimeError("speed reference grid is incomplete")
    expected_replicates = (experiment.DEVELOPMENT_REPLICATE,) if stage == "development" else experiment.REPLICATES
    expected_budgets = (experiment.DEVELOPMENT_BUDGET,) if stage == "development" else experiment.BUDGETS
    expected_keys = {
        (model, replicate, arm, budget, method)
        for model in experiment.MODELS
        for replicate in expected_replicates
        for arm in experiment.ARMS
        for budget in expected_budgets
        for method in methods
    }
    observed: set[tuple[str, int, str, int, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") != "ok":
            raise RuntimeError("speed selector row is malformed")
        key = (str(row.get("model")), int(row.get("replicate", -1)), str(row.get("arm")), int(row.get("budget", -1)), str(row.get("candidate_id")))
        if key in observed:
            raise RuntimeError("speed selector row is duplicated")
        observed.add(key)
        for field in ("score", "total_wall_seconds", "total_cpu_seconds"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or float(value) < 0.0:
                raise RuntimeError(f"speed selector row has invalid {field}")
    if observed != expected_keys:
        raise RuntimeError("speed selector identities differ from the frozen grid")
    determinism = raw.get("determinism")
    if not isinstance(determinism, Mapping) or set(determinism) != set(methods):
        raise RuntimeError("speed determinism evidence is incomplete")
    if any(
        not isinstance(value, Mapping)
        or value.get("exact") is not True
        or value.get("runtime_fields_excluded") is not True
        or value.get("first_signature_sha256") != value.get("second_signature_sha256")
        for value in determinism.values()
    ):
        raise RuntimeError("speed determinism evidence is malformed")
    return raw


def _selection_metrics(scores: np.ndarray, outcomes: np.ndarray) -> dict[str, Any]:
    selected = np.flatnonzero(np.isclose(scores, np.max(scores), rtol=0.0, atol=1.0e-12))
    best = float(np.max(outcomes))
    selected_outcome = float(np.mean(outcomes[selected]))
    rho = float(spearmanr(scores, outcomes).statistic)
    return {
        "regret_pp": 100.0 * (best - selected_outcome),
        "spearman": rho if np.isfinite(rho) else None,
        "exact_best": bool(np.any(np.isclose(outcomes[selected], best, rtol=0.0, atol=1.0e-12))),
        "within_one_pp": bool(best - selected_outcome <= 0.01 + 1.0e-12),
        "selected_backbones": [experiment.MODELS[int(index)] for index in selected],
    }


def _metrics(
    raw: Mapping[str, Any], methods: Sequence[str], *, v1_raw: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    selector = {
        (str(row["model"]), int(row["replicate"]), str(row["arm"]), int(row["budget"]), str(row["candidate_id"])): float(row["score"])
        for row in raw["selector_rows"]
    }
    if v1_raw is not None:
        selector.update(
            {
                (str(row["model"]), int(row["replicate"]), str(row["arm"]), int(row["budget"]), "V1"): float(row["score"])
                for row in v1_raw["selector_rows"]
                if row.get("candidate_id") == "S"
            }
        )
    references = {
        (str(row["backbone"]), int(row["replicate"]), str(row["arm"]), str(row["head"])): float(row["test_accuracy"])
        for row in raw["reference_rows"]
    }
    replicates = sorted({int(row["replicate"]) for row in raw["selector_rows"]})
    budgets = sorted({int(row["budget"]) for row in raw["selector_rows"]})
    output: list[dict[str, Any]] = []
    for replicate in replicates:
        for arm in experiment.ARMS:
            for budget in budgets:
                for head in HEADS:
                    outcomes = np.asarray([references[(model, replicate, arm, head)] for model in experiment.MODELS])
                    for method in methods:
                        scores = np.asarray([selector[(model, replicate, arm, budget, method)] for model in experiment.MODELS])
                        output.append({"replicate": replicate, "arm": arm, "budget": budget, "head": head, "candidate_id": method, **_selection_metrics(scores, outcomes)})
    return output


def _bootstrap(values: Sequence[float], *, key: str, statistic: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise RuntimeError(f"bootstrap {key} needs at least two finite blocks")
    offset = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, offset]))
    draws = rng.integers(0, array.size, size=(BOOTSTRAP_RESAMPLES, array.size))
    sampled = array[draws]
    if statistic == "median":
        distribution = np.median(sampled, axis=1)
        estimate = float(np.median(array))
    elif statistic == "mean":
        distribution = np.mean(sampled, axis=1)
        estimate = float(np.mean(array))
    else:
        raise ValueError("unknown bootstrap statistic")
    return {
        "estimate": estimate,
        "lower95": float(np.quantile(distribution, 0.025)),
        "upper95": float(np.quantile(distribution, 0.975)),
        "block_values": array.tolist(),
        "resamples": BOOTSTRAP_RESAMPLES,
    }


def _metric_lookup(metrics: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str, int, str, str], float]:
    return {
        (int(row["replicate"]), str(row["arm"]), int(row["budget"]), str(row["head"]), str(row["candidate_id"])): float(row["regret_pp"])
        for row in metrics
    }


def _development_candidate_gates(raw: Mapping[str, Any], metrics: Sequence[Mapping[str, Any]], candidate: str) -> list[dict[str, Any]]:
    lookup = _metric_lookup(metrics)
    replicate = experiment.DEVELOPMENT_REPLICATE
    budget = experiment.DEVELOPMENT_BUDGET
    rows: list[dict[str, Any]] = []
    for arm, head, comparator, name in (
        ("baseline", "linear", "LP-FULL", "clean_linear"),
        ("nuisance_full", "linear", "LP-FULL", "nuisance_linear"),
        *(("nonlinearity_full", head, "V1", f"nonlinear_retention_{head}") for head in NONLINEAR_HEADS),
    ):
        delta = lookup[(replicate, arm, budget, head, candidate)] - lookup[(replicate, arm, budget, head, comparator)]
        rows.append({"gate": name, "arm": arm, "head": head, "comparator": comparator, "estimate_pp": delta, "threshold_pp": 1.0, "status": "pass" if delta <= 1.0 else "fail"})
    wall = {
        (str(row["model"]), str(row["arm"]), str(row["candidate_id"])): float(row["total_wall_seconds"])
        for row in raw["selector_rows"]
    }
    for arm in experiment.ARMS:
        ratios = [wall[(model, arm, candidate)] / wall[(model, arm, "LP-FULL")] for model in experiment.MODELS]
        interval = _bootstrap(ratios, key=f"dev-runtime:{candidate}:{arm}", statistic="median")
        rows.append({"gate": "runtime", "arm": arm, "head": None, "comparator": "LP-FULL", **interval, "threshold_median": 1.0, "threshold_upper95": 1.10, "status": "pass" if interval["estimate"] <= 1.0 and interval["upper95"] <= 1.10 else "fail"})
    return rows


def write_development_analysis(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("refusing to overwrite development analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = _verify_artifact(Path(input_dir), "development")
    metrics = _metrics(raw, experiment.DEVELOPMENT_METHODS)
    candidate_rows = {
        candidate: _development_candidate_gates(raw, metrics, candidate)
        for candidate in experiment.PROMOTION_PRECEDENCE
    }
    eligible = [
        candidate
        for candidate in experiment.PROMOTION_PRECEDENCE
        if all(row["status"] == "pass" for row in candidate_rows[candidate])
    ]
    selected = eligible[0] if eligible else None
    protocol_sha, _ = experiment.validate_protocol()
    sources = experiment.source_identity()
    raw_sha = experiment.sha256_path(Path(input_dir) / "raw_results.json")
    manifest_sha = experiment.sha256_path(Path(input_dir) / "manifest.json")
    lock_body = {
        "schema_version": 1,
        "status": "locked_for_full_evaluation" if selected else "stopped_no_eligible",
        "selected_candidate": selected,
        "eligible_candidates": eligible,
        "promotion_precedence": list(experiment.PROMOTION_PRECEDENCE),
        "runner_up_reselection_allowed": False,
        "protocol_sha256": protocol_sha,
        "source_identity": sources,
        "development_raw_sha256": raw_sha,
        "development_manifest_sha256": manifest_sha,
        "candidate_gates": candidate_rows,
        "retrospective_development_only": True,
    }
    lock = dict(lock_body)
    lock["lock_sha256"] = hashlib.sha256(experiment.canonical_json(lock_body).encode("utf-8")).hexdigest()
    summary = {
        "schema_version": 1,
        "artifact_status": "completed",
        "stage": "development",
        "selector_metrics": metrics,
        "candidate_gates": candidate_rows,
        "decision": lock,
    }
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_json(output_dir / "development_lock.json", lock)
    report = [
        "# Speed-focused Food-101 development screen",
        "",
        "Retrospective 30-panel lock screen. No untouched confirmation claim.",
        "",
        f"Decision: **{lock['status']}**; selected candidate: **{selected or 'none'}**.",
        "",
        "| Candidate | Eligible | Failed gates |",
        "|---|---:|---|",
    ]
    for candidate in experiment.PROMOTION_PRECEDENCE:
        failed = [f"{row['gate']}:{row.get('arm')}" for row in candidate_rows[candidate] if row["status"] != "pass"]
        report.append(f"| {candidate} | {'yes' if candidate in eligible else 'no'} | {', '.join(failed) or '—'} |")
    _atomic_text(output_dir / "report.md", "\n".join(report) + "\n")
    _atomic_json(
        output_dir / "manifest.json",
        {
            "artifact_status": "completed",
            "stage": "development",
            "summary_sha256": experiment.sha256_path(output_dir / "summary.json"),
            "lock_sha256": experiment.sha256_path(output_dir / "development_lock.json"),
            "report_sha256": experiment.sha256_path(output_dir / "report.md"),
            "input_raw_sha256": raw_sha,
        },
    )
    return summary


def _load_lock(path: Path) -> dict[str, Any]:
    protocol_sha, _ = experiment.validate_protocol()
    return experiment._validate_lock(Path(path), protocol_sha, experiment.source_identity())


def _full_regret_contrast(metrics: Sequence[Mapping[str, Any]], arm: str, head: str, candidate: str, comparator: str) -> dict[str, Any]:
    lookup = _metric_lookup(metrics)
    values = [
        float(np.mean([lookup[(replicate, arm, budget, head, candidate)] - lookup[(replicate, arm, budget, head, comparator)] for budget in experiment.BUDGETS]))
        for replicate in experiment.REPLICATES
    ]
    return {"arm": arm, "head": head, "candidate_id": candidate, "comparator": comparator, **_bootstrap(values, key=f"full-regret:{candidate}:{comparator}:{arm}:{head}", statistic="mean")}


def _full_runtime(raw: Mapping[str, Any], candidate: str, arm: str) -> dict[str, Any]:
    wall = {
        (str(row["model"]), int(row["replicate"]), str(row["arm"]), int(row["budget"]), str(row["candidate_id"])): float(row["total_wall_seconds"])
        for row in raw["selector_rows"]
    }
    values = [
        float(np.median([wall[(model, replicate, arm, budget, candidate)] / wall[(model, replicate, arm, budget, "LP-FULL")] for model in experiment.MODELS for budget in experiment.BUDGETS]))
        for replicate in experiment.REPLICATES
    ]
    return {"arm": arm, "candidate_id": candidate, "comparator": "LP-FULL", **_bootstrap(values, key=f"full-runtime:{candidate}:{arm}", statistic="median")}


def _aggregate(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for arm in experiment.ARMS:
        for head in HEADS:
            for candidate in sorted({str(row["candidate_id"]) for row in metrics}):
                rows = [row for row in metrics if row["arm"] == arm and row["head"] == head and row["candidate_id"] == candidate]
                rho = [float(row["spearman"]) for row in rows if row["spearman"] is not None]
                output.append({"arm": arm, "head": head, "candidate_id": candidate, "mean_regret_pp": float(np.mean([row["regret_pp"] for row in rows])), "mean_spearman": float(np.mean(rho)) if rho else None, "exact_best_rate": float(np.mean([row["exact_best"] for row in rows])), "within_one_pp_rate": float(np.mean([row["within_one_pp"] for row in rows]))})
    return output


def write_full_analysis(input_dir: Path, lock_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("refusing to overwrite full speed analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = _load_lock(lock_path)
    raw = _verify_artifact(Path(input_dir), "full", lock)
    v1_raw_path = experiment.V1_ARTIFACT_DIR / "raw_results.json"
    v1_raw = _read_json(v1_raw_path)
    if experiment.sha256_path(v1_raw_path) != raw["run_identity"]["source_identity"]["v1_food_raw_sha256"]:
        raise RuntimeError("v1 comparison artifact changed")
    candidate = str(lock["selected_candidate"])
    methods = (candidate, "LP-FULL", "V1")
    metrics = _metrics(raw, methods, v1_raw=v1_raw)
    contrasts = [
        _full_regret_contrast(metrics, "baseline", "linear", candidate, "LP-FULL"),
        _full_regret_contrast(metrics, "nuisance_full", "linear", candidate, "LP-FULL"),
    ]
    for head in NONLINEAR_HEADS:
        contrasts.append(_full_regret_contrast(metrics, "nonlinearity_full", head, candidate, "LP-FULL"))
        contrasts.append(_full_regret_contrast(metrics, "nonlinearity_full", head, candidate, "V1"))
    runtime = [_full_runtime(raw, candidate, arm) for arm in experiment.ARMS]
    gates: list[dict[str, Any]] = []
    for row in contrasts:
        gate = "nonlinear_retention" if row["comparator"] == "V1" else ("nonlinear_vs_probe" if row["arm"] == "nonlinearity_full" else f"{row['arm']}_linear_vs_probe")
        gates.append({"gate": gate, "arm": row["arm"], "head": row["head"], "comparator": row["comparator"], "estimate_pp": row["estimate"], "upper95_pp": row["upper95"], "threshold_upper95_pp": 1.0, "status": "pass" if row["upper95"] <= 1.0 else "fail", "required": True})
    for row in runtime:
        gates.append({"gate": "runtime", "arm": row["arm"], "head": None, "comparator": "LP-FULL", "estimate": row["estimate"], "upper95": row["upper95"], "threshold_median": 1.0, "threshold_upper95": 1.10, "status": "pass" if row["estimate"] <= 1.0 and row["upper95"] <= 1.10 else "fail", "required": True})
    gates.append({"gate": "determinism", "status": "pass", "required": True, "methods": sorted(raw["determinism"])})
    failed = [row for row in gates if row["required"] and row["status"] != "pass"]
    decision = {"status": "pass" if not failed else "fail", "selected_candidate": candidate, "runner_up_reselection_allowed": False, "failed_required_gates": [{"gate": row["gate"], "arm": row.get("arm"), "head": row.get("head")} for row in failed], "retrospective_development_only": True, "confirmation_status": "not_run_not_untouched"}
    summary = {"schema_version": 1, "artifact_status": "completed", "stage": "full", "selector_metrics": metrics, "aggregate": _aggregate(metrics), "regret_contrasts": contrasts, "runtime_contrasts": runtime, "gates": gates, "decision": decision, "development_lock": lock, "input_raw_sha256": experiment.sha256_path(Path(input_dir) / "raw_results.json"), "v1_raw_sha256": experiment.sha256_path(v1_raw_path)}
    _atomic_json(output_dir / "summary.json", summary)
    primary = {("baseline", "linear"), ("nuisance_full", "linear"), *(("nonlinearity_full", head) for head in NONLINEAR_HEADS)}
    report = ["# Speed-focused scalable relevance Food-101 evaluation", "", "Retrospective full replay of the development-locked candidate; no runner-up reselection and no untouched confirmation.", "", f"Decision: **{decision['status'].upper()}** for candidate **{candidate}**.", "", "## Ranking", "", "| Arm | Head | Method | Mean regret pp | Spearman |", "|---|---|---|---:|---:|"]
    for row in summary["aggregate"]:
        if (row["arm"], row["head"]) in primary:
            rho = "—" if row["mean_spearman"] is None else f"{row['mean_spearman']:.3f}"
            report.append(f"| {row['arm']} | {row['head']} | {row['candidate_id']} | {row['mean_regret_pp']:.3f} | {rho} |")
    report.extend(["", "## Runtime", "", "| Arm | Median candidate / LP | Upper 95 | Status |", "|---|---:|---:|---|"])
    for row in runtime:
        gate = next(item for item in gates if item["gate"] == "runtime" and item["arm"] == row["arm"])
        report.append(f"| {row['arm']} | {row['estimate']:.3f} | {row['upper95']:.3f} | {gate['status']} |")
    report.extend(["", "## Takeaway", "", "The locked implementation met both ranking and speed gates." if not failed else "The locked implementation did not meet every frozen ranking/runtime gate; no alternative was selected after seeing full outcomes.", ""])
    _atomic_text(output_dir / "report.md", "\n".join(report))
    _atomic_json(output_dir / "manifest.json", {"artifact_status": "completed", "stage": "full", "summary_sha256": experiment.sha256_path(output_dir / "summary.json"), "report_sha256": experiment.sha256_path(output_dir / "report.md"), "input_raw_sha256": summary["input_raw_sha256"], "development_lock_sha256": lock["lock_sha256"], "decision": decision})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    development = sub.add_parser("development")
    development.add_argument("--input", type=Path, required=True)
    development.add_argument("--output", type=Path, required=True)
    full = sub.add_parser("full")
    full.add_argument("--input", type=Path, required=True)
    full.add_argument("--development-lock", type=Path, required=True)
    full.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "development":
        result = write_development_analysis(args.input, args.output)
    else:
        result = write_full_analysis(args.input, args.development_lock, args.output)
    print(experiment.canonical_json(result["decision"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
