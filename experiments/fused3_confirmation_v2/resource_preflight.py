"""Outcome-blind fresh-process resource preflight for confirmation V2."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.fused3_confirmation_v2 import recipes, runner


ARTIFACT_STATUS = "completed_outcome_blind_resource_preflight"
STUDY = "fused3_backbone_ranking_confirmation_v2"
WORKER_DIRNAME = "worker_smoke"
TERMINAL_FILES = frozenset({"resource_preflight.json", "manifest.json"})
SMOKE_FILES = frozenset(
    {"audited_registry.json", "lineage_lock.json", "raw_results.json", "manifest.json"}
)
FORBIDDEN_OUTCOME_KEYS = frozenset(
    {
        "score",
        "test_accuracy",
        "folds",
        "total_wall_seconds",
        "total_cpu_seconds",
        "fit_wall_seconds",
        "fit_cpu_seconds",
        "score_fixed_wall_seconds",
        "score_fixed_cpu_seconds",
    }
)
RAW_SMOKE_KEYS = frozenset(
    {
        "artifact_status",
        "audited_registry",
        "audited_registry_sha256",
        "code_identity_sha256",
        "determinism_verification",
        "environment",
        "lineage_lock_sha256",
        "mode",
        "outcomes_redacted",
        "panel",
        "protocol",
        "protocol_sha256",
        "ranking_available",
        "repository_provenance",
        "run_identity_sha256",
        "schema_version",
        "source_hashes",
        "study",
        "thread_environment",
    }
)
MANIFEST_SMOKE_KEYS = frozenset(
    {
        "artifact_status",
        "audited_registry",
        "audited_registry_sha256",
        "code_identity_sha256",
        "environment",
        "lineage_lock_sha256",
        "mode",
        "outcomes_redacted",
        "protocol",
        "protocol_sha256",
        "raw_results_sha256",
        "repository_provenance",
        "run_identity_sha256",
        "schema_version",
        "source_hashes",
        "study",
        "thread_environment",
    }
)
PANEL_KEYS = frozenset(
    {
        "artifact_status",
        "panel_identity",
        "reference_rows",
        "schema_version",
        "selector_rows",
    }
)
SELECTOR_DESCRIPTOR_KEYS = frozenset(
    {
        "backbone",
        "budget",
        "candidate_id",
        "candidate_recipe_sha256",
        "dataset_id",
        "error",
        "evaluation_cohort_sha256",
        "execution_order",
        "execution_position",
        "outcomes_redacted",
        "panel_id",
        "replicate",
        "replicate_seed",
        "score_structure",
        "status",
        "training_cohort_sha256",
        "warmup_excluded",
    }
)
REFERENCE_DESCRIPTOR_KEYS = frozenset(
    {
        "accuracy_structure",
        "backbone",
        "budget",
        "dataset_id",
        "error",
        "evaluation_cohort_sha256",
        "evaluation_row_count",
        "head",
        "outcomes_redacted",
        "panel_id",
        "recipe_sha256",
        "replicate",
        "replicate_seed",
        "status",
        "training_cohort_sha256",
        "training_row_count",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object at {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(runner.canonical_json(dict(value)) + "\n", encoding="utf-8")
    temporary.replace(path)


def _exact_files(root: Path, expected: frozenset[str]) -> None:
    observed = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed != expected:
        raise RuntimeError(
            f"artifact file set mismatch: expected {sorted(expected)!r}, got {sorted(observed)!r}"
        )


def _scan_outcome_keys(value: Any, *, path: tuple[Any, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in FORBIDDEN_OUTCOME_KEYS:
                raise RuntimeError(f"persisted outcome-derived field at {path + (key,)!r}")
            _scan_outcome_keys(item, path=path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_outcome_keys(item, path=path + (index,))


def _validate_redacted_smoke(
    smoke_root: Path,
    *,
    registry_path: Path,
    lineage_lock_path: Path,
) -> dict[str, Any]:
    """Validate one canonical smoke without reading a numerical outcome."""

    _exact_files(smoke_root, SMOKE_FILES)
    raw_path = smoke_root / "raw_results.json"
    manifest_path = smoke_root / "manifest.json"
    raw = _read_json(raw_path)
    manifest = _read_json(manifest_path)
    if set(raw) != RAW_SMOKE_KEYS or set(manifest) != MANIFEST_SMOKE_KEYS:
        raise RuntimeError("smoke top-level schema mismatch")
    protocol_sha, protocol = runner.validate_protocol(require_frozen=True)
    sources = runner.source_identity()
    registry = _read_json(registry_path)
    registry_sha = runner.sha256_path(registry_path)
    lock_sha = runner.sha256_path(lineage_lock_path)
    if runner.sha256_path(smoke_root / "audited_registry.json") != registry_sha:
        raise RuntimeError("smoke registry copy differs from the supplied audited registry")
    if runner.sha256_path(smoke_root / "lineage_lock.json") != lock_sha:
        raise RuntimeError("smoke lineage-lock copy differs from the supplied lock")
    if manifest.get("raw_results_sha256") != runner.sha256_path(raw_path):
        raise RuntimeError("smoke raw-results hash mismatch")
    registry_descriptor = {
        "path": "audited_registry.json",
        "sha256": registry_sha,
        "registry_id": registry.get("registry_id"),
    }
    identity_payload = {
        "study": STUDY,
        "mode": "smoke",
        "protocol_sha256": protocol_sha,
        "code_identity_sha256": sources["code_identity_sha256"],
        "lineage_lock_sha256": lock_sha,
        "audited_registry_sha256": registry_sha,
        "environment": dict(runner.EXPECTED_SCORE_ENVIRONMENT),
        "thread_environment": dict(runner.THREAD_ENVIRONMENT),
    }
    run_identity = runner.payload_sha256(identity_payload)
    for value in (raw, manifest):
        if value.get("schema_version") != 1 or value.get("study") != STUDY:
            raise RuntimeError("smoke study/schema identity mismatch")
        if value.get("artifact_status") != "completed_structural_smoke":
            raise RuntimeError("resource preflight requires a completed structural smoke")
        if value.get("mode") != "smoke" or value.get("outcomes_redacted") is not True:
            raise RuntimeError("smoke is not outcome redacted")
        if value.get("protocol_sha256") != protocol_sha or value.get("protocol") != protocol:
            raise RuntimeError("smoke protocol differs from the current frozen protocol")
        if value.get("code_identity_sha256") != sources["code_identity_sha256"]:
            raise RuntimeError("smoke code identity differs from current sources")
        if value.get("source_hashes") != sources["source_hashes"]:
            raise RuntimeError("smoke source hashes differ from current sources")
        if value.get("repository_provenance") != sources["repository"]:
            raise RuntimeError("smoke repository provenance differs from current sources")
        if value.get("audited_registry_sha256") != registry_sha:
            raise RuntimeError("smoke registry identity mismatch")
        if value.get("audited_registry") != registry_descriptor:
            raise RuntimeError("smoke registry descriptor differs from its hash-bound input")
        if value.get("lineage_lock_sha256") != lock_sha:
            raise RuntimeError("smoke lineage identity mismatch")
        if value.get("thread_environment") != runner.THREAD_ENVIRONMENT:
            raise RuntimeError("smoke thread environment mismatch")
        if value.get("environment") != runner.EXPECTED_SCORE_ENVIRONMENT:
            raise RuntimeError("smoke score environment mismatch")
        if value.get("run_identity_sha256") != run_identity:
            raise RuntimeError("smoke run identity mismatch")
    if raw.get("ranking_available") is not False:
        raise RuntimeError("smoke unexpectedly exposes ranking evidence")
    panel = raw.get("panel")
    if not isinstance(panel, Mapping) or set(panel) != PANEL_KEYS:
        raise RuntimeError("smoke panel descriptor is missing")
    if panel.get("artifact_status") != "completed_structural_smoke_panel":
        raise RuntimeError("smoke panel descriptor status mismatch")
    first_identity = runner.panel_identities()[0]
    if panel.get("panel_identity") != first_identity:
        raise RuntimeError("smoke does not use the first frozen panel")
    selectors = panel.get("selector_rows")
    references = panel.get("reference_rows")
    if not isinstance(selectors, list) or len(selectors) != len(recipes.SELECTORS):
        raise RuntimeError("smoke selector descriptors are incomplete")
    if not isinstance(references, list) or len(references) != len(recipes.HEADS):
        raise RuntimeError("smoke reference descriptors are incomplete")
    if {row.get("candidate_id") for row in selectors if isinstance(row, Mapping)} != set(
        recipes.SELECTORS
    ):
        raise RuntimeError("smoke selector descriptor IDs are incomplete")
    if {row.get("head") for row in references if isinstance(row, Mapping)} != set(recipes.HEADS):
        raise RuntimeError("smoke reference descriptor IDs are incomplete")
    for row in selectors:
        if (
            not isinstance(row, Mapping)
            or set(row) != SELECTOR_DESCRIPTOR_KEYS
            or row.get("outcomes_redacted") is not True
        ):
            raise RuntimeError("smoke selector descriptor is malformed")
        descriptor = row.get("score_structure")
        if descriptor != {"present": True, "type": "float", "finite": True}:
            raise RuntimeError("smoke selector score descriptor is malformed")
    for row in references:
        if (
            not isinstance(row, Mapping)
            or set(row) != REFERENCE_DESCRIPTOR_KEYS
            or row.get("outcomes_redacted") is not True
        ):
            raise RuntimeError("smoke reference descriptor is malformed")
        descriptor = row.get("accuracy_structure")
        if descriptor != {"present": True, "type": "float", "finite": True}:
            raise RuntimeError("smoke reference accuracy descriptor is malformed")
    determinism = runner._validate_determinism(raw.get("determinism_verification", {}))
    by_selector = {str(row["candidate_id"]): row for row in selectors}
    by_head = {str(row["head"]): row for row in references}
    for identity, descriptor in determinism.items():
        if identity.startswith("HEAD:"):
            row = by_head[identity.removeprefix("HEAD:")]
            observed = runner._structural_signature(row, row_kind="reference")
        else:
            row = by_selector[identity]
            observed = runner._structural_signature(row, row_kind="selector")
        if descriptor["first_signature_sha256"] != observed:
            raise RuntimeError("smoke structural determinism hash does not match its visible row")
    for key, value in raw.items():
        if key != "protocol":
            _scan_outcome_keys(value, path=(key,))
    for key, value in manifest.items():
        if key != "protocol":
            _scan_outcome_keys(value, path=(key,))
    return {
        "raw_results_sha256": runner.sha256_path(raw_path),
        "manifest_sha256": runner.sha256_path(manifest_path),
        "run_identity_sha256": run_identity,
    }


def _parse_peak_rss(stderr: str) -> int:
    matches = re.findall(r"^\s*(\d+)\s+maximum resident set size\s*$", stderr, flags=re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError("could not parse one child maximum resident set size")
    result = int(matches[0])
    if result <= 0:
        raise RuntimeError("child peak RSS must be positive")
    return result


def _execute_fresh_child(
    *, registry_path: Path, lineage_lock_path: Path, worker_output: Path
) -> dict[str, Any]:
    if platform.system() != "Darwin" or not Path("/usr/bin/time").is_file():
        raise RuntimeError("resource preflight is frozen to macOS /usr/bin/time -l")
    command = [
        "/usr/bin/time",
        "-l",
        sys.executable,
        "-m",
        "experiments.fused3_confirmation_v2.runner",
        "--registry",
        str(registry_path),
        "--lineage-lock",
        str(lineage_lock_path),
        "--output",
        str(worker_output),
        "--smoke",
    ]
    environment = dict(os.environ)
    environment.update(runner.THREAD_ENVIRONMENT)
    environment["PYTHONPATH"] = str(runner.ROOT)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=runner.ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    wall = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if completed.returncode != 0:
        raise RuntimeError("fresh structural-smoke resource child failed")
    user_cpu = float(after.ru_utime - before.ru_utime)
    system_cpu = float(after.ru_stime - before.ru_stime)
    peak_rss = _parse_peak_rss(completed.stderr)
    values = (wall, user_cpu, system_cpu)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise RuntimeError("fresh child resource clocks must be finite and nonnegative")
    return {
        "scope": "fresh_child_full_structural_smoke_including_imports_validation_excluded_warmup_and_measured_call",
        "wall_seconds": float(wall),
        "user_cpu_seconds": user_cpu,
        "system_cpu_seconds": system_cpu,
        "peak_rss_bytes": peak_rss,
        "peak_rss_recipe": "macos_usr_bin_time_l_maximum_resident_set_size_bytes",
        "warmup_included": True,
        "outcomes_persisted": False,
        "descriptive_only": True,
        "promotion_veto": False,
    }


def run_resource_preflight(
    *,
    smoke_path: Path,
    registry_path: Path,
    lineage_lock_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Run and seal the distinct outcome-blind resource-preflight stage."""

    if output.exists() and any(output.iterdir()):
        raise FileExistsError("resource preflight output exists and will not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    prior_smoke = _validate_redacted_smoke(
        smoke_path,
        registry_path=registry_path,
        lineage_lock_path=lineage_lock_path,
    )
    worker_output = output / WORKER_DIRNAME
    resources = _execute_fresh_child(
        registry_path=registry_path,
        lineage_lock_path=lineage_lock_path,
        worker_output=worker_output,
    )
    worker_smoke = _validate_redacted_smoke(
        worker_output,
        registry_path=registry_path,
        lineage_lock_path=lineage_lock_path,
    )
    protocol_sha, protocol = runner.validate_protocol(require_frozen=True)
    sources = runner.source_identity()
    shared = {
        "schema_version": 1,
        "study": STUDY,
        "artifact_status": ARTIFACT_STATUS,
        "mode": "resource_preflight",
        "protocol_sha256": protocol_sha,
        "code_identity_sha256": sources["code_identity_sha256"],
        "lineage_lock_sha256": runner.sha256_path(lineage_lock_path),
        "audited_registry_sha256": runner.sha256_path(registry_path),
        "source_hashes": sources["source_hashes"],
        "repository_provenance": sources["repository"],
        "thread_environment": dict(runner.THREAD_ENVIRONMENT),
        "environment": dict(runner.EXPECTED_SCORE_ENVIRONMENT),
        "protocol": protocol,
        "prior_smoke": prior_smoke,
        "worker_smoke": worker_smoke,
        "resources": resources,
        "outcomes_redacted": True,
        "ranking_available": False,
        "status": "pass",
    }
    preflight_path = output / "resource_preflight.json"
    _atomic_json(preflight_path, shared)
    worker_files = {
        str(path.relative_to(output)): runner.sha256_path(path)
        for path in sorted(worker_output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        **shared,
        "resource_preflight_sha256": runner.sha256_path(preflight_path),
        "worker_file_manifest": worker_files,
    }
    _atomic_json(output / "manifest.json", manifest)
    expected = frozenset({*TERMINAL_FILES, *(f"{WORKER_DIRNAME}/{name}" for name in SMOKE_FILES)})
    _exact_files(output, expected)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--lineage-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run_resource_preflight(
        smoke_path=args.smoke,
        registry_path=args.registry,
        lineage_lock_path=args.lineage_lock,
        output=args.output,
    )
    print(
        runner.canonical_json(
            {
                "artifact_status": manifest["artifact_status"],
                "manifest_sha256": runner.sha256_path(args.output / "manifest.json"),
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["run_resource_preflight"]
