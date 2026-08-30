"""Fail-closed analysis for the isolated FUSED3 confirmation V2 artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import datasets, lineage, recipes, runner, statistics


PACKAGE_DIR = Path(__file__).resolve().parent
CANDIDATE_AUTHORITY_PATH = PACKAGE_DIR / "candidate_authority.json"
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SIDECAR = PACKAGE_DIR / "protocol.sha256"
LINEAGE_LOCK_FILENAME = "lineage_lock.json"
AUDITED_REGISTRY_FILENAME = "audited_registry.json"
TABLE_FILES = {
    "selector_rows.jsonl": "selector_rows",
    "reference_rows.jsonl": "reference_rows",
}
SHARED_METADATA_FIELDS = (
    "schema_version",
    "study",
    "artifact_status",
    "mode",
    "protocol_sha256",
    "code_identity_sha256",
    "run_identity_sha256",
    "audited_registry_sha256",
    "lineage_lock_sha256",
    "environment",
    "thread_environment",
    "repository_provenance",
    "source_hashes",
    "audited_registry",
    "grid",
    "table_manifest",
    "checkpoint_count",
    "determinism_verification",
    "runtime_gate",
    "protocol",
)
EXPECTED_SCORE_ENVIRONMENT = {
    "python": "3.12.2",
    "numpy": "2.1.3",
    "sklearn": "1.6.1",
}
EXPECTED_THREAD_ENVIRONMENT = {
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "LOKY_MAX_CPU_COUNT": "1",
}
CLASS_COUNTS = {
    "torchvision_cifar10": 10,
    "torchvision_stl10": 10,
    "torchvision_gtsrb": 43,
    "torchvision_fgvc_aircraft": 100,
    "torchvision_dtd": 47,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _read_canonical_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    try:
        payload = path.read_bytes()
        text = payload.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"could not read canonical JSONL table: {path}") from exc
    rows: list[dict[str, Any]] = []
    rebuilt: list[str] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise ValueError(f"blank JSONL line at {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSONL row at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(row)
        rebuilt.append(canonical_json(row) + "\n")
    if "".join(rebuilt).encode("utf-8") != payload:
        raise ValueError(f"table is not canonical compact JSONL: {path}")
    return rows, payload


def _expected_grid() -> dict[str, Any]:
    return {
        "datasets": list(statistics.DATASET_IDS),
        "backbones": list(statistics.BACKBONES),
        "replicate_seeds": list(statistics.REPLICATE_SEEDS),
        "budgets": list(statistics.BUDGETS),
        "selectors": list(statistics.METHODS),
        "heads": list(statistics.HEADS),
        "panel_count": 500,
    }


def _finite_nonnegative(value: Any, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{field} must be a finite nonnegative number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be a finite nonnegative number")
    return result


def _require_sha(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not _is_sha256(value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return str(value)


def _validate_determinism(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("determinism_verification must be a mapping")
    expected = {*statistics.METHODS, *(f"HEAD:{head}" for head in statistics.HEADS)}
    if set(value) != expected:
        raise ValueError("determinism_verification has an incomplete method/head set")
    first_panel_id = (
        f"{statistics.DATASET_IDS[0]}__{statistics.BACKBONES[0]}__r0__b{statistics.BUDGETS[0]}"
    )
    for identity, descriptor in value.items():
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"determinism descriptor for {identity!r} is malformed")
        if set(descriptor) != {
            "first_signature_sha256",
            "second_signature_sha256",
            "exact",
            "runtime_fields_excluded",
            "panel_id",
        }:
            raise ValueError(f"determinism descriptor for {identity!r} has invalid keys")
        if descriptor.get("exact") is not True or descriptor.get("runtime_fields_excluded") is not True:
            raise ValueError(f"determinism descriptor for {identity!r} is not exact")
        first = descriptor.get("first_signature_sha256")
        second = descriptor.get("second_signature_sha256")
        if not _is_sha256(first) or first != second:
            raise ValueError(f"determinism signatures for {identity!r} do not match")
        if descriptor.get("panel_id") != first_panel_id:
            raise ValueError(f"determinism descriptor for {identity!r} is not on the first frozen panel")


def _panel_id(dataset: str, backbone: str, replicate: int, budget: int) -> str:
    return f"{dataset}__{backbone}__r{replicate}__b{budget}"


def _validate_selector_folds(
    folds: Any,
    *,
    candidate: str,
    replicate_seed: int,
    expected_rows: int,
) -> None:
    if not isinstance(folds, list) or len(folds) != 5 or any(
        not isinstance(fold, Mapping) for fold in folds
    ):
        raise ValueError("selector row must carry exactly five fold descriptors")
    if [fold.get("fold") for fold in folds] != list(range(5)):
        raise ValueError("selector fold indices must be exactly 0..4")
    for fold_index, fold in enumerate(folds):
        expected_seed = replicate_seed + fold_index if candidate == "FUSED3" else replicate_seed
        if type(fold.get("seed")) is not int or fold.get("seed") != expected_seed:
            raise ValueError("selector fold seed does not match the frozen recipe")
        if candidate == "LP-FULL" and (
            fold.get("split_seed") != replicate_seed
            or fold.get("model_random_state") != replicate_seed
        ):
            raise ValueError("LP-FULL fold seed identity does not match the frozen recipe")
        train_size = fold.get("train_size")
        holdout_size = fold.get("holdout_size")
        if (
            type(train_size) is not int
            or type(holdout_size) is not int
            or train_size <= 0
            or holdout_size <= 0
            or train_size + holdout_size != expected_rows
        ):
            raise ValueError("selector fold train/holdout sizes are invalid")
        _finite_nonnegative(fold.get("score"), field="fold score")
        _require_sha(fold, "candidate_config_sha256")
        for name in (
            "fit_wall_seconds",
            "fit_cpu_seconds",
        ):
            _finite_nonnegative(fold.get(name), field=f"fold {name}")
        if candidate == "FUSED3":
            for name in ("score_fixed_wall_seconds", "score_fixed_cpu_seconds"):
                _finite_nonnegative(fold.get(name), field=f"fold {name}")
            _require_sha(fold, "numerical_state_sha256")
            if fold.get("state_unchanged_after_score_fixed") is not True:
                raise ValueError("FUSED3 fold reports fitted-state mutation")
        else:
            for name in ("predict_wall_seconds", "predict_cpu_seconds"):
                _finite_nonnegative(fold.get(name), field=f"fold {name}")


def _validate_selector_structure(rows: Sequence[Mapping[str, Any]]) -> None:
    recipe_hashes = {
        candidate: recipes.payload_sha256(recipes.candidate_recipe(candidate))
        for candidate in statistics.METHODS
    }
    panel_position = {
        (dataset, backbone, replicate, budget): position
        for position, (dataset, backbone, replicate, budget) in enumerate(
            (dataset, backbone, replicate, budget)
            for dataset in statistics.DATASET_IDS
            for backbone in statistics.BACKBONES
            for replicate in range(len(statistics.REPLICATE_SEEDS))
            for budget in statistics.BUDGETS
        )
    }
    for row in rows:
        if set(row) != {
            "panel_id",
            "dataset_id",
            "backbone",
            "replicate",
            "replicate_seed",
            "budget",
            "candidate_id",
            "score",
            "total_wall_seconds",
            "total_cpu_seconds",
            "fit_wall_seconds",
            "fit_cpu_seconds",
            "score_fixed_wall_seconds",
            "score_fixed_cpu_seconds",
            "execution_position",
            "execution_order",
            "warmup_excluded",
            "candidate_recipe_sha256",
            "training_cohort_sha256",
            "evaluation_cohort_sha256",
            "folds",
            "status",
            "error",
        }:
            raise ValueError("selector row has unknown or missing fields")
        if row.get("error") is not None or row.get("warmup_excluded") is not True:
            raise ValueError("selector row must be successful and exclude warmup")
        candidate = str(row.get("candidate_id"))
        if row.get("candidate_recipe_sha256") != recipe_hashes.get(candidate):
            raise ValueError("selector candidate recipe hash mismatch")
        for field in (
            "score",
            "total_wall_seconds",
            "total_cpu_seconds",
            "fit_wall_seconds",
            "fit_cpu_seconds",
            "score_fixed_wall_seconds",
            "score_fixed_cpu_seconds",
        ):
            value = _finite_nonnegative(row.get(field), field=f"selector {field}")
            if field == "score" and not 0.0 <= value <= 1.0:
                raise ValueError("selector score must be in [0, 1]")
        _require_sha(row, "training_cohort_sha256")
        _require_sha(row, "evaluation_cohort_sha256")
        dataset = str(row.get("dataset_id"))
        budget = int(row.get("budget", -1))
        _validate_selector_folds(
            row.get("folds"),
            candidate=candidate,
            replicate_seed=int(row.get("replicate_seed", -1)),
            expected_rows=CLASS_COUNTS.get(dataset, -1) * budget,
        )
        key = (
            dataset,
            str(row.get("backbone")),
            int(row.get("replicate", -1)),
            int(row.get("budget", -1)),
        )
        if key not in panel_position:
            raise ValueError("selector row panel identity is outside the frozen grid")
        if row.get("panel_id") != _panel_id(*key):
            raise ValueError("selector panel_id does not match its frozen identity")
        order = recipes.method_order(panel_position[key])
        if row.get("execution_order") != list(order):
            raise ValueError("selector execution order does not match the frozen schedule")
        if type(row.get("execution_position")) is not int or int(row["execution_position"]) != order.index(candidate):
            raise ValueError("selector execution position does not match the frozen schedule")


def _validate_reference_structure(rows: Sequence[Mapping[str, Any]]) -> None:
    recipe_hashes = {
        head: recipes.payload_sha256(recipes.head_recipe(head))
        for head in statistics.HEADS
    }
    for row in rows:
        if set(row) != {
            "panel_id",
            "dataset_id",
            "backbone",
            "replicate",
            "replicate_seed",
            "budget",
            "head",
            "test_accuracy",
            "total_wall_seconds",
            "total_cpu_seconds",
            "training_row_count",
            "evaluation_row_count",
            "recipe_sha256",
            "training_cohort_sha256",
            "evaluation_cohort_sha256",
            "status",
            "error",
        }:
            raise ValueError("reference row has unknown or missing fields")
        if row.get("error") is not None:
            raise ValueError("reference row must be successful")
        head = str(row.get("head"))
        if row.get("recipe_sha256") != recipe_hashes.get(head):
            raise ValueError("reference-head recipe hash mismatch")
        accuracy = _finite_nonnegative(row.get("test_accuracy"), field="test_accuracy")
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError("test_accuracy must be in [0, 1]")
        for field in ("total_wall_seconds", "total_cpu_seconds"):
            _finite_nonnegative(row.get(field), field=f"reference {field}")
        _require_sha(row, "training_cohort_sha256")
        _require_sha(row, "evaluation_cohort_sha256")
        dataset = str(row.get("dataset_id"))
        budget = int(row.get("budget", -1))
        replicate = int(row.get("replicate", -1))
        backbone = str(row.get("backbone"))
        if row.get("panel_id") != _panel_id(dataset, backbone, replicate, budget):
            raise ValueError("reference panel_id does not match its frozen identity")
        if type(row.get("training_row_count")) is not int or int(row["training_row_count"]) != CLASS_COUNTS.get(dataset, -1) * budget:
            raise ValueError("reference training row count does not match class count times budget")
        if type(row.get("evaluation_row_count")) is not int or int(row["evaluation_row_count"]) != CLASS_COUNTS.get(dataset, -1) * 20:
            raise ValueError("reference evaluation row count does not match 20 rows per class")


def _validate_cohort_pairing(
    selector_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> None:
    observed: dict[tuple[str, int, int], tuple[str, str]] = {}
    evaluation_by_dataset: dict[str, str] = {}
    for row in (*selector_rows, *reference_rows):
        dataset = str(row["dataset_id"])
        key = (dataset, int(row["replicate_seed"]), int(row["budget"]))
        identity = (
            str(row["training_cohort_sha256"]),
            str(row["evaluation_cohort_sha256"]),
        )
        prior = observed.setdefault(key, identity)
        if prior != identity:
            raise ValueError("selector/reference cohort hashes differ within a frozen panel block")
        evaluation = evaluation_by_dataset.setdefault(dataset, identity[1])
        if evaluation != identity[1]:
            raise ValueError("evaluation cohort hash differs across budgets or replicates")


def _verify_local_lineage(root: Path, raw: Mapping[str, Any]) -> dict[str, Any]:
    protocol_sha = _sha256_path(PROTOCOL_PATH)
    if raw.get("protocol_sha256") != protocol_sha:
        raise ValueError("raw artifact does not bind the current frozen protocol bytes")
    sidecar_lines = PROTOCOL_SIDECAR.read_text(encoding="utf-8").splitlines()
    if sidecar_lines != [protocol_sha]:
        raise ValueError("frozen protocol SHA-256 sidecar mismatch")
    if raw.get("protocol") != _read_json(PROTOCOL_PATH):
        raise ValueError("raw protocol payload differs from the hash-bound protocol file")

    lock_path = root / LINEAGE_LOCK_FILENAME
    if raw.get("lineage_lock_sha256") != _sha256_path(lock_path):
        raise ValueError("local lineage lock SHA-256 mismatch")
    authority = _read_json(CANDIDATE_AUTHORITY_PATH)
    authority_sha = _sha256_path(CANDIDATE_AUTHORITY_PATH)
    if (
        authority.get("status") != "locked_for_untouched_confirmation"
        or authority.get("selected_candidate") != statistics.LOCKED_CANDIDATE
        or authority.get("selected_candidate_recipe_sha256")
        != recipes.payload_sha256(recipes.candidate_recipe(statistics.LOCKED_CANDIDATE))
        or authority.get("runner_up_reselection_allowed") is not False
    ):
        raise ValueError("candidate authority does not seal the exact FUSED3 recipe")
    full = authority.get("fused3_full")
    if not isinstance(full, Mapping) or (
        full.get("decision_status") != "pass_full_retrospective"
        or full.get("failed_required_gate_count") != 0
        or full.get("resource_status")
        != "passed_all_per_call_and_full_panel_gates_in_all_three_food_arms"
    ):
        raise ValueError("candidate authority does not bind the passed full-Food result")
    constraints = authority.get("confirmation_constraints")
    if not isinstance(constraints, Mapping) or (
        constraints.get("candidate_ids") != list(statistics.METHODS)
        or constraints.get("promotable_candidate_ids") != [statistics.LOCKED_CANDIDATE]
        or constraints.get("no_reselection") is not True
        or constraints.get("no_recipe_change") is not True
        or constraints.get("no_gate_change") is not True
        or constraints.get("runtime_inheritance") != "fused3_full"
        or constraints.get("confirmation_runtime") != "descriptive_only"
    ):
        raise ValueError("candidate authority confirmation constraints are invalid")
    expected_lineage = lineage.build_lineage_from_files(
        protocol_path=PROTOCOL_PATH,
        protocol_sidecar_path=PROTOCOL_SIDECAR,
        code_identity_sha256=str(raw.get("code_identity_sha256")),
    )
    lock = lineage.validate_lock_envelope(
        _read_json(lock_path),
        expected_lineage=expected_lineage,
        expected_candidate_authority_sha256=authority_sha,
    )
    if lock["lineage"]["status"] != "frozen_v2_design":
        raise ValueError("lineage lock status must be the immutable frozen_v2_design")
    if lock["lineage"]["protocol_sha256"] != protocol_sha:
        raise ValueError("lineage lock protocol hash differs from the raw artifact")
    if lock["lineage"]["code_identity_sha256"] != raw.get("code_identity_sha256"):
        raise ValueError("lineage lock code identity differs from the raw artifact")

    registry_path = root / AUDITED_REGISTRY_FILENAME
    registry_sha = _sha256_path(registry_path)
    if raw.get("audited_registry_sha256") != registry_sha:
        raise ValueError("local audited registry SHA-256 mismatch")
    descriptor = raw.get("audited_registry")
    if not isinstance(descriptor, Mapping) or descriptor != {
        "path": AUDITED_REGISTRY_FILENAME,
        "sha256": registry_sha,
        "registry_id": "fused3_confirmation_v2_inputs",
    }:
        raise ValueError("raw audited_registry descriptor is not canonical")
    registry = datasets.load_and_validate_registry(
        registry_path,
        expected_protocol_sha=protocol_sha,
        expected_lock_sha=str(raw["lineage_lock_sha256"]),
        expected_parent_registry_sha=str(
            lock["lineage"]["v1_confirmation_registry_sha256"]
        ),
    )
    return {
        "candidate_authority": authority,
        "lineage_lock": lock,
        "audited_registry": registry,
    }


def _validate_execution_identity(raw: Mapping[str, Any]) -> None:
    environment = raw.get("environment")
    expected_environment = {
        **EXPECTED_SCORE_ENVIRONMENT,
        "thread_environment": EXPECTED_THREAD_ENVIRONMENT,
    }
    if environment != expected_environment:
        raise ValueError("confirmation scoring environment is not the frozen environment")
    if raw.get("thread_environment") != EXPECTED_THREAD_ENVIRONMENT:
        raise ValueError("confirmation thread environment is not the frozen one-thread setting")
    sources = raw.get("source_hashes")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("source_hashes must be a nonempty mapping")
    if any(not isinstance(path, str) or not _is_sha256(digest) for path, digest in sources.items()):
        raise ValueError("source_hashes contains a malformed path or digest")
    if recipes.payload_sha256(sources) != raw.get("code_identity_sha256"):
        raise ValueError("code_identity_sha256 does not digest the exact source hash mapping")
    expected_sources = {
        str(path): _sha256_path(path) for path in runner._source_paths()
    }
    if dict(sources) != expected_sources:
        raise ValueError("source_hashes differ from the exact current executed source set")
    required_local = (
        Path(__file__).resolve(),
        Path(statistics.__file__).resolve(),
        Path(recipes.__file__).resolve(),
        Path(lineage.__file__).resolve(),
        PROTOCOL_PATH.resolve(),
        CANDIDATE_AUTHORITY_PATH.resolve(),
    )
    for local_path in required_local:
        try:
            suffix = str(local_path.relative_to(PACKAGE_DIR.parent.parent))
        except ValueError:  # focused tests may seal identity files in a temp root
            suffix = str(local_path)
        matches = [digest for path, digest in sources.items() if str(path).endswith(suffix)]
        if matches != [_sha256_path(local_path)]:
            raise ValueError(f"source_hashes does not bind current source {suffix!r}")
    run_payload = {
        "study": statistics.STUDY,
        "mode": "full",
        "protocol_sha256": raw["protocol_sha256"],
        "code_identity_sha256": raw["code_identity_sha256"],
        "lineage_lock_sha256": raw["lineage_lock_sha256"],
        "audited_registry_sha256": raw["audited_registry_sha256"],
        "environment": dict(EXPECTED_SCORE_ENVIRONMENT),
        "thread_environment": dict(EXPECTED_THREAD_ENVIRONMENT),
    }
    if recipes.payload_sha256(run_payload) != raw.get("run_identity_sha256"):
        raise ValueError("run_identity_sha256 does not digest the frozen execution identity")


def verify_completed_artifact(input_dir: Path | str) -> dict[str, Any]:
    """Verify exact bytes, duplicated lineage metadata, tables, and row grids."""

    root = Path(input_dir)
    manifest_path = root / "manifest.json"
    raw_path = root / "raw_results.json"
    manifest = _read_json(manifest_path)
    raw = _read_json(raw_path)
    if set(raw) != set(SHARED_METADATA_FIELDS):
        raise ValueError("raw_results has unknown or missing terminal metadata fields")
    if set(manifest) != {*SHARED_METADATA_FIELDS, "raw_results_sha256"}:
        raise ValueError("manifest has unknown or missing terminal metadata fields")
    if manifest.get("raw_results_sha256") != _sha256_path(raw_path):
        raise ValueError("terminal manifest raw_results SHA-256 mismatch")
    for field in SHARED_METADATA_FIELDS:
        if manifest.get(field) != raw.get(field):
            raise ValueError(f"manifest/raw metadata mismatch for {field!r}")
    if raw.get("schema_version") != 1:
        raise ValueError("confirmation artifact schema_version must be 1")
    if raw.get("study") != statistics.STUDY:
        raise ValueError("confirmation artifact study mismatch")
    if raw.get("artifact_status") != "completed":
        raise ValueError("confirmation artifact is not completed")
    if raw.get("mode") != "full":
        raise ValueError("confirmation analysis accepts only the completed full run")
    for field in (
        "protocol_sha256",
        "code_identity_sha256",
        "run_identity_sha256",
        "audited_registry_sha256",
        "lineage_lock_sha256",
    ):
        _require_sha(raw, field)
    local_identity = _verify_local_lineage(root, raw)
    _validate_execution_identity(raw)
    if raw.get("grid") != _expected_grid():
        raise ValueError("confirmation artifact grid does not match the exact frozen grid")
    if raw.get("checkpoint_count") != 500:
        raise ValueError("confirmation artifact checkpoint_count must be exactly 500")
    if raw.get("runtime_gate") != {
        "status": "inherited_pass",
        "source": "hash_bound_fused3_full_food_resource_result",
        "confirmation_clocks": "descriptive_only",
    }:
        raise ValueError("confirmation runtime gate inheritance is invalid")
    _validate_determinism(raw.get("determinism_verification"))
    table_manifest = raw.get("table_manifest")
    if not isinstance(table_manifest, Mapping) or set(table_manifest) != set(TABLE_FILES):
        raise ValueError("confirmation table_manifest has an invalid table set")

    loaded_rows: dict[str, list[dict[str, Any]]] = {}
    for filename, name in TABLE_FILES.items():
        descriptor = table_manifest.get(filename)
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"table manifest for {name!r} is malformed")
        if set(descriptor) != {"row_count", "sha256"}:
            raise ValueError(f"table manifest for {name!r} has invalid keys")
        rows, payload = _read_canonical_jsonl(root / filename)
        if descriptor.get("sha256") != _sha256_bytes(payload):
            raise ValueError(f"table manifest SHA-256 mismatch for {name!r}")
        if type(descriptor.get("row_count")) is not int or int(descriptor["row_count"]) != len(rows):
            raise ValueError(f"table manifest row count mismatch for {name!r}")
        loaded_rows[name] = rows
    selectors, references = statistics.validate_confirmation_inputs(
        loaded_rows["selector_rows"], loaded_rows["reference_rows"]
    )
    _validate_selector_structure(selectors)
    _validate_reference_structure(references)
    _validate_cohort_pairing(selectors, references)
    return {
        "root": root,
        "manifest": manifest,
        "raw": raw,
        "selector_rows": selectors,
        "reference_rows": references,
        **local_identity,
        "input_hashes": {
            "manifest_sha256": _sha256_path(manifest_path),
            "raw_results_sha256": _sha256_path(raw_path),
            "selector_rows_sha256": table_manifest["selector_rows.jsonl"]["sha256"],
            "reference_rows_sha256": table_manifest["reference_rows.jsonl"]["sha256"],
            "lineage_lock_sha256": raw["lineage_lock_sha256"],
            "audited_registry_sha256": raw["audited_registry_sha256"],
        },
    }


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _runtime_summary(
    selector_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selector_groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in selector_rows:
        selector_groups[(str(row["dataset_id"]), int(row["budget"]), str(row["candidate_id"]))].append(row)
    selector_summary = []
    for (dataset, budget, candidate), rows in sorted(selector_groups.items()):
        walls = [float(row["total_wall_seconds"]) for row in rows]
        cpus = [float(row["total_cpu_seconds"]) for row in rows]
        selector_summary.append(
            {
                "dataset_id": dataset,
                "budget": budget,
                "candidate_id": candidate,
                "call_count": len(rows),
                "median_wall_seconds": float(np.median(walls)),
                "p95_wall_seconds": _percentile(walls, 0.95),
                "median_cpu_seconds": float(np.median(cpus)),
                "p95_cpu_seconds": _percentile(cpus, 0.95),
                "inferential": False,
            }
        )

    panels: dict[tuple[str, int, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in selector_rows:
        panels[(str(row["dataset_id"]), int(row["replicate_seed"]), int(row["budget"]), str(row["candidate_id"]))].append(row)
    panel_groups: dict[tuple[str, int, str], list[tuple[float, float]]] = defaultdict(list)
    for (dataset, _seed, budget, candidate), rows in panels.items():
        if len(rows) != len(statistics.BACKBONES):
            raise ValueError("descriptive runtime panel is missing a backbone")
        panel_groups[(dataset, budget, candidate)].append(
            (
                float(sum(float(row["total_wall_seconds"]) for row in rows)),
                float(sum(float(row["total_cpu_seconds"]) for row in rows)),
            )
        )
    full_panel_summary = []
    for (dataset, budget, candidate), values in sorted(panel_groups.items()):
        walls = [value[0] for value in values]
        cpus = [value[1] for value in values]
        full_panel_summary.append(
            {
                "dataset_id": dataset,
                "budget": budget,
                "candidate_id": candidate,
                "panel_count": len(values),
                "median_wall_seconds": float(np.median(walls)),
                "p95_wall_seconds": _percentile(walls, 0.95),
                "median_cpu_seconds": float(np.median(cpus)),
                "p95_cpu_seconds": _percentile(cpus, 0.95),
                "inferential": False,
            }
        )

    reference_groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in reference_rows:
        reference_groups[(str(row["dataset_id"]), int(row["budget"]), str(row["head"]))].append(row)
    reference_summary = []
    for (dataset, budget, head), rows in sorted(reference_groups.items()):
        walls = [float(row["total_wall_seconds"]) for row in rows]
        cpus = [float(row["total_cpu_seconds"]) for row in rows]
        reference_summary.append(
            {
                "dataset_id": dataset,
                "budget": budget,
                "head": head,
                "fit_count": len(rows),
                "median_wall_seconds": float(np.median(walls)),
                "p95_wall_seconds": _percentile(walls, 0.95),
                "median_cpu_seconds": float(np.median(cpus)),
                "p95_cpu_seconds": _percentile(cpus, 0.95),
                "inferential": False,
            }
        )
    return {
        "gate_source": "hash_bound_passed_fused3_food_resource_evidence",
        "confirmation_timings": "descriptive_only",
        "promotion_veto": False,
        "selector_calls": selector_summary,
        "full_ten_backbone_panels": full_panel_summary,
        "reference_heads": reference_summary,
    }


def analyze_loaded(loaded: Mapping[str, Any]) -> dict[str, Any]:
    summary = statistics.summarize(
        loaded["selector_rows"], loaded["reference_rows"]
    )
    raw = loaded["raw"]
    summary["artifact_status"] = "completed"
    summary["input_hashes"] = dict(loaded["input_hashes"])
    summary["lineage"] = {
        "protocol_sha256": raw["protocol_sha256"],
        "code_identity_sha256": raw["code_identity_sha256"],
        "run_identity_sha256": raw["run_identity_sha256"],
        "audited_registry_sha256": raw["audited_registry_sha256"],
        "lineage_lock_sha256": raw["lineage_lock_sha256"],
        "inherited_food_resource_status": "pass_hash_bound_by_lineage_lock",
    }
    summary["runtime"] = _runtime_summary(
        loaded["selector_rows"], loaded["reference_rows"]
    )
    summary["decision"]["lineage_lock_sha256"] = raw["lineage_lock_sha256"]
    summary["decision"]["runtime_gate"] = "inherited_pass_hash_bound_by_lineage_lock"
    return summary


def render_report(summary: Mapping[str, Any]) -> str:
    decision = summary["decision"]
    lines = [
        "# FUSED3 untouched confirmation V2",
        "",
        f"Decision: **{decision['status']}** for the locked `FUSED3` candidate.",
        "",
        "This is a one-time confirmation of the frozen class-stratified, same-distribution five-dataset panel. It is not a distribution-shift or universal backbone-ranking claim.",
        "",
        "No candidate reselection, fallback, fusion, recipe change, or gate reinterpretation was permitted.",
        "",
        "## Primary confirmation gates",
        "",
        "| Head | Estimate pp | Lower 95% | Upper 95% | Favorable datasets | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for head in statistics.HEADS:
        gate = summary["confirmation"]["gates"][head]
        favorable = gate.get("favorable_dataset_count", "—")
        lines.append(
            f"| {head} | {gate['estimate']:.3f} | {gate['lower_95']:.3f} | {gate['upper_95']:.3f} | {favorable} | {gate['status']} |"
        )
    rank = summary["rank_support"]
    lines.extend(
        [
            "",
            "## Supporting rank evidence",
            "",
            f"Two-budget log2 rank-AUC support status: **{rank['status']}**; {rank['defined_curve_count']} of {rank['expected_curve_count']} curves were defined. Rank AUC is descriptive and not a promotion veto.",
            "",
            "## Dataset selection summary",
            "",
            "| Dataset | Head | Selector | Mean regret pp | Exact best | Within 1 pp | Mean Spearman |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["aggregate"]:
        rho = "—" if row["mean_spearman"] is None else f"{row['mean_spearman']:.3f}"
        lines.append(
            f"| {row['dataset_id']} | {row['head']} | {row['candidate_id']} | {row['mean_regret_pp']:.3f} | {row['exact_best_rate']:.3f} | {row['within_one_pp_rate']:.3f} | {rho} |"
        )
    lines.extend(
        [
            "",
            "## Runtime and resources",
            "",
            "The viability resource gate is inherited from the exact hash-bound passing FUSED3 Food evidence. Confirmation wall/CPU measurements, including full ten-backbone panel totals, are descriptive only and cannot rescue or reject the locked candidate.",
            "",
            f"Lineage lock: `{summary['lineage']['lineage_lock_sha256']}`.",
            "",
        ]
    )
    if decision["status"] == "pass_confirmed":
        lines.append(
            "FUSED3 passed all four predeclared confirmation gates on the exact frozen panel."
        )
    else:
        lines.append(
            "FUSED3 failed at least one predeclared confirmation gate. The locked candidate is rejected; no runner-up is substituted."
        )
    lines.append("")
    return "\n".join(lines)


def _csv_bytes(rows: Sequence[Mapping[str, Any]], *, sort_fields: Sequence[str]) -> bytes:
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: tuple(str(row.get(field, "")) for field in sort_fields),
    )
    fields = sorted({str(key) for row in ordered for key in row})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in ordered:
        writer.writerow(
            {
                field: canonical_json(row.get(field))
                if isinstance(row.get(field), (Mapping, list, tuple))
                else row.get(field)
                for field in fields
            }
        )
    return buffer.getvalue().encode("utf-8")


def _write_file(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def write_analysis_bundle(input_dir: Path | str, output_dir: Path | str) -> dict[str, Any]:
    """Verify, analyze once, and publish one atomic non-overwriting bundle."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    loaded = verify_completed_artifact(input_dir)
    summary = analyze_loaded(loaded)
    decision = dict(summary["decision"])
    payloads = {
        "summary.json": (canonical_json(summary) + "\n").encode("utf-8"),
        "decision.json": (canonical_json(decision) + "\n").encode("utf-8"),
        "panel_metrics.csv": _csv_bytes(
            summary["panel_metrics"],
            sort_fields=("dataset_id", "replicate_seed", "budget", "head", "candidate_id"),
        ),
        "regret_contrasts.csv": _csv_bytes(
            summary["regret_contrasts"],
            sort_fields=("dataset_id", "replicate_seed", "budget", "head"),
        ),
        "rank_auc.csv": _csv_bytes(
            summary["rank_auc"],
            sort_fields=("dataset_id", "replicate_seed", "head", "candidate_id"),
        ),
        "runtime_descriptive.csv": _csv_bytes(
            [
                *summary["runtime"]["selector_calls"],
                *summary["runtime"]["full_ten_backbone_panels"],
                *summary["runtime"]["reference_heads"],
            ],
            sort_fields=("dataset_id", "budget", "candidate_id", "head"),
        ),
        "report.md": render_report(summary).encode("utf-8"),
    }
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        for name, payload in payloads.items():
            _write_file(staging / name, payload)
        analysis_manifest = {
            "schema_version": 1,
            "study": statistics.STUDY,
            "stage": "confirmation_analysis",
            "artifact_status": "completed",
            "decision_status": decision["status"],
            "input_hashes": dict(summary["input_hashes"]),
            "protocol_sha256": summary["lineage"]["protocol_sha256"],
            "code_identity_sha256": summary["lineage"]["code_identity_sha256"],
            "audited_registry_sha256": summary["lineage"]["audited_registry_sha256"],
            "lineage_lock_sha256": summary["lineage"]["lineage_lock_sha256"],
            "files": {
                name: {"sha256": _sha256_bytes(payload), "size_bytes": len(payload)}
                for name, payload in sorted(payloads.items())
            },
        }
        manifest_payload = (canonical_json(analysis_manifest) + "\n").encode("utf-8")
        _write_file(staging / "analysis_manifest.json", manifest_payload)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = write_analysis_bundle(args.input, args.output)
    print(canonical_json(summary["decision"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "analyze_loaded",
    "canonical_json",
    "main",
    "render_report",
    "verify_completed_artifact",
    "write_analysis_bundle",
]
