"""Atomic runner for the frozen FUSED3 five-dataset confirmation panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import sklearn

from experiments.fused3_confirmation_v2 import recipes


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[1]
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SIDECAR = PACKAGE_DIR / "protocol.sha256"
CANDIDATE_AUTHORITY_PATH = PACKAGE_DIR / "candidate_authority.json"
THREAD_ENVIRONMENT = {
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "LOKY_MAX_CPU_COUNT": "1",
}
EXPECTED_SCORE_ENVIRONMENT = {
    "python": "3.12.2",
    "numpy": "2.1.3",
    "sklearn": "1.6.1",
}
TERMINAL_FILES = frozenset(
    {"raw_results.json", "selector_rows.jsonl", "reference_rows.jsonl", "manifest.json"}
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_path(path: os.PathLike[str] | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: os.PathLike[str] | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object at {path}")
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, canonical_json(value) + "\n")


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    materialized = list(rows)
    _atomic_text(path, "".join(canonical_json(dict(row)) + "\n" for row in materialized))
    return len(materialized)


def validate_protocol(*, require_frozen: bool = True) -> tuple[str, dict[str, Any]]:
    if not PROTOCOL_PATH.is_file():
        raise RuntimeError("confirmation protocol is missing")
    observed = sha256_path(PROTOCOL_PATH)
    if not PROTOCOL_SIDECAR.is_file():
        if require_frozen:
            raise RuntimeError("confirmation protocol sidecar is missing")
    else:
        expected = PROTOCOL_SIDECAR.read_text(encoding="utf-8").strip()
        if len(expected) != 64 or expected != observed:
            raise RuntimeError("confirmation protocol sidecar mismatch")
    protocol = _read_json(PROTOCOL_PATH)
    expected_status = "frozen_before_confirmation_outcomes"
    if require_frozen and protocol.get("status") != expected_status:
        raise RuntimeError(f"confirmation protocol status must be {expected_status!r}")
    panel = protocol.get("panel")
    expected_panel = {
        "datasets": [
            {"id": "torchvision_cifar10", "training_split": "train", "evaluation_split": "test", "classes": 10},
            {"id": "torchvision_stl10", "training_split": "train", "evaluation_split": "test", "classes": 10},
            {"id": "torchvision_gtsrb", "training_split": "train", "evaluation_split": "test", "classes": 43},
            {"id": "torchvision_fgvc_aircraft", "training_split": "trainval", "evaluation_split": "test", "classes": 100},
            {"id": "torchvision_dtd", "training_split": "train_plus_val_partition_1", "evaluation_split": "test_partition_1", "classes": 47},
        ],
        "backbones": list(recipes.BACKBONES),
        "budgets_per_class": list(recipes.BUDGETS),
        "replicate_seeds": list(recipes.REPLICATE_SEEDS),
        "heads": list(recipes.HEADS),
        "ranking_panel_count": 50,
        "backbone_panel_count": 500,
        "selector_row_count": 1000,
        "reference_row_count": 2000,
    }
    if panel != expected_panel:
        raise RuntimeError("confirmation protocol panel drift")
    return observed, protocol


def _thread_environment() -> dict[str, str]:
    observed = {key: os.environ.get(key) for key in THREAD_ENVIRONMENT}
    if observed != THREAD_ENVIRONMENT:
        raise RuntimeError(
            f"one-thread environment mismatch: expected {THREAD_ENVIRONMENT!r}, got {observed!r}"
        )
    return {key: str(value) for key, value in observed.items()}


def _score_environment() -> dict[str, str]:
    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }
    if observed != EXPECTED_SCORE_ENVIRONMENT:
        raise RuntimeError(
            f"confirmation score environment mismatch: expected {EXPECTED_SCORE_ENVIRONMENT!r}, got {observed!r}"
        )
    return observed


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_paths() -> tuple[Path, ...]:
    local = tuple(sorted(PACKAGE_DIR.glob("*.py"))) + (
        PROTOCOL_PATH,
        PROTOCOL_SIDECAR,
        CANDIDATE_AUTHORITY_PATH,
        ROOT / "pyproject.toml",
        ROOT / "experiments" / "scalable_relevance_kmeans" / "scalable.py",
        ROOT / "experiments" / "scalable_relevance_kmeans" / "scalable_v2.py",
        ROOT / "experiments" / "scalable_relevance_kmeans" / "fused_prototypes.py",
        ROOT / "experiments" / "scalable_relevance_kmeans" / "fused_food_screen.py",
        ROOT / "experiments" / "m50_backbone_ranking" / "food101.py",
        ROOT / "experiments" / "m50_backbone_ranking" / "manifest.py",
        ROOT / "experiments" / "m50_backbone_ranking" / "candidates.py",
        ROOT / "experiments" / "heteroscedastic_distance_conditioning" / "conditioning_adapter.py",
        Path("/Users/niklasmelton/code/vertabrae/examples/food101_nonlinear_backbone_bridge.py"),
    )
    return local + tuple(sorted((ROOT / "overlapindex").glob("*.py")))


def source_identity() -> dict[str, Any]:
    paths = _source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing confirmation sources: {missing!r}")
    source_hashes = {str(path): sha256_path(path) for path in paths}
    return {
        "source_hashes": source_hashes,
        "code_identity_sha256": payload_sha256(source_hashes),
        "repository": {
            "branch": _git("branch", "--show-current"),
            "experiment_commit": _git("rev-parse", "HEAD"),
            "confirmation_start_commit": "a5d246911a928fb76928ddabac70e3ca3be84747",
            "develop_merge_base": _git("merge-base", "HEAD", "develop"),
            "dirty_status_recorded": _git("status", "--short"),
        },
    }


def _clock_free(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _clock_free(item)
            for key, item in value.items()
            if not str(key).endswith("_wall_seconds")
            and not str(key).endswith("_cpu_seconds")
            and str(key) not in {"wall_seconds", "cpu_seconds"}
        }
    if isinstance(value, (list, tuple)):
        return [_clock_free(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _signature(value: Mapping[str, Any]) -> str:
    return payload_sha256(_clock_free(value))


def _panel_id(dataset_id: str, backbone: str, replicate: int, budget: int) -> str:
    return f"{dataset_id}__{backbone}__r{replicate}__b{budget}"


def panel_identities() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    panel_index = 0
    for dataset_id in recipes.DATASET_IDS:
        for backbone in recipes.BACKBONES:
            for replicate, seed in enumerate(recipes.REPLICATE_SEEDS):
                for budget in recipes.BUDGETS:
                    rows.append(
                        {
                            "panel_index": panel_index,
                            "panel_id": _panel_id(dataset_id, backbone, replicate, budget),
                            "dataset_id": dataset_id,
                            "backbone": backbone,
                            "replicate": replicate,
                            "replicate_seed": seed,
                            "budget": budget,
                            "execution_order": list(recipes.method_order(panel_index)),
                        }
                    )
                    panel_index += 1
    if len(rows) != 500:
        raise AssertionError("frozen panel count drift")
    return tuple(rows)


def _result_row(
    result: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    position: int,
    wall: float,
    cpu: float,
    cohort: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = str(result["candidate_id"])
    recipe = recipes.candidate_recipe(candidate)
    if result.get("candidate_recipe_sha256") not in (
        None,
        recipes.payload_sha256(recipe),
    ):
        raise RuntimeError(f"{candidate} recipe identity drift")
    return {
        **{key: identity[key] for key in (
            "panel_id", "dataset_id", "backbone", "replicate", "replicate_seed", "budget"
        )},
        "candidate_id": candidate,
        "score": float(result["score"]),
        "total_wall_seconds": float(wall),
        "total_cpu_seconds": float(cpu),
        "fit_wall_seconds": float(result.get("fit_wall_seconds", 0.0)),
        "fit_cpu_seconds": float(result.get("fit_cpu_seconds", 0.0)),
        "score_fixed_wall_seconds": float(result.get("score_fixed_wall_seconds", 0.0)),
        "score_fixed_cpu_seconds": float(result.get("score_fixed_cpu_seconds", 0.0)),
        "execution_position": int(position),
        "execution_order": list(identity["execution_order"]),
        "warmup_excluded": True,
        "candidate_recipe_sha256": recipes.payload_sha256(recipe),
        "training_cohort_sha256": str(cohort["training_cohort_sha256"]),
        "evaluation_cohort_sha256": str(cohort["evaluation_cohort_sha256"]),
        "folds": list(result.get("folds", [])),
        "status": "ok",
        "error": None,
    }


def _reference_row(
    result: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    wall: float,
    cpu: float,
    cohort: Mapping[str, Any],
) -> dict[str, Any]:
    head = str(result["head"])
    return {
        **{key: identity[key] for key in (
            "panel_id", "dataset_id", "backbone", "replicate", "replicate_seed", "budget"
        )},
        "head": head,
        "test_accuracy": float(result["test_accuracy"]),
        "total_wall_seconds": float(wall),
        "total_cpu_seconds": float(cpu),
        "training_row_count": int(result["training_row_count"]),
        "evaluation_row_count": int(result["evaluation_row_count"]),
        "recipe_sha256": str(result["recipe_sha256"]),
        "training_cohort_sha256": str(cohort["training_cohort_sha256"]),
        "evaluation_cohort_sha256": str(cohort["evaluation_cohort_sha256"]),
        "status": "ok",
        "error": None,
    }


def execute_panel(
    identity: Mapping[str, Any],
    panel: Mapping[str, Any],
    *,
    selector_executor: Callable[..., Mapping[str, Any]] = recipes.execute_selector,
    reference_executor: Callable[..., Mapping[str, Any]] = recipes.execute_reference_head,
) -> dict[str, Any]:
    train = np.asarray(panel["training_values"], dtype=np.float32)
    train_labels = np.asarray(panel["training_labels"])
    evaluation = np.asarray(panel["evaluation_values"], dtype=np.float32)
    evaluation_labels = np.asarray(panel["evaluation_labels"])
    cohort_hashes = panel.get("cohort_hashes")
    if isinstance(cohort_hashes, Mapping):
        cohort = {
            "training_cohort_sha256": str(
                cohort_hashes["training_sample_ids_sha256"]
            ),
            "evaluation_cohort_sha256": str(
                cohort_hashes["evaluation_sample_ids_sha256"]
            ),
        }
    else:
        cohort = {
            "training_cohort_sha256": str(panel["training_cohort_sha256"]),
            "evaluation_cohort_sha256": str(panel["evaluation_cohort_sha256"]),
        }
    if train.ndim != 2 or evaluation.ndim != 2 or train.shape[1] != evaluation.shape[1]:
        raise RuntimeError("panel matrices have invalid dimensions")
    if len(train) != len(train_labels) or len(evaluation) != len(evaluation_labels):
        raise RuntimeError("panel matrices and labels are misaligned")
    expected_train = int(identity["budget"]) * len(np.unique(train_labels))
    if len(train) != expected_train:
        raise RuntimeError("panel training cohort does not match class-balanced budget")
    selector_rows: list[dict[str, Any]] = []
    for position, candidate in enumerate(identity["execution_order"]):
        started_wall, started_cpu = time.perf_counter(), time.process_time()
        result = selector_executor(
            candidate, train, train_labels, seed=int(identity["replicate_seed"])
        )
        wall = time.perf_counter() - started_wall
        cpu = time.process_time() - started_cpu
        selector_rows.append(
            _result_row(
                result,
                identity=identity,
                position=position,
                wall=wall,
                cpu=cpu,
                cohort=cohort,
            )
        )
    reference_rows: list[dict[str, Any]] = []
    for head_index, head in enumerate(recipes.HEADS):
        started_wall, started_cpu = time.perf_counter(), time.process_time()
        result = reference_executor(
            head,
            train,
            train_labels,
            evaluation,
            evaluation_labels,
            seed=int(identity["replicate_seed"]) + head_index,
        )
        wall = time.perf_counter() - started_wall
        cpu = time.process_time() - started_cpu
        reference_rows.append(
            _reference_row(
                result,
                identity=identity,
                wall=wall,
                cpu=cpu,
                cohort=cohort,
            )
        )
    return {
        "schema_version": 1,
        "artifact_status": "completed_panel",
        "panel_identity": dict(identity),
        "selector_rows": selector_rows,
        "reference_rows": reference_rows,
    }


def _redacted_descriptor(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        kind = "bool"
    elif isinstance(value, int):
        kind = "int"
    elif isinstance(value, float):
        kind = "float"
    else:
        kind = type(value).__name__
    finite = bool(np.isfinite(value)) if isinstance(value, (int, float)) else True
    return {"present": True, "type": kind, "finite": finite}


def redact_smoke_panel(panel: Mapping[str, Any]) -> dict[str, Any]:
    selectors = []
    for row in panel["selector_rows"]:
        selectors.append(
            {
                key: row[key]
                for key in (
                    "panel_id", "dataset_id", "backbone", "replicate", "replicate_seed",
                    "budget", "candidate_id", "execution_position", "execution_order",
                    "warmup_excluded", "candidate_recipe_sha256", "training_cohort_sha256",
                    "evaluation_cohort_sha256", "status", "error"
                )
            }
            | {"score_structure": _redacted_descriptor(row["score"]), "outcomes_redacted": True}
        )
    references = []
    for row in panel["reference_rows"]:
        references.append(
            {
                key: row[key]
                for key in (
                    "panel_id", "dataset_id", "backbone", "replicate", "replicate_seed",
                    "budget", "head", "training_row_count", "evaluation_row_count",
                    "recipe_sha256", "training_cohort_sha256", "evaluation_cohort_sha256",
                    "status", "error"
                )
            }
            | {
                "accuracy_structure": _redacted_descriptor(row["test_accuracy"]),
                "outcomes_redacted": True,
            }
        )
    return {
        "schema_version": 1,
        "artifact_status": "completed_structural_smoke_panel",
        "panel_identity": dict(panel["panel_identity"]),
        "selector_rows": selectors,
        "reference_rows": references,
    }


def _checkpoint_path(output: Path, panel_id: str) -> Path:
    name = hashlib.sha256(panel_id.encode("utf-8")).hexdigest()[:24]
    return output / "checkpoints" / f"{name}.json"


def _validate_checkpoint(
    value: Mapping[str, Any], *, identity: Mapping[str, Any], run_identity: str
) -> None:
    if value.get("run_identity_sha256") != run_identity:
        raise RuntimeError("checkpoint run identity mismatch")
    if value.get("panel_identity") != identity:
        raise RuntimeError("checkpoint panel identity mismatch")
    if value.get("artifact_status") != "completed_panel":
        raise RuntimeError("checkpoint is not a completed panel")
    selectors = value.get("selector_rows")
    references = value.get("reference_rows")
    if not isinstance(selectors, list) or not isinstance(references, list):
        raise RuntimeError("checkpoint tables are malformed")
    if len(selectors) != len(recipes.SELECTORS) or any(
        not isinstance(row, Mapping) for row in selectors
    ):
        raise RuntimeError("checkpoint selector rows are incomplete")
    if {row.get("candidate_id") for row in selectors} != set(recipes.SELECTORS):
        raise RuntimeError("checkpoint selector rows are incomplete")
    if len(references) != len(recipes.HEADS) or any(
        not isinstance(row, Mapping) for row in references
    ):
        raise RuntimeError("checkpoint reference rows are incomplete")
    if {row.get("head") for row in references} != set(recipes.HEADS):
        raise RuntimeError("checkpoint reference rows are incomplete")
    expected_panel = {
        key: identity[key]
        for key in (
            "panel_id",
            "dataset_id",
            "backbone",
            "replicate",
            "replicate_seed",
            "budget",
        )
    }
    for row in (*selectors, *references):
        if any(row.get(key) != expected for key, expected in expected_panel.items()):
            raise RuntimeError("checkpoint row identity differs from its panel")
        if row.get("status") != "ok" or row.get("error") is not None:
            raise RuntimeError("checkpoint contains a failed row")
        field = "score" if "candidate_id" in row else "test_accuracy"
        try:
            finite = np.isfinite(float(row[field]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("checkpoint outcome is malformed") from exc
        if not finite:
            raise RuntimeError("checkpoint outcome is non-finite")


def _validate_determinism(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_ids = {
        *recipes.SELECTORS,
        *(f"HEAD:{head}" for head in recipes.HEADS),
    }
    if not isinstance(value, Mapping) or set(value) != expected_ids:
        raise RuntimeError("confirmation determinism descriptor set is incomplete")
    first_panel = panel_identities()[0]["panel_id"]
    keys = {
        "first_signature_sha256",
        "second_signature_sha256",
        "exact",
        "runtime_fields_excluded",
        "panel_id",
    }
    normalized: dict[str, Any] = {}
    for identity in sorted(expected_ids):
        descriptor = value[identity]
        if not isinstance(descriptor, Mapping) or set(descriptor) != keys:
            raise RuntimeError("confirmation determinism descriptor schema mismatch")
        first = descriptor["first_signature_sha256"]
        second = descriptor["second_signature_sha256"]
        if (
            not isinstance(first, str)
            or len(first) != 64
            or any(character not in "0123456789abcdef" for character in first)
            or second != first
            or descriptor["exact"] is not True
            or descriptor["runtime_fields_excluded"] is not True
            or descriptor["panel_id"] != first_panel
        ):
            raise RuntimeError("confirmation deterministic repeat did not match exactly")
        normalized[identity] = dict(descriptor)
    return normalized


def _load_registry(
    registry_path: Path,
    protocol_sha: str,
    lock_sha: str,
    parent_registry_sha: str,
) -> tuple[dict[str, Any], Callable[..., Mapping[str, Any]]]:
    from experiments.fused3_confirmation_v2 import datasets

    registry = datasets.load_and_validate_registry(
        registry_path,
        expected_protocol_sha=protocol_sha,
        expected_lock_sha=lock_sha,
        expected_parent_registry_sha=parent_registry_sha,
    )
    by_dataset = {
        str(record["dataset_id"]): record for record in registry["datasets"]
    }

    def load_from_registry(
        _registry: Mapping[str, Any],
        *,
        dataset_id: str,
        backbone: str,
        replicate_seed: int,
        budget: int,
    ) -> Mapping[str, Any]:
        del _registry
        return datasets.load_panel(
            by_dataset[dataset_id],
            backbone,
            replicate_seed,
            budget,
            registry_root=registry_path.parent,
        )

    return dict(registry), load_from_registry


def _validate_terminal_counts(
    selectors: Sequence[Mapping[str, Any]], references: Sequence[Mapping[str, Any]]
) -> None:
    if len(selectors) != 1000 or len(references) != 2000:
        raise RuntimeError("terminal row counts are incomplete")
    selector_keys = {
        (
            row.get("dataset_id"), row.get("backbone"), row.get("replicate"),
            row.get("budget"), row.get("candidate_id")
        )
        for row in selectors
    }
    reference_keys = {
        (
            row.get("dataset_id"), row.get("backbone"), row.get("replicate"),
            row.get("budget"), row.get("head")
        )
        for row in references
    }
    if len(selector_keys) != 1000 or len(reference_keys) != 2000:
        raise RuntimeError("terminal rows contain duplicates")
    for row in (*selectors, *references):
        if row.get("status") != "ok" or row.get("error") is not None:
            raise RuntimeError("terminal rows contain a failed result")
        field = "score" if "candidate_id" in row else "test_accuracy"
        if not np.isfinite(float(row[field])):
            raise RuntimeError("terminal rows contain a non-finite outcome")


def _prepare_output(output: Path, *, run_identity: str, resume: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    terminal = output / "manifest.json"
    if terminal.exists():
        raise RuntimeError("terminal confirmation artifact exists; refuse overwrite/resume")
    running = output / "running_manifest.json"
    if running.exists():
        value = _read_json(running)
        if not resume or value.get("run_identity_sha256") != run_identity:
            raise RuntimeError("running confirmation artifact identity mismatch")
    else:
        unexpected = [path for path in output.iterdir() if path.name not in {"checkpoints"}]
        if unexpected:
            raise RuntimeError("nonempty confirmation output lacks an identity manifest")
        _atomic_json(
            running,
            {
                "schema_version": 1,
                "study": "fused3_backbone_ranking_confirmation_v2",
                "artifact_status": "running",
                "run_identity_sha256": run_identity,
            },
        )


def run_confirmation(
    *,
    registry_path: Path,
    lineage_lock_path: Path,
    output: Path,
    resume: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    protocol_sha, protocol = validate_protocol(require_frozen=True)
    threads = _thread_environment()
    environment = _score_environment()
    sources = source_identity()
    from experiments.fused3_confirmation_v2 import lineage

    lock_payload = _read_json(lineage_lock_path)
    lineage.validate_candidate_authority_file(CANDIDATE_AUTHORITY_PATH)
    expected_lineage = lineage.build_lineage_from_files(
        protocol_path=PROTOCOL_PATH,
        protocol_sidecar_path=PROTOCOL_SIDECAR,
        code_identity_sha256=sources["code_identity_sha256"],
    )
    lineage.validate_lock_envelope(
        lock_payload,
        expected_lineage=expected_lineage,
        expected_candidate_authority_sha256=lineage.authority_sha256(
            CANDIDATE_AUTHORITY_PATH
        ),
    )
    lock_sha = sha256_path(lineage_lock_path)
    registry, loader = _load_registry(
        Path(registry_path),
        protocol_sha,
        lock_sha,
        str(lock_payload["lineage"]["v1_confirmation_registry_sha256"]),
    )
    registry_sha = sha256_path(registry_path)

    # Validate every one of the 50 embedding matrix/manifest pairs before the
    # first selector or reference outcome can be computed.  A corruption in a
    # later dataset/backbone must stop the stage without leaving a partial
    # confirmation panel behind.
    for dataset_id in recipes.DATASET_IDS:
        for backbone in recipes.BACKBONES:
            loader(
                registry,
                dataset_id=dataset_id,
                backbone=backbone,
                replicate_seed=recipes.REPLICATE_SEEDS[0],
                budget=recipes.BUDGETS[0],
            )
    identity_payload = {
        "study": "fused3_backbone_ranking_confirmation_v2",
        "mode": "smoke" if smoke else "full",
        "protocol_sha256": protocol_sha,
        "code_identity_sha256": sources["code_identity_sha256"],
        "lineage_lock_sha256": lock_sha,
        "audited_registry_sha256": registry_sha,
        "environment": environment,
        "thread_environment": threads,
    }
    run_identity = payload_sha256(identity_payload)
    _prepare_output(Path(output), run_identity=run_identity, resume=resume)
    local_registry_path = Path(output) / "audited_registry.json"
    local_lock_path = Path(output) / "lineage_lock.json"
    if local_registry_path.exists() or local_lock_path.exists():
        if not resume:
            raise RuntimeError("local confirmation identity copies exist without --resume")
        if (
            sha256_path(local_registry_path) != registry_sha
            or sha256_path(local_lock_path) != lock_sha
        ):
            raise RuntimeError("local confirmation identity copy mismatch")
    else:
        _atomic_text(local_registry_path, Path(registry_path).read_text(encoding="utf-8"))
        _atomic_text(local_lock_path, Path(lineage_lock_path).read_text(encoding="utf-8"))
    identities = panel_identities()[:1] if smoke else panel_identities()
    checkpoints: list[dict[str, Any]] = []
    running_path = Path(output) / "running_manifest.json"
    running_payload = _read_json(running_path)
    cached_determinism = running_payload.get("determinism_verification", {})
    if not isinstance(cached_determinism, Mapping):
        raise RuntimeError("running confirmation determinism metadata is malformed")
    determinism: dict[str, Any] = (
        _validate_determinism(cached_determinism)
        if cached_determinism
        else {}
    )
    for index, identity in enumerate(identities):
        checkpoint = _checkpoint_path(Path(output), str(identity["panel_id"]))
        if checkpoint.exists():
            if not resume:
                raise RuntimeError("checkpoint exists without --resume")
            cached = _read_json(checkpoint)
            _validate_checkpoint(cached, identity=identity, run_identity=run_identity)
            checkpoints.append(cached)
            continue
        panel = loader(
            registry,
            dataset_id=str(identity["dataset_id"]),
            backbone=str(identity["backbone"]),
            replicate_seed=int(identity["replicate_seed"]),
            budget=int(identity["budget"]),
        )
        if index == 0:
            # The first call is the excluded warm-up; the second is the
            # measured panel that enters the terminal tables.  Clock-free
            # signatures must match exactly.
            warmup = execute_panel(identity, panel)
            result = execute_panel(identity, panel)
            for candidate in recipes.SELECTORS:
                first = next(row for row in warmup["selector_rows"] if row["candidate_id"] == candidate)
                second = next(row for row in result["selector_rows"] if row["candidate_id"] == candidate)
                first_sha, second_sha = _signature(first), _signature(second)
                determinism[candidate] = {
                    "first_signature_sha256": first_sha,
                    "second_signature_sha256": second_sha,
                    "exact": first_sha == second_sha,
                    "runtime_fields_excluded": True,
                    "panel_id": identity["panel_id"],
                }
            for head in recipes.HEADS:
                first = next(row for row in warmup["reference_rows"] if row["head"] == head)
                second = next(row for row in result["reference_rows"] if row["head"] == head)
                first_sha, second_sha = _signature(first), _signature(second)
                determinism[f"HEAD:{head}"] = {
                    "first_signature_sha256": first_sha,
                    "second_signature_sha256": second_sha,
                    "exact": first_sha == second_sha,
                    "runtime_fields_excluded": True,
                    "panel_id": identity["panel_id"],
                }
            if not all(value["exact"] for value in determinism.values()):
                raise RuntimeError("confirmation deterministic repeat failed")
            determinism = _validate_determinism(determinism)
            running_payload["determinism_verification"] = determinism
            _atomic_json(running_path, running_payload)
        else:
            result = execute_panel(identity, panel)
        payload = {**result, "run_identity_sha256": run_identity}
        # A structural smoke is allowed to compute a result transiently only
        # to validate the execution path.  It must never persist a recoverable
        # numeric selector/reference outcome, including in checkpoints.
        if not smoke:
            _atomic_json(checkpoint, payload)
        checkpoints.append(payload)
        print(f"completed {index + 1}/{len(identities)} {identity['panel_id']}", flush=True)
    if smoke:
        redacted = redact_smoke_panel(checkpoints[0])
        raw = {
            **identity_payload,
            "schema_version": 1,
            "study": "fused3_backbone_ranking_confirmation_v2",
            "artifact_status": "completed_structural_smoke",
            "run_identity_sha256": run_identity,
            "repository_provenance": sources["repository"],
            "source_hashes": sources["source_hashes"],
            "audited_registry": {
                "path": "audited_registry.json",
                "sha256": registry_sha,
                "registry_id": registry.get("registry_id"),
            },
            "protocol": protocol,
            "outcomes_redacted": True,
            "ranking_available": False,
            "panel": redacted,
            "determinism_verification": determinism,
        }
        _atomic_json(Path(output) / "raw_results.json", raw)
        manifest = {
            **identity_payload,
            "schema_version": 1,
            "study": "fused3_backbone_ranking_confirmation_v2",
            "artifact_status": "completed_structural_smoke",
            "run_identity_sha256": run_identity,
            "raw_results_sha256": sha256_path(Path(output) / "raw_results.json"),
            "repository_provenance": sources["repository"],
            "source_hashes": sources["source_hashes"],
            "audited_registry": {
                "path": "audited_registry.json",
                "sha256": registry_sha,
                "registry_id": registry.get("registry_id"),
            },
            "protocol": protocol,
            "outcomes_redacted": True,
        }
        _atomic_json(Path(output) / "manifest.json", manifest)
        running_path.unlink(missing_ok=True)
        return manifest
    selector_rows = [row for checkpoint in checkpoints for row in checkpoint["selector_rows"]]
    reference_rows = [row for checkpoint in checkpoints for row in checkpoint["reference_rows"]]
    _validate_terminal_counts(selector_rows, reference_rows)
    selector_path = Path(output) / "selector_rows.jsonl"
    reference_path = Path(output) / "reference_rows.jsonl"
    selector_count = _atomic_jsonl(selector_path, selector_rows)
    reference_count = _atomic_jsonl(reference_path, reference_rows)
    table_manifest = {
        "selector_rows.jsonl": {
            "row_count": selector_count,
            "sha256": sha256_path(selector_path),
        },
        "reference_rows.jsonl": {
            "row_count": reference_count,
            "sha256": sha256_path(reference_path),
        },
    }
    grid = {
        "datasets": list(recipes.DATASET_IDS),
        "backbones": list(recipes.BACKBONES),
        "replicate_seeds": list(recipes.REPLICATE_SEEDS),
        "budgets": list(recipes.BUDGETS),
        "selectors": list(recipes.SELECTORS),
        "heads": list(recipes.HEADS),
        "panel_count": 500,
    }
    normalized_environment = {**environment, "thread_environment": threads}
    shared_metadata = {
        "schema_version": 1,
        "study": "fused3_backbone_ranking_confirmation_v2",
        "artifact_status": "completed",
        "mode": "full",
        "protocol_sha256": protocol_sha,
        "code_identity_sha256": sources["code_identity_sha256"],
        "run_identity_sha256": run_identity,
        "audited_registry_sha256": registry_sha,
        "lineage_lock_sha256": lock_sha,
        "environment": normalized_environment,
        "thread_environment": threads,
        "repository_provenance": sources["repository"],
        "source_hashes": sources["source_hashes"],
        "audited_registry": {
            "path": "audited_registry.json",
            "sha256": registry_sha,
            "registry_id": registry.get("registry_id"),
        },
        "grid": grid,
        "table_manifest": table_manifest,
        "checkpoint_count": len(checkpoints),
        "determinism_verification": determinism,
        "runtime_gate": {
            "status": "inherited_pass",
            "source": "hash_bound_fused3_full_food_resource_result",
            "confirmation_clocks": "descriptive_only",
        },
        "protocol": protocol,
    }
    raw = {
        **shared_metadata,
    }
    raw_path = Path(output) / "raw_results.json"
    _atomic_json(raw_path, raw)
    manifest = {
        **shared_metadata,
        "raw_results_sha256": sha256_path(raw_path),
    }
    _atomic_json(Path(output) / "manifest.json", manifest)
    (Path(output) / "running_manifest.json").unlink(missing_ok=True)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--lineage-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_confirmation(
        registry_path=args.registry,
        lineage_lock_path=args.lineage_lock,
        output=args.output,
        resume=bool(args.resume),
        smoke=bool(args.smoke),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EXPECTED_SCORE_ENVIRONMENT",
    "CANDIDATE_AUTHORITY_PATH",
    "PROTOCOL_PATH",
    "PROTOCOL_SIDECAR",
    "THREAD_ENVIRONMENT",
    "execute_panel",
    "panel_identities",
    "redact_smoke_panel",
    "run_confirmation",
    "source_identity",
    "validate_protocol",
]
