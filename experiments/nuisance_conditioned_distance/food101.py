"""Retrospective Food-101 replay for nuisance-conditioned OI candidates.

The runner reads the archived, hash-locked embedding caches and reference-head
outcomes.  It does not import or modify Vertebrae's scoring adapter: A/B are
instantiated directly from the current upstream :class:`OverlapIndex`, while
C/D/E use the private experiment-local :class:`ConditionedOverlapIndex`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter, process_time
from typing import Any

_THREAD_ENVIRONMENT = {
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "LOKY_MAX_CPU_COUNT": "1",
}
for _name, _value in _THREAD_ENVIRONMENT.items():
    os.environ[_name] = _value

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from overlapindex import OverlapIndex  # noqa: E402
from experiments.nuisance_conditioned_distance.conditioning_adapter import (  # noqa: E402
    ConditionedOverlapIndex,
)


ROOT = Path(__file__).resolve().parents[2]
ARCHIVED_ROOT = Path("/Users/niklasmelton/code/vertabrae")
DEFAULT_STEM = "food101_nonlinear_backbone_bridge_food101_k10_89095e3bc0db"
DEFAULT_DRIVER = ARCHIVED_ROOT / "examples/food101_nonlinear_backbone_bridge.py"
DEFAULT_RESULT = ARCHIVED_ROOT / "examples/output" / f"{DEFAULT_STEM}.json"
DEFAULT_COHORT = ARCHIVED_ROOT / "examples/output" / f"{DEFAULT_STEM}.cohorts.json"
DEFAULT_CACHE = ARCHIVED_ROOT / "examples/output/cache"
DEFAULT_PRIOR = Path(
    "/Users/niklasmelton/.codex/worktrees/a9b5/vertabrae/examples/research/"
    "selector_replay/artifacts/full/raw_results.json"
)
PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
STARTING_COMMIT = "165fa344653a72dd2abc04a20387d90b891b9acc"
FIRST_PROTOCOL_COMMIT = "7b3faee12dcb65e3c971e5c115b66fad16031423"
EXPECTED_BRANCH = "codex/experiment-nuisance-conditioned-distance"
EXPERIMENT_SOURCE_PATHS = (
    "experiments/nuisance_conditioned_distance/conditioning_adapter.py",
    "experiments/nuisance_conditioned_distance/fixtures.py",
    "experiments/nuisance_conditioned_distance/runner.py",
    "experiments/nuisance_conditioned_distance/food101.py",
    "experiments/nuisance_conditioned_distance/analysis.py",
    "experiments/nuisance_conditioned_distance/reporting.py",
    "experiments/nuisance_conditioned_distance/protocol.json",
    "experiments/nuisance_conditioned_distance/protocol.sha256",
)

MODELS = (
    "dinov2-small",
    "deit-tiny",
    "convnext-tiny",
    "mobilenetv3-large",
    "openclip-vit-b-32",
    "resnet50",
    "efficientnet-b0",
    "swin-tiny",
    "vit-small-16",
    "densenet121",
)
ARMS = (
    ("baseline", 0.0, 0.0),
    ("nonlinearity_full", 1.0, 0.0),
    ("nuisance_full", 0.0, 1.5),
)
BUDGETS = (64, 68, 72, 80)
REPLICATES = tuple(range(5))
FOLDS = 5
K = 10
SEED = 42
CAP_ROWS = 2048
SOURCE_CONFIGURATION_HASH = "89095e3bc0db391235a71489fdb86013a8503a113673ce7a757989a27cd05529"
DRIVER_SHA256 = "63f906f12070aaf697d88c193dd11dc2f430bc9fa6b0389756d556636ad85eae"
RESULT_SHA256 = "0826871de7a37d72ad6618e48a39b4895c74bb6ab30d9eb7521a2b767e483b55"
COHORT_SHA256 = "038c5ecff1fa42b4a71bb235a35fb562d2a9542ce52caa336566be56d0f9d88d"
PRIOR_SHA256 = "56d394e8c7f07a83e37cc14e0c33fd6188c2020c4011f3b5d635a2d13934586d"

CANDIDATES = (
    ("A", "oi_unrefined_raw", "none", False, True),
    ("B", "oi_refined_raw", "none", True, True),
    ("C", "oi_refined_global_isotropy", "global_isotropy", True, False),
    ("D", "oi_refined_pooled_diagonal", "pooled_diagonal", True, False),
    ("E", "oi_refined_pooled_full", "pooled_full", True, False),
)
PROMOTION_CANDIDATES = ("C", "D", "E")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    parser.add_argument("--source-result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--source-cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--prior-replay", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--replicates", default=",".join(str(value) for value in REPLICATES))
    parser.add_argument("--budgets", default=",".join(str(value) for value in BUDGETS))
    parser.add_argument("--arms", default=",".join(name for name, _, _ in ARMS))
    parser.add_argument("--candidates", default=",".join(value[0] for value in CANDIDATES))
    parser.add_argument(
        "--promotion-decision",
        type=Path,
        default=None,
        help="hashed screen promotion_decision.json used to configure comparator G",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), sort_keys=True, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_json_safe(row.get(key)), sort_keys=True)
                    if isinstance(row.get(key), (dict, list, tuple))
                    else _json_safe(row.get(key))
                    for key in fields
                }
            )


def _parse_subset(raw: str, allowed: Sequence[Any], cast: Any = str) -> tuple[Any, ...]:
    result = tuple(cast(value.strip()) for value in raw.split(",") if value.strip())
    if not result or len(set(result)) != len(result) or any(value not in allowed for value in result):
        raise ValueError(f"Invalid subset {result!r}; expected values from {tuple(allowed)!r}.")
    return result


def _load_driver(path: Path) -> Any:
    if _sha256(path) != DRIVER_SHA256:
        raise ValueError("Frozen Food-101 driver hash mismatch.")
    specification = importlib.util.spec_from_file_location("_frozen_food101_bridge", path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load frozen driver at {path}.")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _validate_sources(result: Mapping[str, Any], cohort: Mapping[str, Any]) -> None:
    if result.get("artifact_status") != "completed":
        raise ValueError("Frozen source result is not complete.")
    if result.get("configuration_hash") != SOURCE_CONFIGURATION_HASH:
        raise ValueError("Frozen source result configuration hash mismatch.")
    if cohort.get("configuration_hash") != SOURCE_CONFIGURATION_HASH:
        raise ValueError("Frozen source cohort configuration hash mismatch.")
    if len(cohort.get("extracted_sample_ids", ())) != 28_480:
        raise ValueError("Frozen Food-101 cohort must contain exactly 28,480 rows.")


def _load_cache(
    cache_dir: Path,
    model: str,
    expected_sample_hash: str,
    *,
    verify_hash: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest_path = cache_dir / f"food101_{model}_final.json"
    manifest = _read_json(manifest_path)
    if manifest.get("model") != model:
        raise ValueError(f"Cache model mismatch for {model}.")
    if manifest.get("sample_ids_sha256") != expected_sample_hash:
        raise ValueError(f"Cache sample identity mismatch for {model}.")
    matrix_path = Path(str(manifest["path"]))
    if not matrix_path.is_absolute():
        candidate = manifest_path.with_name(matrix_path.name)
        matrix_path = candidate if candidate.exists() else ARCHIVED_ROOT / matrix_path
    if verify_hash and _sha256(matrix_path) != str(manifest["sha256"]):
        raise ValueError(f"Cache SHA-256 mismatch for {model}.")
    matrix = np.load(matrix_path, mmap_mode="r")
    if list(matrix.shape) != list(manifest["shape"]):
        raise ValueError(f"Cache shape mismatch for {model}.")
    return matrix, manifest


def _row_l2(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


def _stratification_encoding(labels: np.ndarray) -> tuple[np.ndarray, tuple[Any, ...]]:
    """Encode scalar labels deterministically for scikit-learn splitting only.

    Older scikit-learn releases reject object arrays containing otherwise valid
    integer labels.  OI must still receive the caller's original labels, so this
    first-observed integer encoding is used only to plan stratified fold indices.
    """

    target = np.asarray(labels)
    if target.ndim != 1:
        raise ValueError("labels must be a one-dimensional scalar-label array")
    encoded = np.empty(target.shape[0], dtype=np.int64)
    class_to_index: dict[Any, int] = {}
    classes: list[Any] = []
    for row, raw_label in enumerate(target):
        label = raw_label.item() if isinstance(raw_label, np.generic) else raw_label
        if isinstance(label, (list, tuple, set, frozenset, dict, np.ndarray)):
            raise ValueError("labels must contain hashable scalar values")
        try:
            class_index = class_to_index.get(label)
        except TypeError as exc:
            raise ValueError("labels must contain hashable scalar values") from exc
        if class_index is None:
            class_index = len(classes)
            try:
                class_to_index[label] = class_index
            except TypeError as exc:
                raise ValueError("labels must contain hashable scalar values") from exc
            classes.append(label)
        encoded[row] = class_index
    return encoded, tuple(classes)


def _stratified_folds(
    labels: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return deterministic folds without exposing object labels to sklearn."""

    encoded, _ = _stratification_encoding(labels)
    splitter = StratifiedKFold(
        n_splits=int(n_splits), shuffle=True, random_state=int(seed)
    )
    placeholder = np.zeros((encoded.size, 1), dtype=np.uint8)
    return tuple(
        (np.asarray(train, dtype=np.int64), np.asarray(holdout, dtype=np.int64))
        for train, holdout in splitter.split(placeholder, encoded)
    )


def _oi_kwargs(k_per_class: Mapping[Any, int], seed: int) -> dict[str, Any]:
    """Return a fresh, non-aliased mapping for one candidate and fold."""

    return {
        "model_type": "MiniBatchKMeans",
        "kmeans_k": dict(k_per_class),
        "kmeans_kwargs": {"random_state": int(seed)},
    }


def _refinement_summary(model: Any) -> dict[str, Any]:
    direct = getattr(model, "prototype_refinement_", None)
    if isinstance(direct, Mapping):
        return dict(direct)
    return {}


def _conditioning_summary(model: Any) -> dict[str, Any]:
    value = getattr(model, "conditioning_diagnostics_", None)
    return dict(value) if isinstance(value, Mapping) else {}


def _conditioning_runtime_summary(model: Any) -> dict[str, Any]:
    value = getattr(model, "runtime_diagnostics_", None)
    return dict(value) if isinstance(value, Mapping) else {}


def _cross_fit_determinism_signature(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the structural cross-fit result, excluding every clock field."""

    folds: list[dict[str, Any]] = []
    for fold in result.get("folds", ()):
        if not isinstance(fold, Mapping):
            continue
        folds.append(
            {
                "fold": fold.get("fold"),
                "seed": fold.get("seed"),
                "train_size": fold.get("train_size"),
                "holdout_size": fold.get("holdout_size"),
                "score": fold.get("score"),
                "refinement": fold.get("refinement"),
                "conditioning": fold.get("conditioning"),
            }
        )
    return _json_safe(
        {
            "candidate_id": result.get("candidate_id"),
            "candidate_name": result.get("candidate_name"),
            "mode": result.get("mode"),
            "prototype_refinement": result.get("prototype_refinement"),
            "score": result.get("score"),
            "refinement": result.get("refinement"),
            "conditioning": result.get("conditioning"),
            "folds": folds,
        }
    )


def _cross_fit_determinism_comparison(
    warmup: Mapping[str, Any], measured: Mapping[str, Any]
) -> dict[str, Any]:
    first = _cross_fit_determinism_signature(warmup)
    second = _cross_fit_determinism_signature(measured)
    first_bytes = json.dumps(first, sort_keys=True, separators=(",", ":")).encode()
    second_bytes = json.dumps(second, sort_keys=True, separators=(",", ":")).encode()
    return {
        "exact": first_bytes == second_bytes,
        "warmup_signature_sha256": hashlib.sha256(first_bytes).hexdigest(),
        "measured_signature_sha256": hashlib.sha256(second_bytes).hexdigest(),
        "runtime_fields_excluded": True,
    }


def _cross_fitted_score(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    candidate: tuple[str, str, str, bool, bool],
    seed: int,
) -> dict[str, Any]:
    candidate_id, name, mode, refined, direct = candidate
    if type(refined) is not bool:
        raise TypeError("candidate refinement must be a strict Python bool")
    values = _row_l2(matrix)
    target = np.asarray(labels)
    encoded_target, classes = _stratification_encoding(target)
    counts = np.bincount(encoded_target, minlength=len(classes))
    if int(np.min(counts)) < FOLDS:
        raise ValueError("Every class must have at least five rows for cross-fitting.")
    planned_folds = _stratified_folds(target, n_splits=FOLDS, seed=int(seed))
    scores: list[float] = []
    folds: list[dict[str, Any]] = []
    totals = {
        "prototype_count_before": 0,
        "prototype_count_after": 0,
        "eligible_count": 0,
        "attempted_count": 0,
        "applied_count": 0,
        "skipped_count": 0,
    }
    conditioning_rows: list[dict[str, Any]] = []
    for fold, (train, holdout) in enumerate(planned_folds):
        fold_seed = int(seed) + int(fold)
        train_labels = target[train]
        train_encoded = encoded_target[train]
        train_counts = {
            label: int(np.count_nonzero(train_encoded == class_index))
            for class_index, label in enumerate(classes)
        }
        k_per_class = {
            label: min(K, max(1, count // 5), count) for label, count in train_counts.items()
        }
        kwargs = _oi_kwargs(k_per_class, fold_seed)
        if direct:
            estimator: Any = OverlapIndex(
                prototype_refinement=refined,
                **kwargs,
            )
        else:
            estimator = ConditionedOverlapIndex(
                mode=mode,
                prototype_refinement=refined,
                conditioning_kwargs={
                    "condition_number_cap": 10_000.0,
                    "relative_eigenvalue_floor": 1e-8,
                },
                overlap_index_kwargs=kwargs,
            )
        fit_wall = perf_counter()
        fit_cpu = process_time()
        estimator.fit(values[train], train_labels)
        fit_wall_seconds = perf_counter() - fit_wall
        fit_cpu_seconds = process_time() - fit_cpu
        score_wall = perf_counter()
        score_cpu = process_time()
        score = float(estimator.score_fixed(values[holdout], target[holdout]))
        score_wall_seconds = perf_counter() - score_wall
        score_cpu_seconds = process_time() - score_cpu
        scores.append(score)
        refinement = _refinement_summary(estimator)
        conditioning = _conditioning_summary(estimator)
        conditioning_runtime = _conditioning_runtime_summary(estimator)
        conditioning_rows.append(conditioning)
        for field in totals:
            totals[field] += int(refinement.get(field, 0) or 0)
        folds.append(
            {
                "fold": fold,
                "seed": fold_seed,
                "train_size": int(train.size),
                "holdout_size": int(holdout.size),
                "score": score,
                "fit_wall_seconds": float(fit_wall_seconds),
                "fit_cpu_seconds": float(fit_cpu_seconds),
                "score_fixed_wall_seconds": float(score_wall_seconds),
                "score_fixed_cpu_seconds": float(score_cpu_seconds),
                "refinement": refinement,
                "conditioning": conditioning,
                "conditioning_runtime": conditioning_runtime,
            }
        )
    before = totals["prototype_count_before"]
    return {
        "candidate_id": candidate_id,
        "candidate_name": name,
        "mode": mode,
        "prototype_refinement": refined,
        "score": float(np.mean(scores)),
        "fit_wall_seconds": float(sum(row["fit_wall_seconds"] for row in folds)),
        "fit_cpu_seconds": float(sum(row["fit_cpu_seconds"] for row in folds)),
        "score_fixed_wall_seconds": float(
            sum(row["score_fixed_wall_seconds"] for row in folds)
        ),
        "score_fixed_cpu_seconds": float(
            sum(row["score_fixed_cpu_seconds"] for row in folds)
        ),
        "refinement": {
            **totals,
            "applied_rate": float(totals["applied_count"] / before) if before else 0.0,
        },
        "conditioning": conditioning_rows,
        "folds": folds,
    }


def _stratified_cap(labels: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    target = np.asarray(labels)
    if len(target) <= maximum:
        return np.arange(len(target), dtype=int)
    encoded, classes = _stratification_encoding(target)
    base, extra = divmod(int(maximum), len(classes))
    rows: list[np.ndarray] = []
    for position, _label in enumerate(classes):
        available = np.flatnonzero(encoded == position)
        count = base + int(position < extra)
        generator = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(position), int(len(target))])
        )
        rows.append(generator.permutation(available)[:count])
    return np.sort(np.concatenate(rows).astype(int))


def _capped_probe(matrix: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, Any]:
    selected = _stratified_cap(labels, CAP_ROWS, seed)
    values = np.asarray(matrix[selected], dtype=np.float32)
    target = np.asarray(labels[selected])
    encoded_target, _ = _stratification_encoding(target)
    predictions = np.empty(encoded_target.size, dtype=np.int64)
    wall_start, cpu_start = perf_counter(), process_time()
    for train, holdout in _stratified_folds(
        encoded_target, n_splits=FOLDS, seed=int(seed)
    ):
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2_000, random_state=int(seed), n_jobs=1),
        )
        estimator.fit(values[train], encoded_target[train])
        predictions[holdout] = estimator.predict(values[holdout])
    return {
        "score": float(accuracy_score(encoded_target, predictions)),
        "n_rows": int(encoded_target.size),
        "wall_seconds": float(perf_counter() - wall_start),
        "cpu_seconds": float(process_time() - cpu_start),
    }


def _guardrail_policy_rows(
    selector_rows: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    *,
    promoted_candidate: str,
) -> list[dict[str, Any]]:
    """Apply the frozen G trigger once per model-selection panel."""

    if promoted_candidate not in PROMOTION_CANDIDATES:
        raise ValueError(
            f"promoted_candidate must be one of {PROMOTION_CANDIDATES!r}"
        )
    panel_fields = ("replicate", "arm", "budget")
    selector_by_panel: dict[tuple[Any, ...], dict[str, dict[str, Mapping[str, Any]]]] = {}
    for row in selector_rows:
        if row.get("status") != "ok":
            continue
        panel = tuple(row.get(field) for field in panel_fields)
        model = str(row.get("model"))
        candidate = str(row.get("candidate_id"))
        selector_by_panel.setdefault(panel, {}).setdefault(model, {})[candidate] = row
    probe_lookup = {
        (
            row.get("replicate"), row.get("arm"), row.get("budget"),
            str(row.get("model")),
        ): row
        for row in probe_rows
    }
    output: list[dict[str, Any]] = []
    for panel, by_model in sorted(selector_by_panel.items(), key=repr):
        trigger_models: list[dict[str, Any]] = []
        for model, by_candidate in sorted(by_model.items()):
            b_row = by_candidate.get("B")
            e_row = by_candidate.get("E")
            if b_row is None or e_row is None:
                raise ValueError("G requires complete B and E diagnostics for every model")
            refinement = b_row.get("refinement", {})
            refinement_rate = float(refinement.get("applied_rate", 0.0))
            disagreement = abs(float(b_row["score"]) - float(e_row["score"]))
            if refinement_rate >= 0.50 or disagreement >= 0.05:
                trigger_models.append(
                    {
                        "model": model,
                        "b_refinement_applied_rate": refinement_rate,
                        "abs_b_minus_e_score": disagreement,
                    }
                )
        triggered = bool(trigger_models)
        for model, by_candidate in sorted(by_model.items()):
            promoted = by_candidate.get(promoted_candidate)
            if promoted is None:
                raise ValueError(
                    f"G requires promoted candidate {promoted_candidate} for every model"
                )
            probe = probe_lookup.get((*panel, model))
            if probe is None:
                raise ValueError("G requires one capped-probe component for every model")
            required_oi = {"B", "E"}
            if not triggered:
                required_oi.add(promoted_candidate)
            oi_wall = sum(
                float(by_candidate[candidate].get("outer_wall_seconds", 0.0))
                for candidate in required_oi
            )
            oi_cpu = sum(
                float(by_candidate[candidate].get("outer_cpu_seconds", 0.0))
                for candidate in required_oi
            )
            source = "capped_linear_probe" if triggered else promoted_candidate
            selected = probe if triggered else promoted
            output.append(
                {
                    "candidate_id": "G",
                    "candidate_name": "panel_selective_capped_probe_guardrail",
                    "model": model,
                    "backbone": model,
                    "replicate": int(panel[0]),
                    "arm": str(panel[1]),
                    "budget": int(panel[2]),
                    "panel_triggered": triggered,
                    "trigger_models": trigger_models,
                    "trigger_b_refinement_threshold": 0.50,
                    "trigger_abs_b_minus_e_threshold": 0.05,
                    "promoted_candidate": promoted_candidate,
                    "selected_score_source": source,
                    "score": float(selected["score"]),
                    "policy_wall_seconds": float(
                        oi_wall + (float(probe.get("wall_seconds", 0.0)) if triggered else 0.0)
                    ),
                    "policy_cpu_seconds": float(
                        oi_cpu + (float(probe.get("cpu_seconds", 0.0)) if triggered else 0.0)
                    ),
                    "status": "ok",
                }
            )
    return output


def _git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.PIPE
    ).strip()


def _git_head() -> str | None:
    try:
        return _git_output("rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        return None


def _repository_provenance() -> dict[str, Any]:
    """Verify the recorded develop base and hash all executable recipe files."""

    try:
        branch = _git_output("branch", "--show-current")
        develop = _git_output("rev-parse", "develop")
        merge_base = _git_output("merge-base", "HEAD", "develop")
        freeze_parent = _git_output("rev-parse", f"{FIRST_PROTOCOL_COMMIT}^")
        protocol_commit = _git_output(
            "log", "-1", "--format=%H", "--",
            "experiments/nuisance_conditioned_distance/protocol.json",
        )
        dirty_rows = _git_output("status", "--porcelain=v1").splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not verify experiment repository provenance.") from exc
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(
            f"Experiment must run on {EXPECTED_BRANCH!r}; found {branch!r}."
        )
    if (
        develop != STARTING_COMMIT
        or merge_base != STARTING_COMMIT
        or freeze_parent != STARTING_COMMIT
    ):
        raise RuntimeError(
            "Recorded develop starting point mismatch: expected "
            f"{STARTING_COMMIT}, develop={develop}, merge-base={merge_base}, "
            f"first-protocol-parent={freeze_parent}."
        )
    source_hashes = {}
    for relative in EXPERIMENT_SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Required experiment recipe file is missing: {relative}")
        source_hashes[relative] = _sha256(path)
    code_identity_sha256 = hashlib.sha256(
        json.dumps(
            {
                "schema": "declared_experiment_sources_v1",
                "starting_commit": STARTING_COMMIT,
                "source_hashes": source_hashes,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "branch": branch,
        "starting_commit": STARTING_COMMIT,
        "develop_commit_at_execution": develop,
        "merge_base_with_develop": merge_base,
        "first_protocol_commit": FIRST_PROTOCOL_COMMIT,
        "first_protocol_parent": freeze_parent,
        "protocol_commit": protocol_commit,
        "experiment_commit": _git_head(),
        "git_dirty": bool(dirty_rows),
        "git_status_porcelain": dirty_rows,
        "experiment_source_sha256": source_hashes,
        "code_identity_sha256": code_identity_sha256,
    }


def _environment(provenance: Mapping[str, Any]) -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "overlapindex"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "thread_environment": dict(_THREAD_ENVIRONMENT),
        "starting_commit": STARTING_COMMIT,
        "protocol_commit": provenance.get("protocol_commit"),
        "experiment_commit": provenance.get("experiment_commit"),
        "git_dirty": provenance.get("git_dirty"),
    }


def _write_food_artifacts(
    *,
    output: Path,
    artifact_status: str,
    stop_reason: str | None,
    stop_error: str | None,
    args: argparse.Namespace,
    models: Sequence[str],
    replicates: Sequence[int],
    budgets: Sequence[int],
    arm_names: Sequence[str],
    candidate_ids: Sequence[str],
    promoted_candidate: str | None,
    promotion_decision_sha256: str | None,
    provenance: Mapping[str, Any],
    source_paths: Mapping[str, tuple[Path, str]],
    cache_identity: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    guardrail_rows: Sequence[Mapping[str, Any]],
    prior_full_probe_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    parity: Sequence[Mapping[str, Any]],
    deterministic_repeats: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Write one complete, analyzable Food artifact bundle.

    This is also the failure-path writer.  Candidate, determinism, and frozen
    baseline-parity stops must leave the accumulated rows and provenance at the
    top level before the caller raises.  Each individual file is replaced via
    :func:`_write_json`'s temporary-file protocol, so a stopped run cannot
    expose a half-written JSON document to resume/analysis tooling.
    """

    raw_result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_status": str(artifact_status),
        "study": "food101_nuisance_conditioned_distance",
        "retrospective": True,
        "configuration": {
            "models": list(models),
            "replicates": list(replicates),
            "budgets": list(budgets),
            "arms": list(arm_names),
            "candidates": list(candidate_ids),
            "promoted_candidate_for_G": promoted_candidate,
            "promotion_decision": (
                {
                    "path": str(args.promotion_decision.resolve()),
                    "sha256": promotion_decision_sha256,
                }
                if args.promotion_decision is not None
                else None
            ),
            "folds": FOLDS,
            "k": K,
            "seed": SEED,
            "capped_probe_maximum_rows": CAP_ROWS,
            "counterbalanced": True,
            "serial_one_thread": True,
            "warmup_excluded": True,
            "smoke": bool(args.smoke),
            "cache_matrix_sha256_verified": True,
        },
        "environment": _environment(provenance),
        "repository_provenance": dict(provenance),
        "protocol": {
            "path": str(PROTOCOL_PATH),
            "sha256": _sha256(PROTOCOL_PATH),
        },
        "sources": {
            name: {"path": str(path), "sha256": expected}
            for name, (path, expected) in source_paths.items()
        },
        "cache_identity": dict(cache_identity),
        "selector_rows": [dict(row) for row in rows],
        "capped_probe_rows": [dict(row) for row in probe_rows],
        "guardrail_rows": [dict(row) for row in guardrail_rows],
        "prior_full_probe_rows": [dict(row) for row in prior_full_probe_rows],
        "reference_rows": [dict(row) for row in reference_rows],
        "baseline_parity_rows": [dict(row) for row in parity],
        "baseline_parity": {
            "n": len(parity),
            "exact": bool(parity) and all(row["exact"] for row in parity),
            "max_absolute_delta": max(
                (abs(float(row["delta"])) for row in parity), default=None
            ),
        },
        "determinism_verification": {
            "status": (
                "pass"
                if set(deterministic_repeats) == set(candidate_ids)
                and all(
                    value.get("exact") is True
                    for value in deterministic_repeats.values()
                )
                else "inconclusive"
            ),
            "exact": (
                True
                if set(deterministic_repeats) == set(candidate_ids)
                and all(
                    value.get("exact") is True
                    for value in deterministic_repeats.values()
                )
                else None
            ),
            "basis": "excluded per-candidate first warmup versus identical first measured Food cell",
            "runtime_fields_excluded": True,
            "candidates": {
                str(candidate_id): dict(comparison)
                for candidate_id, comparison in deterministic_repeats.items()
            },
        },
        "deviations": [
            "The panel is retrospective development evidence, not untouched confirmation.",
            "Peak memory requires a separate fresh-process benchmark and is not inferred here.",
            "The capped probe component is measured for G but kept separate from OI candidates.",
            "Capped-probe components are measured for every model to support paired runtime analysis; G applies them only in panels satisfying the frozen trigger.",
        ],
    }
    if stop_reason is not None:
        raw_result["stop_reason"] = str(stop_reason)
        raw_result["stop_error"] = str(stop_error) if stop_error is not None else None

    _write_json(output / "raw_results.json", raw_result)
    manifest = {
        key: raw_result[key]
        for key in (
            "schema_version",
            "artifact_status",
            "study",
            "configuration",
            "environment",
            "repository_provenance",
            "protocol",
            "sources",
            "cache_identity",
            "baseline_parity",
            "determinism_verification",
            "deviations",
        )
    }
    if stop_reason is not None:
        manifest["stop_reason"] = raw_result["stop_reason"]
        manifest["stop_error"] = raw_result["stop_error"]
    _write_json(output / "manifest.json", manifest)
    _write_csv(output / "selector_rows.csv", rows)
    _write_csv(output / "capped_probe_rows.csv", probe_rows)
    _write_csv(output / "guardrail_rows.csv", guardrail_rows)
    _write_csv(output / "baseline_parity_rows.csv", parity)
    return raw_result


def _baseline_parity_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    old_method: Mapping[str, str],
    prior_lookup: Mapping[tuple[str, int, str, int, str], float],
) -> list[dict[str, Any]]:
    """Materialize parity observations already available in ``rows``."""

    parity: list[dict[str, Any]] = []
    for row in rows:
        method = old_method.get(str(row.get("candidate_id")))
        if method is None or row.get("score") is None:
            continue
        key = (
            str(row["backbone"]),
            int(row["replicate"]),
            str(row["arm"]),
            int(row["budget"]),
            method,
        )
        if key in prior_lookup:
            delta = float(row["score"]) - prior_lookup[key]
            parity.append(
                {
                    "candidate_id": row["candidate_id"],
                    "model": row["model"],
                    "replicate": row["replicate"],
                    "arm": row["arm"],
                    "budget": row["budget"],
                    "new_score": row["score"],
                    "prior_score": prior_lookup[key],
                    "delta": delta,
                    "exact": delta == 0.0,
                }
            )
    return parity


def _run(args: argparse.Namespace) -> int:
    provenance = _repository_provenance()
    models = _parse_subset(args.models, MODELS)
    replicates = _parse_subset(args.replicates, REPLICATES, int)
    budgets = _parse_subset(args.budgets, BUDGETS, int)
    arm_names = _parse_subset(args.arms, tuple(value[0] for value in ARMS))
    candidate_ids = _parse_subset(args.candidates, tuple(value[0] for value in CANDIDATES))
    promotion_decision: dict[str, Any] | None = None
    promotion_decision_sha256: str | None = None
    promoted_candidate: str | None = None
    if args.promotion_decision is not None:
        promotion_path = args.promotion_decision.resolve()
        promotion_decision = _read_json(promotion_path)
        promotion_decision_sha256 = _sha256(promotion_path)
        promoted_candidate = promotion_decision.get("selected_candidate")
        if (
            promotion_decision.get("status") != "promoted_for_full_evaluation"
            or promotion_decision.get("decision_stage") != "screen"
            or promoted_candidate not in PROMOTION_CANDIDATES
        ):
            raise ValueError(
                "Promotion decision is not a valid screen-locked C/D/E full-evaluation decision."
            )
    if not args.smoke and promotion_decision is None:
        raise ValueError(
            "--promotion-decision is required for a full Food-101 run so "
            "product-policy comparator G is bound to the frozen screen decision"
        )
    if not args.smoke:
        expected = {
            "models": MODELS,
            "replicates": REPLICATES,
            "budgets": BUDGETS,
            "arms": tuple(value[0] for value in ARMS),
            "candidates": tuple(value[0] for value in CANDIDATES),
        }
        received = {
            "models": models,
            "replicates": replicates,
            "budgets": budgets,
            "arms": arm_names,
            "candidates": candidate_ids,
        }
        if received != expected:
            raise ValueError(
                "Full Food-101 replay requires the exact frozen models, "
                "replicates, budgets, arms, and A-E candidates; use --smoke "
                "for bounded subsets."
            )
    if args.smoke:
        models, replicates, budgets = models[:1], replicates[:1], budgets[:1]
        if "nuisance_full" in arm_names:
            arm_names = ("nuisance_full",)
        else:
            arm_names = arm_names[:1]

    source_paths = {
        "driver": (args.driver.resolve(), DRIVER_SHA256),
        "source_result": (args.source_result.resolve(), RESULT_SHA256),
        "source_cohort": (args.source_cohort.resolve(), COHORT_SHA256),
        "prior_replay": (args.prior_replay.resolve(), PRIOR_SHA256),
    }
    for name, (path, expected) in source_paths.items():
        received = _sha256(path)
        if received != expected:
            raise ValueError(f"{name} hash mismatch: expected {expected}, got {received}.")

    driver = _load_driver(args.driver.resolve())
    source_result = _read_json(args.source_result.resolve())
    cohort = _read_json(args.source_cohort.resolve())
    prior = _read_json(args.prior_replay.resolve())
    _validate_sources(source_result, cohort)
    sample_ids = [str(value) for value in cohort["extracted_sample_ids"]]
    sample_hash = hashlib.sha256(json.dumps(sample_ids).encode()).hexdigest()
    labels = np.asarray([value.split("/")[2] for value in sample_ids], dtype=object)
    roles = {int(key): value for key, value in cohort["roles"].items()}
    candidate_lookup = {value[0]: value for value in CANDIDATES}
    selected_candidates = tuple(candidate_lookup[value] for value in candidate_ids)
    arm_lookup = {name: (lam, nu) for name, lam, nu in ARMS}
    prior_lookup = {
        (
            str(row["backbone"]), int(row["replicate"]), str(row["arm"]),
            int(row["budget"]), str(row["method"]),
        ): float(row["score"])
        for row in prior.get("selector_rows", ())
    }
    old_method = {
        "A": "overlap_unrefined_cross_fitted",
        "B": "overlap_refined_cross_fitted",
    }
    # These source/reference rows are available before any candidate cell is
    # fitted, so a stopped run can still carry the same analyzable reference
    # surface as a completed run.
    reference_rows = [
        dict(row) for row in source_result.get("reference_rows", ())
        if str(row.get("backbone")) in models
        and int(row.get("replicate")) in replicates
        and str(row.get("arm")) in arm_names
    ]
    prior_full_probe_rows = [
        dict(row) for row in prior.get("selector_rows", ())
        if str(row.get("backbone")) in models
        and int(row.get("replicate")) in replicates
        and str(row.get("arm")) in arm_names
        and int(row.get("budget")) in budgets
        and str(row.get("method")) == "linear_probe_oof"
    ]

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    cache_identity: dict[str, Any] = {}
    warmed_candidates: set[str] = set()
    deterministic_repeats: dict[str, dict[str, Any]] = {}
    for model_position, model in enumerate(models):
        matrix, cache_manifest = _load_cache(
            args.cache_dir.resolve(), model, sample_hash,
            verify_hash=True,
        )
        cache_identity[model] = {
            "path": cache_manifest.get("path"),
            "sha256": cache_manifest.get("sha256"),
            "shape": cache_manifest.get("shape"),
            "sample_ids_sha256": cache_manifest.get("sample_ids_sha256"),
        }
        for replicate in replicates:
            role = roles[int(replicate)]
            indices = np.asarray(role["selector"], dtype=np.int64)
            raw = np.asarray(matrix[indices], dtype=np.float32)
            target = labels[indices]
            banks = driver._paired_split_banks(raw, target, seed=SEED + int(replicate) + 11)
            nested = driver._nested_stratified_indices(
                target, budgets, seed=SEED + int(replicate) + 23
            )
            for arm_position, arm in enumerate(arm_names):
                lam, nu = arm_lookup[arm]
                transformed = driver._bridge_transform(
                    raw,
                    donor=banks["donor"],
                    mode=banks["mode"],
                    nuisance=banks["nuisance"],
                    q=1.0,
                    lam=lam,
                    nu=nu,
                )
                for budget_position, budget in enumerate(budgets):
                    selected = nested[int(budget)]
                    subset = transformed[selected]
                    subset_target = target[selected]
                    checkpoint = checkpoint_dir / (
                        f"{model}__replicate-{replicate}__{arm}__budget-{budget}.json"
                    )
                    checkpoint_identity = {
                        "code_identity_sha256": provenance["code_identity_sha256"],
                        "protocol_sha256": _sha256(PROTOCOL_PATH),
                        "external_source_sha256": {
                            name: expected for name, (_path, expected) in source_paths.items()
                        },
                        "cache_matrix_sha256": cache_manifest.get("sha256"),
                        "sample_ids_sha256": sample_hash,
                        "model": model,
                        "replicate": int(replicate),
                        "arm": arm,
                        "budget": int(budget),
                        "candidate_ids": [candidate[0] for candidate in selected_candidates],
                        "promotion_decision_sha256": promotion_decision_sha256,
                        "seed": SEED + int(replicate),
                    }
                    if args.resume and checkpoint.exists():
                        cached = _read_json(checkpoint)
                        if cached.get("identity") != checkpoint_identity:
                            raise RuntimeError(
                                f"Resume checkpoint identity mismatch: {checkpoint}"
                            )
                        cached_rows_raw = cached.get("rows", ())
                        cached_probe_rows_raw = cached.get("probe_rows", ())
                        cached_rows = tuple(
                            row for row in cached_rows_raw
                            if isinstance(row, Mapping)
                        ) if isinstance(cached_rows_raw, Sequence) and not isinstance(
                            cached_rows_raw, (str, bytes)
                        ) else ()
                        cached_probe_rows = tuple(
                            row for row in cached_probe_rows_raw
                            if isinstance(row, Mapping)
                        ) if isinstance(cached_probe_rows_raw, Sequence) and not isinstance(
                            cached_probe_rows_raw, (str, bytes)
                        ) else ()
                        cached_status = cached.get("artifact_status")
                        expected_candidate_ids = {
                            str(candidate[0]) for candidate in selected_candidates
                        }
                        cached_repeats = cached.get("determinism_verification")
                        cached_repeats_valid = (
                            isinstance(cached_repeats, Mapping)
                            and set(cached_repeats) == expected_candidate_ids
                            and all(
                                isinstance(comparison, Mapping)
                                and comparison.get("exact") is True
                                for comparison in cached_repeats.values()
                            )
                        )
                        # Validate the checkpoint-local determinism record
                        # before consulting/merging the cumulative signatures.
                        # Otherwise a later incomplete checkpoint could be
                        # masked by an earlier complete one during resume.
                        invalid_reason: str | None = (
                            None
                            if cached_repeats_valid
                            else "resume_checkpoint_incomplete_determinism"
                        )
                        if cached_status != "completed":
                            invalid_reason = "resume_stopped_checkpoint"
                        elif (
                            not isinstance(cached_rows_raw, Sequence)
                            or isinstance(cached_rows_raw, (str, bytes))
                            or len(cached_rows) != len(cached_rows_raw)
                            or not cached_rows
                            or any(row.get("status") != "ok" for row in cached_rows)
                        ):
                            invalid_reason = "resume_checkpoint_failed_rows"
                        else:
                            cached_candidate_ids = {
                                str(row.get("candidate_id")) for row in cached_rows
                            }
                            probe_complete = (
                                isinstance(cached_probe_rows_raw, Sequence)
                                and not isinstance(cached_probe_rows_raw, (str, bytes))
                                and len(cached_probe_rows) == len(cached_probe_rows_raw) == 1
                                and str(cached_probe_rows[0].get("candidate_id"))
                                == "G_probe_component"
                                and cached_probe_rows[0].get("score") is not None
                                and cached_probe_rows[0].get("wall_seconds") is not None
                                and cached_probe_rows[0].get("cpu_seconds") is not None
                            )
                            if (
                                len(cached_rows) != len(expected_candidate_ids)
                                or cached_candidate_ids != expected_candidate_ids
                            ):
                                invalid_reason = (
                                    invalid_reason
                                    or "resume_checkpoint_incomplete_rows"
                                )
                            elif not probe_complete:
                                invalid_reason = (
                                    invalid_reason
                                    or "resume_checkpoint_incomplete_probe"
                                )
                        if cached_repeats_valid:
                            for candidate_id, comparison in cached_repeats.items():
                                existing = deterministic_repeats.get(str(candidate_id))
                                if existing is not None and existing != comparison:
                                    invalid_reason = (
                                        invalid_reason
                                        or "resume_checkpoint_determinism_mismatch"
                                    )
                                    continue
                                if isinstance(comparison, Mapping):
                                    deterministic_repeats[str(candidate_id)] = dict(
                                        comparison
                                    )
                            if (
                                set(deterministic_repeats) != expected_candidate_ids
                                or any(
                                    comparison.get("exact") is not True
                                    for comparison in deterministic_repeats.values()
                                )
                            ):
                                invalid_reason = (
                                    invalid_reason
                                    or "resume_checkpoint_incomplete_determinism"
                                )
                        if invalid_reason is not None:
                            rows.extend(cached_rows)
                            probe_rows.extend(cached_probe_rows)
                            stop_error = (
                                f"Cannot resume Food-101 checkpoint {checkpoint}: "
                                f"{invalid_reason}; artifact_status={cached_status!r}."
                            )
                            _write_food_artifacts(
                                output=output,
                                artifact_status="stopped",
                                stop_reason=invalid_reason,
                                stop_error=stop_error,
                                args=args,
                                models=models,
                                replicates=replicates,
                                budgets=budgets,
                                arm_names=arm_names,
                                candidate_ids=candidate_ids,
                                promoted_candidate=promoted_candidate,
                                promotion_decision_sha256=promotion_decision_sha256,
                                provenance=provenance,
                                source_paths=source_paths,
                                cache_identity=cache_identity,
                                rows=rows,
                                probe_rows=probe_rows,
                                guardrail_rows=(),
                                prior_full_probe_rows=prior_full_probe_rows,
                                reference_rows=reference_rows,
                                parity=_baseline_parity_rows(
                                    rows,
                                    old_method=old_method,
                                    prior_lookup=prior_lookup,
                                ),
                                deterministic_repeats=deterministic_repeats,
                            )
                            raise RuntimeError(stop_error)
                        rows.extend(cached_rows)
                        probe_rows.extend(cached_probe_rows)
                        continue
                    offset = (
                        model_position + int(replicate) + arm_position + budget_position
                    ) % len(selected_candidates)
                    execution = selected_candidates[offset:] + selected_candidates[:offset]
                    block: list[dict[str, Any]] = []
                    for order, candidate in enumerate(execution):
                        warmup_result: dict[str, Any] | None = None
                        started_wall: float | None = None
                        started_cpu: float | None = None
                        try:
                            if candidate[0] not in warmed_candidates:
                                warmup_result = _cross_fitted_score(
                                    subset,
                                    subset_target,
                                    candidate=candidate,
                                    seed=SEED + int(replicate),
                                )
                                warmed_candidates.add(candidate[0])
                            # The first call is a real warm-up and is excluded
                            # from the measured candidate clocks.
                            started_wall, started_cpu = perf_counter(), process_time()
                            result = _cross_fitted_score(
                                subset,
                                subset_target,
                                candidate=candidate,
                                seed=SEED + int(replicate),
                            )
                            if warmup_result is not None:
                                comparison = _cross_fit_determinism_comparison(
                                    warmup_result, result
                                )
                                deterministic_repeats[candidate[0]] = comparison
                                if not comparison["exact"]:
                                    raise RuntimeError(
                                        "nondeterministic warmup/measured fitted-state or score signature"
                                    )
                            status, error = "ok", None
                        except Exception as exc:  # preserve failed research rows
                            if started_wall is None or started_cpu is None:
                                started_wall, started_cpu = perf_counter(), process_time()
                            result = {
                                "candidate_id": candidate[0],
                                "candidate_name": candidate[1],
                                "mode": candidate[2],
                                "prototype_refinement": candidate[3],
                                "score": None,
                            }
                            status, error = "error", f"{type(exc).__name__}: {exc}"
                        assert started_wall is not None and started_cpu is not None
                        block.append(
                            {
                                "model": model,
                                "backbone": model,
                                "replicate": int(replicate),
                                "arm": arm,
                                "lambda": lam,
                                "nu": nu,
                                "budget": int(budget),
                                "n_rows": int(subset.shape[0]),
                                "n_features": int(subset.shape[1]),
                                "seed": SEED + int(replicate),
                                "execution_order": int(order),
                                "status": status,
                                "error": error,
                                "outer_wall_seconds": float(perf_counter() - started_wall),
                                "outer_cpu_seconds": float(process_time() - started_cpu),
                                **result,
                            }
                        )
                    failed_rows = [row for row in block if row.get("status") != "ok"]
                    if failed_rows:
                        # Preserve the failed paired cell for diagnosis, but
                        # do not continue into parity, probe, or later panels
                        # after a candidate error. This is an explicit frozen
                        # early-stop condition, not a missing observation to
                        # be silently dropped by analysis.
                        rows.extend(block)
                        failures = "; ".join(
                            f"{row.get('candidate_id')}: {row.get('error')}"
                            for row in failed_rows
                        )
                        stop_reason = (
                            "nondeterminism_failure"
                            if any(
                                "nondeterministic" in str(row.get("error", "")).lower()
                                for row in failed_rows
                            )
                            else "candidate_failure"
                        )
                        _write_json(
                            checkpoint,
                            {
                                "identity": checkpoint_identity,
                                "rows": block,
                                "probe_rows": [],
                                "determinism_verification": deterministic_repeats,
                                "artifact_status": "stopped_candidate_error",
                                "stop_reason": stop_reason,
                                "stop_error": failures,
                            },
                        )
                        _write_food_artifacts(
                            output=output,
                            artifact_status="stopped",
                            stop_reason=stop_reason,
                            stop_error=failures,
                            args=args,
                            models=models,
                            replicates=replicates,
                            budgets=budgets,
                            arm_names=arm_names,
                            candidate_ids=candidate_ids,
                            promoted_candidate=promoted_candidate,
                            promotion_decision_sha256=promotion_decision_sha256,
                            provenance=provenance,
                            source_paths=source_paths,
                            cache_identity=cache_identity,
                            rows=rows,
                            probe_rows=probe_rows,
                            guardrail_rows=(),
                            prior_full_probe_rows=prior_full_probe_rows,
                            reference_rows=reference_rows,
                            parity=_baseline_parity_rows(
                                rows, old_method=old_method, prior_lookup=prior_lookup
                            ),
                            deterministic_repeats=deterministic_repeats,
                        )
                        raise RuntimeError(
                            "Food-101 candidate cell failed; frozen early stop: "
                            f"{failures}"
                        )
                    rows.extend(block)
                    try:
                        for baseline_row in block:
                            method = old_method.get(str(baseline_row["candidate_id"]))
                            if method is None:
                                continue
                            parity_key = (
                                str(baseline_row["backbone"]),
                                int(baseline_row["replicate"]),
                                str(baseline_row["arm"]),
                                int(baseline_row["budget"]),
                                method,
                            )
                            expected_score = prior_lookup.get(parity_key)
                            if expected_score is None:
                                raise RuntimeError(
                                    f"Missing frozen A/B parity row for {parity_key!r}."
                                )
                            if baseline_row.get("score") != expected_score:
                                raise RuntimeError(
                                    "Frozen A/B parity mismatch for "
                                    f"{parity_key!r}: expected {expected_score!r}, "
                                    f"got {baseline_row.get('score')!r}."
                                )
                    except RuntimeError as exc:
                        parity_failure = str(exc)
                        _write_json(
                            checkpoint,
                            {
                                "identity": checkpoint_identity,
                                "rows": block,
                                "probe_rows": [],
                                "determinism_verification": deterministic_repeats,
                                "artifact_status": "stopped_parity_failure",
                                "stop_reason": "baseline_parity_failure",
                                "stop_error": parity_failure,
                            },
                        )
                        _write_food_artifacts(
                            output=output,
                            artifact_status="stopped",
                            stop_reason="baseline_parity_failure",
                            stop_error=parity_failure,
                            args=args,
                            models=models,
                            replicates=replicates,
                            budgets=budgets,
                            arm_names=arm_names,
                            candidate_ids=candidate_ids,
                            promoted_candidate=promoted_candidate,
                            promotion_decision_sha256=promotion_decision_sha256,
                            provenance=provenance,
                            source_paths=source_paths,
                            cache_identity=cache_identity,
                            rows=rows,
                            probe_rows=probe_rows,
                            guardrail_rows=(),
                            prior_full_probe_rows=prior_full_probe_rows,
                            reference_rows=reference_rows,
                            parity=_baseline_parity_rows(
                                rows, old_method=old_method, prior_lookup=prior_lookup
                            ),
                            deterministic_repeats=deterministic_repeats,
                        )
                        raise
                    probe = _capped_probe(
                        subset, subset_target, SEED + int(replicate)
                    )
                    probe_block = [{
                        "model": model,
                        "backbone": model,
                        "replicate": int(replicate),
                        "arm": arm,
                        "budget": int(budget),
                        "seed": SEED + int(replicate),
                        "candidate_id": "G_probe_component",
                        "candidate_name": "capped_linear_probe",
                        **probe,
                    }]
                    _write_json(
                        checkpoint,
                        {
                            "identity": checkpoint_identity,
                            "rows": block,
                            "probe_rows": probe_block,
                            "determinism_verification": deterministic_repeats,
                            "artifact_status": "completed",
                        },
                    )
                    probe_rows.extend(probe_block)

    parity = _baseline_parity_rows(
        rows, old_method=old_method, prior_lookup=prior_lookup
    )
    guardrail_rows = (
        _guardrail_policy_rows(
            rows, probe_rows, promoted_candidate=str(promoted_candidate)
        )
        if promoted_candidate is not None
        else []
    )
    artifact_status = (
        "completed" if all(row["status"] == "ok" for row in rows) else "partial"
    )
    raw_result = _write_food_artifacts(
        output=output,
        artifact_status=artifact_status,
        stop_reason=None,
        stop_error=None,
        args=args,
        models=models,
        replicates=replicates,
        budgets=budgets,
        arm_names=arm_names,
        candidate_ids=candidate_ids,
        promoted_candidate=promoted_candidate,
        promotion_decision_sha256=promotion_decision_sha256,
        provenance=provenance,
        source_paths=source_paths,
        cache_identity=cache_identity,
        rows=rows,
        probe_rows=probe_rows,
        guardrail_rows=guardrail_rows,
        prior_full_probe_rows=prior_full_probe_rows,
        reference_rows=reference_rows,
        parity=parity,
        deterministic_repeats=deterministic_repeats,
    )
    print(f"Food-101 replay complete: {output}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return _run(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
