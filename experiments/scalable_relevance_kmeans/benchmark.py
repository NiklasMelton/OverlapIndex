"""Bounded real-shape timing anchor for scalable relevance-weighted K-means."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from overlapindex import OverlapIndex

from experiments.m50_backbone_ranking import food101, runtime_benchmark
from experiments.scalable_relevance_kmeans.scalable import (
    ScalableOneUpdateRelevanceKMeans,
)


MODEL = "resnet50"
ARM = "baseline"
BUDGETS = (128, 640)
METHODS = ("raw_unrefined_oi", "scalable_one_update", "linear_probe")
THREAD_ENVIRONMENT = {
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "LOKY_MAX_CPU_COUNT": "1",
}
PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SIDECAR = PACKAGE_DIR / "protocol.sha256"
SOURCE_PATHS = (
    PACKAGE_DIR / "__init__.py",
    PACKAGE_DIR / "scalable.py",
    PACKAGE_DIR / "benchmark.py",
    PACKAGE_DIR / "synthetic_check.py",
    PROTOCOL_PATH,
    REPOSITORY_ROOT / "pyproject.toml",
    Path(food101.__file__).resolve(),
    Path(runtime_benchmark.__file__).resolve(),
) + tuple(sorted((REPOSITORY_ROOT / "overlapindex").glob("*.py")))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(_canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_frozen_protocol() -> str:
    if not PROTOCOL_PATH.is_file() or not PROTOCOL_SIDECAR.is_file():
        raise RuntimeError("protocol.json and protocol.sha256 must exist before timing")
    expected = PROTOCOL_SIDECAR.read_text(encoding="utf-8").strip()
    observed = _sha256(PROTOCOL_PATH)
    if len(expected) != 64 or expected != observed:
        raise RuntimeError("protocol sidecar does not match protocol.json")
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen_before_real_shape_timing":
        raise RuntimeError("protocol is not frozen for the real-shape timing anchor")
    return observed


def _validate_thread_environment() -> dict[str, str]:
    observed = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    invalid = {
        name: value
        for name, value in observed.items()
        if value != THREAD_ENVIRONMENT[name]
    }
    if invalid:
        raise RuntimeError(
            "one-thread environment mismatch; expected "
            f"{THREAD_ENVIRONMENT!r}, got {observed!r}"
        )
    return {name: str(value) for name, value in observed.items()}


def _fold(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return food101._stratified_folds(labels, n_splits=food101.FOLDS, seed=seed)[0]


def _fit_score(
    method: str,
    raw_values: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    train, holdout = _fold(labels, seed)
    encoded, _classes = food101._stratification_encoding(labels)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    normalization_wall = 0.0
    normalization_cpu = 0.0
    fit_wall = fit_cpu = score_wall = score_cpu = 0.0
    diagnostics: dict[str, Any] = {}

    if method in {"raw_unrefined_oi", "scalable_one_update"}:
        normalize_started_wall = time.perf_counter()
        normalize_started_cpu = time.process_time()
        values = food101._row_l2(raw_values)
        normalization_wall = time.perf_counter() - normalize_started_wall
        normalization_cpu = time.process_time() - normalize_started_cpu
        k_per_class = runtime_benchmark._fold_k_per_class(labels[train])
        if method == "raw_unrefined_oi":
            selector: Any = OverlapIndex(
                prototype_refinement=False,
                **food101._oi_kwargs(k_per_class, seed),
            )
        else:
            selector = ScalableOneUpdateRelevanceKMeans(
                k_per_class=k_per_class,
                kmeans_kwargs={"random_state": int(seed)},
                memory_budget_mb=64,
            )
        fit_started_wall = time.perf_counter()
        fit_started_cpu = time.process_time()
        selector.fit(values[train], labels[train])
        fit_wall = time.perf_counter() - fit_started_wall
        fit_cpu = time.process_time() - fit_started_cpu
        score_started_wall = time.perf_counter()
        score_started_cpu = time.process_time()
        score = float(selector.score_fixed(values[holdout], labels[holdout]))
        score_wall = time.perf_counter() - score_started_wall
        score_cpu = time.process_time() - score_started_cpu
        if method == "scalable_one_update":
            diagnostics = {
                key: value
                for key, value in dict(selector.diagnostics_ or {}).items()
                if not key.endswith("_wall_seconds")
            }
    elif method == "linear_probe":
        pipeline = make_pipeline(
            Normalizer(norm="l2"),
            LogisticRegression(C=1.0, max_iter=2000, random_state=seed, n_jobs=1),
        )
        fit_started_wall = time.perf_counter()
        fit_started_cpu = time.process_time()
        pipeline.fit(raw_values[train], encoded[train])
        fit_wall = time.perf_counter() - fit_started_wall
        fit_cpu = time.process_time() - fit_started_cpu
        score_started_wall = time.perf_counter()
        score_started_cpu = time.process_time()
        score = float(pipeline.score(raw_values[holdout], encoded[holdout]))
        score_wall = time.perf_counter() - score_started_wall
        score_cpu = time.process_time() - score_started_cpu
    else:
        raise ValueError(f"unknown method {method!r}")

    total_wall = time.perf_counter() - started_wall
    total_cpu = time.process_time() - started_cpu
    if not np.isfinite(score):
        raise RuntimeError(f"{method} produced a non-finite score")
    return {
        "method": method,
        "score": score,
        "total_wall_seconds": float(total_wall),
        "total_cpu_seconds": float(total_cpu),
        "normalization_wall_seconds": float(normalization_wall),
        "normalization_cpu_seconds": float(normalization_cpu),
        "fit_wall_seconds": float(fit_wall),
        "fit_cpu_seconds": float(fit_cpu),
        "score_wall_seconds": float(score_wall),
        "score_cpu_seconds": float(score_cpu),
        "train_rows": int(train.size),
        "heldout_rows": int(holdout.size),
        "diagnostics": diagnostics,
    }


def _method_order(budget: int) -> tuple[str, ...]:
    offset = BUDGETS.index(int(budget)) % len(METHODS)
    return METHODS[offset:] + METHODS[:offset]


def run_anchor(
    output: Path,
    *,
    dataset_factory: Callable[[str, str, int], tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Run the immutable two-cell anchor and write a self-contained bundle."""

    protocol_sha = _validate_frozen_protocol()
    thread_environment = _validate_thread_environment()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    factory = dataset_factory or runtime_benchmark.food_dataset_factory(food101.DEFAULT_CACHE)
    rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        raw_values, labels = factory(MODEL, ARM, budget)
        raw_values = np.asarray(raw_values, dtype=np.float32)
        labels = np.asarray(labels)
        if raw_values.ndim != 2 or labels.shape != (raw_values.shape[0],):
            raise ValueError("dataset factory returned malformed values or labels")
        seed = int(food101.SEED + budget)
        for position, method in enumerate(_method_order(budget)):
            warmup = _fit_score(method, raw_values, labels, seed=seed)
            measured = _fit_score(method, raw_values, labels, seed=seed)
            if warmup["score"] != measured["score"]:
                raise RuntimeError(f"{method} was not deterministic at budget {budget}")
            measured.update(
                {
                    "model": MODEL,
                    "arm": ARM,
                    "budget_per_class": int(budget),
                    "seed": seed,
                    "execution_position": int(position),
                    "warmup_excluded": True,
                    "warmup_score_exact": True,
                    "row_count": int(raw_values.shape[0]),
                    "feature_count": int(raw_values.shape[1]),
                }
            )
            rows.append(measured)

    lookup = {
        (int(row["budget_per_class"]), str(row["method"])): row for row in rows
    }
    comparisons: list[dict[str, Any]] = []
    for budget in BUDGETS:
        raw = float(lookup[(budget, "raw_unrefined_oi")]["total_wall_seconds"])
        scalable = float(lookup[(budget, "scalable_one_update")]["total_wall_seconds"])
        probe = float(lookup[(budget, "linear_probe")]["total_wall_seconds"])
        comparisons.append(
            {
                "budget_per_class": int(budget),
                "scalable_over_raw_oi": scalable / raw,
                "scalable_over_linear_probe": scalable / probe,
                "raw_oi_over_linear_probe": raw / probe,
            }
        )
    source_hashes = {str(path): _sha256(path) for path in SOURCE_PATHS}
    payload = {
        "schema_version": 1,
        "study": "scalable_one_update_relevance_kmeans",
        "artifact_status": "completed",
        "interpretation": "development-only real-shape timing and finite-score anchor",
        "protocol_sha256": protocol_sha,
        "source_hashes": source_hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "thread_environment": thread_environment,
        },
        "rows": rows,
        "comparisons": comparisons,
    }
    raw_path = output / "raw_results.json"
    _atomic_json(raw_path, payload)
    manifest_payload = {
        "schema_version": 1,
        "artifact_status": "completed",
        "study": payload["study"],
        "protocol_sha256": protocol_sha,
        "raw_results_sha256": _sha256(raw_path),
        "row_count": len(rows),
        "source_hashes": source_hashes,
    }
    _atomic_json(output / "manifest.json", manifest_payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = run_anchor(args.output)
    print(_canonical_json({"artifact_status": payload["artifact_status"], "comparisons": payload["comparisons"]}))


if __name__ == "__main__":
    main()
