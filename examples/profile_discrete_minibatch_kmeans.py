"""Profile the discrete MiniBatchKMeans OverlapIndex execution path.

The benchmark sweeps sample count and feature dimensionality and records three
non-overlapping phases of one ``OverlapIndex.fit_offline`` call:

``clustering``
    The per-class MiniBatchKMeans fits and center bookkeeping performed by the
    clustering adapter.
``overlap_scoring``
    The backend-neutral universal offline scorer, including bounded score
    tiles and top-two prototype selection.
``other``
    The residual of the end-to-end ``fit_offline`` time after the two timed
    phases.  This includes validation, preprocessing, label bookkeeping, and
    final aggregation.

Raw repetition-level rows and a median/mean summary are written as CSV, with a
JSON metadata file describing the run.  The plotting companion
``plot_discrete_minibatch_kmeans_profile.py`` consumes the summary CSV.

The defaults are intentionally bounded (at most 5,000 samples and 500
features), but all grids and run parameters can be overridden from the CLI.
One small, untimed representative fit is run first by default to warm BLAS and
scikit-learn initialization; pass ``--no-warmup`` to disable it.  This module
does not execute the sweep on import.  Offline score tiles use the
``offline_memory_budget_mb=256`` scratch-memory budget by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import sklearn

from overlapindex import OverlapIndex


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "profiling" / "discrete_minibatch_kmeans_universal_scorer"
DEFAULT_SAMPLES = (250, 500, 1_000, 2_000, 3_500, 5_000)
DEFAULT_DIMENSIONS = (10, 25, 50, 100, 250, 500)
MAX_SAMPLES = 5_000
MAX_DIMENSIONS = 500
DEFAULT_OFFLINE_MEMORY_BUDGET_MB = 256

# These values mirror the adapter's effective MiniBatchKMeans defaults.  Keep
# them explicit in the profile so a run remains tied to the documented
# backend configuration even when scikit-learn changes its own defaults.
DEFAULT_MINIBATCH_KMEANS_KWARGS = {
    "batch_size": 256,
    "max_no_improvement": 5,
    "compute_labels": False,
    "n_init": 1,
    "init": "random",
}


@dataclass
class _PhaseTimer:
    """Mutable phase totals used by one instrumented fit."""

    clustering_seconds: float = 0.0
    overlap_scoring_seconds: float = 0.0


class _TimedMiniBatchOverlapIndex(OverlapIndex):
    """OverlapIndex subclass that times the two expensive internal phases.

    The backend is wrapped after each reset so that the timer follows the
    actual model used by ``fit_offline``.  The scorer hook covers the
    backend-neutral universal offline path used by this benchmark's generated
    single-label data.  The public package implementation remains untouched by
    profiling.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._profile_timer: _PhaseTimer | None = None
        super().__init__(*args, **kwargs)

    def _build_model(self):  # type: ignore[no-untyped-def]
        model = super()._build_model()
        timer = self._profile_timer
        if timer is None or self.model_type != "MiniBatchKMeans":
            return model

        original_fit = model.fit_offline

        def timed_fit(X: np.ndarray, Y: np.ndarray, *args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original_fit(X, Y, *args, **kwargs)
            finally:
                timer.clustering_seconds += time.perf_counter() - started

        # Assigning the closure to this instance deliberately avoids changing
        # the production adapter or affecting other estimators in the process.
        model.fit_offline = timed_fit  # type: ignore[method-assign]
        return model

    def _fit_offline_centroid_optimized(self, X_prep, Y, classes):  # type: ignore[no-untyped-def]
        timer = self._profile_timer
        started = time.perf_counter()
        try:
            return super()._fit_offline_centroid_optimized(X_prep, Y, classes)
        finally:
            if timer is not None:
                timer.overlap_scoring_seconds += time.perf_counter() - started


def _validate_grid(values: Iterable[int], name: str, cap: int) -> tuple[int, ...]:
    """Validate, deduplicate, and sort one sweep axis."""

    result = []
    for value in values:
        integer = int(value)
        if integer <= 0:
            raise ValueError(f"{name} values must be positive; got {value!r}.")
        if integer > cap:
            raise ValueError(f"{name} values cannot exceed {cap}; got {integer}.")
        if integer not in result:
            result.append(integer)
    if not result:
        raise ValueError(f"{name} must contain at least one value.")
    return tuple(sorted(result))


def _effective_kmeans_kwargs(
    seed: int,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one cell's effective MiniBatchKMeans kwargs.

    The adapter supplies these defaults internally, but spelling them out here
    keeps profile runs reproducible and makes any caller overrides explicit.
    """

    kwargs = dict(DEFAULT_MINIBATCH_KMEANS_KWARGS)
    kwargs["random_state"] = int(seed)
    if overrides:
        kwargs.update(overrides)
    return kwargs


def make_dataset(
    n_samples: int,
    n_dimensions: int,
    *,
    n_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic, dense, single-label benchmark dataset.

    Classes have equal support (up to one sample) and modestly separated means
    in the first feature.  The geometry is intentionally simple: this
    benchmark measures runtime, not clustering quality.
    """

    if n_classes < 2:
        raise ValueError("n_classes must be at least 2 for pairwise overlap scoring.")
    if n_samples < n_classes:
        raise ValueError("n_samples must be at least n_classes.")
    if n_dimensions < 1:
        raise ValueError("n_dimensions must be positive.")

    rng = np.random.default_rng(seed)
    labels = np.arange(n_samples, dtype=int) % n_classes
    rng.shuffle(labels)
    X = rng.normal(0.0, 1.0, size=(n_samples, n_dimensions)).astype(np.float32)
    # Keep the class signal bounded in high dimensions while avoiding a
    # perfectly degenerate all-zero first coordinate.
    X[:, 0] += (labels.astype(np.float32) - (n_classes - 1) / 2.0) * 1.5
    return X, labels


def profile_one(
    n_samples: int,
    n_dimensions: int,
    *,
    repetition: int,
    seed: int,
    n_classes: int = 5,
    k: int = 8,
    chunk_size: int | None = 10_000,
    offline_memory_budget_mb: int = DEFAULT_OFFLINE_MEMORY_BUDGET_MB,
    kmeans_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one timed fit and return a tidy raw-result row.

    ``other_seconds`` is calculated as the residual of the measured
    end-to-end call, making the three component columns exhaustive by
    construction (up to normal clock noise).
    """

    X, y = make_dataset(
        n_samples,
        n_dimensions,
        n_classes=n_classes,
        seed=seed,
    )
    kwargs = _effective_kmeans_kwargs(seed, kmeans_kwargs)

    estimator = _TimedMiniBatchOverlapIndex(
        model_type="MiniBatchKMeans",
        kmeans_k=k,
        kmeans_kwargs=kwargs,
        offline_chunk_size=chunk_size,
        offline_memory_budget_mb=offline_memory_budget_mb,
    )
    timer = _PhaseTimer()
    estimator._profile_timer = timer

    started = time.perf_counter()
    try:
        score = estimator.fit_offline(X, y, reset_state=True)
    finally:
        total_seconds = time.perf_counter() - started

    other_seconds = total_seconds - timer.clustering_seconds - timer.overlap_scoring_seconds
    return {
        "n_samples": int(n_samples),
        "n_dimensions": int(n_dimensions),
        "repetition": int(repetition),
        "seed": int(seed),
        "n_classes": int(n_classes),
        "k": int(k),
        "offline_chunk_size": "None" if chunk_size is None else int(chunk_size),
        "offline_memory_budget_mb": int(offline_memory_budget_mb),
        "total_seconds": float(total_seconds),
        "clustering_seconds": float(timer.clustering_seconds),
        "overlap_scoring_seconds": float(timer.overlap_scoring_seconds),
        "other_seconds": float(other_seconds),
        "index": float(score),
    }


def warmup_fit(
    *,
    n_samples: int,
    n_dimensions: int,
    seed: int,
    n_classes: int,
    k: int,
    chunk_size: int | None,
    offline_memory_budget_mb: int = DEFAULT_OFFLINE_MEMORY_BUDGET_MB,
    kmeans_kwargs: dict[str, Any] | None = None,
) -> None:
    """Run one small untimed fit to remove first-cell initialization noise."""

    X, y = make_dataset(
        n_samples,
        n_dimensions,
        n_classes=n_classes,
        seed=seed,
    )
    kwargs = _effective_kmeans_kwargs(seed, kmeans_kwargs)
    OverlapIndex(
        model_type="MiniBatchKMeans",
        kmeans_k=k,
        kmeans_kwargs=kwargs,
        offline_chunk_size=chunk_size,
        offline_memory_budget_mb=offline_memory_budget_mb,
    ).fit_offline(X, y, reset_state=True)


def run_sweep(
    samples: Sequence[int] = DEFAULT_SAMPLES,
    dimensions: Sequence[int] = DEFAULT_DIMENSIONS,
    *,
    repetitions: int = 3,
    seed: int = 20260807,
    n_classes: int = 5,
    k: int = 8,
    chunk_size: int | None = 10_000,
    offline_memory_budget_mb: int = DEFAULT_OFFLINE_MEMORY_BUDGET_MB,
    kmeans_kwargs: dict[str, Any] | None = None,
    warmup: bool = True,
    warmup_samples: int = 250,
    warmup_dimensions: int = 10,
) -> list[dict[str, Any]]:
    """Run the requested sample/dimension sweep and return raw rows."""

    samples = _validate_grid(samples, "samples", MAX_SAMPLES)
    dimensions = _validate_grid(dimensions, "dimensions", MAX_DIMENSIONS)
    repetitions = int(repetitions)
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be positive or None.")
    warmup_samples = int(warmup_samples)
    warmup_dimensions = int(warmup_dimensions)
    if warmup_samples <= 0 or warmup_samples > MAX_SAMPLES:
        raise ValueError(f"warmup_samples must be in [1, {MAX_SAMPLES}].")
    if warmup_dimensions <= 0 or warmup_dimensions > MAX_DIMENSIONS:
        raise ValueError(f"warmup_dimensions must be in [1, {MAX_DIMENSIONS}].")
    if warmup_samples < n_classes:
        raise ValueError("warmup_samples must be at least n_classes.")

    rows: list[dict[str, Any]] = []
    if warmup:
        warmup_fit(
            n_samples=warmup_samples,
            n_dimensions=warmup_dimensions,
            seed=int(seed - 1),
            n_classes=n_classes,
            k=k,
            chunk_size=chunk_size,
            offline_memory_budget_mb=offline_memory_budget_mb,
            kmeans_kwargs=kmeans_kwargs,
        )
    cell_number = 0
    for n_samples in samples:
        for n_dimensions in dimensions:
            for repetition in range(repetitions):
                cell_seed = int(seed + cell_number * repetitions + repetition)
                row = profile_one(
                    n_samples,
                    n_dimensions,
                    repetition=repetition,
                    seed=cell_seed,
                    n_classes=n_classes,
                    k=k,
                    chunk_size=chunk_size,
                    offline_memory_budget_mb=offline_memory_budget_mb,
                    kmeans_kwargs=kmeans_kwargs,
                )
                rows.append(row)
                print(
                    f"n={n_samples:5d} d={n_dimensions:4d} "
                    f"rep={repetition + 1}/{repetitions} "
                    f"total={row['total_seconds']:.4f}s "
                    f"cluster={row['clustering_seconds']:.4f}s "
                    f"score={row['overlap_scoring_seconds']:.4f}s "
                    f"other={row['other_seconds']:.4f}s",
                    flush=True,
                )
            cell_number += 1
    return rows


TIME_COLUMNS = (
    "total_seconds",
    "clustering_seconds",
    "overlap_scoring_seconds",
    "other_seconds",
)
RAW_FIELDS = (
    "n_samples",
    "n_dimensions",
    "repetition",
    "seed",
    "n_classes",
    "k",
    "offline_chunk_size",
    "offline_memory_budget_mb",
    *TIME_COLUMNS,
    "index",
)


def summarize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate raw rows by grid cell using median and common spread stats."""

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["n_samples"]), int(row["n_dimensions"]))
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for (n_samples, n_dimensions), group in sorted(grouped.items()):
        result: dict[str, Any] = {
            "n_samples": n_samples,
            "n_dimensions": n_dimensions,
            "n_repetitions": len(group),
        }
        for column in (*TIME_COLUMNS, "index"):
            values = np.asarray([float(row[column]) for row in group], dtype=float)
            result[f"median_{column}"] = float(np.median(values))
            result[f"mean_{column}"] = float(np.mean(values))
            result[f"std_{column}"] = float(np.std(values, ddof=0))
            result[f"min_{column}"] = float(np.min(values))
            result[f"max_{column}"] = float(np.max(values))
        summary.append(result)
    return summary


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def save_results(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    config: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Save raw rows, summaries, and run metadata; return their paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "profile_raw.csv"
    summary_path = output_dir / "profile_summary.csv"
    metadata_path = output_dir / "profile_metadata.json"
    _write_csv(raw_path, rows, RAW_FIELDS)

    summary_rows = summarize_rows(rows)
    summary_fields = ["n_samples", "n_dimensions", "n_repetitions"]
    for column in (*TIME_COLUMNS, "index"):
        summary_fields.extend(
            f"{stat}_{column}" for stat in ("median", "mean", "std", "min", "max")
        )
    _write_csv(summary_path, summary_rows, summary_fields)

    metadata = {
        "schema_version": 1,
        "benchmark": "discrete_overlapindex_minibatch_kmeans",
        "components": {
            "clustering_seconds": "MiniBatchKMeans backend fit and center assembly",
            "overlap_scoring_seconds": (
                "backend-neutral universal offline scorer with bounded row/prototype tiles"
            ),
            "other_seconds": "fit_offline total minus the two timed components",
        },
        "dataset": {
            "representation": "dense numpy.ndarray",
            "dtype": "float32",
            "generator": "numpy.random.default_rng(seed)",
            "features": "Gaussian N(0, 1), with a class-dependent shift in feature 0",
            "labels": "balanced single-label classes; labels shuffled with the same cell seed",
            "purpose": "runtime measurement only; geometry is not intended as a quality benchmark",
        },
        "effective_minibatch_kmeans": {
            **DEFAULT_MINIBATCH_KMEANS_KWARGS,
            "n_clusters_per_class": config.get("k"),
            "random_state": "cell seed (metadata seed plus cell/repetition offset)",
        },
        "offline_memory_budget_mb": config.get(
            "offline_memory_budget_mb", DEFAULT_OFFLINE_MEMORY_BUDGET_MB
        ),
        "warmup": {
            "enabled": bool(config.get("warmup", True)),
            "timed": False,
            "n_samples": config.get("warmup_samples"),
            "n_dimensions": config.get("warmup_dimensions"),
            "seed": config.get("warmup_seed"),
        },
        "reproducibility": _reproducibility_metadata(),
        **config,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return raw_path, summary_path, metadata_path


def _reproducibility_metadata() -> dict[str, Any]:
    """Return environment details that affect timing reproducibility."""

    thread_keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_DYNAMIC",
        "MKL_DYNAMIC",
        "LOKY_MAX_CPU_COUNT",
    )
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "python_executable": sys.executable,
        "thread_environment": {key: os.environ.get(key) for key in thread_keys},
    }


def refresh_metadata(metadata_path: Path) -> Path:
    """Enrich an existing metadata JSON without rerunning any benchmark cell."""

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = {
        key: metadata[key]
        for key in ("k", "offline_memory_budget_mb")
        if key in metadata
    }
    metadata.setdefault("dataset", {
        "representation": "dense numpy.ndarray",
        "dtype": "float32",
        "generator": "numpy.random.default_rng(seed)",
        "features": "Gaussian N(0, 1), with a class-dependent shift in feature 0",
        "labels": "balanced single-label classes; labels shuffled with the same cell seed",
        "purpose": "runtime measurement only; geometry is not intended as a quality benchmark",
    })
    effective_kmeans = dict(metadata.get("effective_minibatch_kmeans", {}))
    effective_kmeans.update(DEFAULT_MINIBATCH_KMEANS_KWARGS)
    effective_kmeans.update(
        {
            "n_clusters_per_class": config.get("k", effective_kmeans.get("n_clusters_per_class")),
            "random_state": "cell seed (metadata seed plus cell/repetition offset)",
        }
    )
    metadata["effective_minibatch_kmeans"] = effective_kmeans
    metadata["components"] = {
        **metadata.get("components", {}),
        "clustering_seconds": "MiniBatchKMeans backend fit and center assembly",
        "overlap_scoring_seconds": (
            "backend-neutral universal offline scorer with bounded row/prototype tiles"
        ),
        "other_seconds": "fit_offline total minus the two timed components",
    }
    metadata["offline_memory_budget_mb"] = metadata.get(
        "offline_memory_budget_mb",
        DEFAULT_OFFLINE_MEMORY_BUDGET_MB,
    )
    metadata.setdefault("warmup", {
        "enabled": False,
        "timed": False,
        "n_samples": None,
        "n_dimensions": None,
        "seed": None,
    })
    metadata["reproducibility"] = _reproducibility_metadata()
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, nargs="+", default=DEFAULT_SAMPLES)
    parser.add_argument("--dimensions", type=int, nargs="+", default=DEFAULT_DIMENSIONS)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--classes", type=int, default=5, dest="n_classes")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument(
        "--offline-memory-budget-mb",
        type=int,
        default=DEFAULT_OFFLINE_MEMORY_BUDGET_MB,
        help=(
            "Scratch-memory budget for backend-neutral offline score tiles "
            "(default: 256 MiB)."
        ),
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one untimed representative fit before the sweep (default: enabled).",
    )
    parser.add_argument("--warmup-samples", type=int, default=250)
    parser.add_argument("--warmup-dimensions", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = run_sweep(
        args.samples,
        args.dimensions,
        repetitions=args.repetitions,
        seed=args.seed,
        n_classes=args.n_classes,
        k=args.k,
        chunk_size=args.chunk_size,
        offline_memory_budget_mb=args.offline_memory_budget_mb,
        warmup=args.warmup,
        warmup_samples=args.warmup_samples,
        warmup_dimensions=args.warmup_dimensions,
    )
    # ``run_sweep`` owns grid validation and normalization.  Derive the
    # effective axes from its rows instead of validating the same inputs twice.
    samples = sorted({int(row["n_samples"]) for row in rows})
    dimensions = sorted({int(row["n_dimensions"]) for row in rows})
    config = {
        "samples": samples,
        "dimensions": dimensions,
        "repetitions": int(args.repetitions),
        "seed": int(args.seed),
        "n_classes": int(args.n_classes),
        "k": int(args.k),
        "offline_chunk_size": int(args.chunk_size),
        "offline_memory_budget_mb": int(args.offline_memory_budget_mb),
        "warmup": bool(args.warmup),
        "warmup_samples": int(args.warmup_samples),
        "warmup_dimensions": int(args.warmup_dimensions),
        "warmup_seed": int(args.seed - 1),
    }
    paths = save_results(rows, args.output_dir.resolve(), config=config)
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
