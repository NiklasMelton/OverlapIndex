"""Small exactness and speed probe for the M50 robust diagonal conditioner.

The canonical research adapter deliberately uses a stable full sort for every
feature and materializes dense diagonal matrices.  This module prototypes two
outcome-neutral implementation changes:

1. select the exact same lower median with ``numpy.partition`` when row
   weights are uniform; and
2. retain the diagonal covariance/transform as vectors rather than dense
   ``d x d`` matrices.

The default command uses one real Food-101 runtime fold.  It is opt-in and is
not part of the repository unit suite::

    VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
    PYTHONPATH=. python -m experiments.m50_fast_conditioning.benchmark \
      --model resnet50 --budget 640 --folds 1 --repeats 2

No downstream outcomes are read by the benchmark.  The optional frozen runtime
ledger is used only to project wall-time ratios after replacing each recorded
M1 conditioning component by the observed microbenchmark speedup.  That
projection is descriptive and cannot satisfy the frozen bootstrap or memory
gates by itself.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.m50_backbone_ranking import conditioning_adapter as canonical
from experiments.m50_backbone_ranking import food101, runtime_benchmark


THREAD_ENVIRONMENT = {
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "LOKY_MAX_CPU_COUNT": "1",
}


def _require_thread_environment() -> dict[str, str]:
    observed = {key: os.environ.get(key) for key in THREAD_ENVIRONMENT}
    if observed != THREAD_ENVIRONMENT:
        raise RuntimeError(
            "benchmark requires the six one-thread environment variables; "
            f"expected {THREAD_ENVIRONMENT!r}, got {observed!r}"
        )
    return {key: str(value) for key, value in observed.items()}


def uniform_lower_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the canonical weighted-median value for uniform row weights.

    The frozen implementation selects the lower of the two central values for
    an even row count.  ``k=(n-1)//2`` reproduces that convention without a
    full stable ordering.  Ties need no index-order reconstruction because the
    selected scalar value is identical.
    """

    matrix = np.asarray(values, dtype=np.float64)
    row_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("lower median requires a non-empty two-dimensional matrix")
    if row_weights.size != matrix.shape[0]:
        raise ValueError("lower median inputs have incompatible shapes")
    if not np.isfinite(matrix).all():
        raise ValueError("lower median requires finite values")
    if (
        not np.isfinite(row_weights).all()
        or np.any(row_weights <= 0.0)
        or not np.all(row_weights == row_weights[0])
    ):
        raise ValueError("fast lower median requires uniform positive finite weights")
    kth = int((matrix.shape[0] - 1) // 2)
    partitioned = np.partition(matrix, kth=kth, axis=0)
    return np.asarray(partitioned[kth], dtype=np.float64)


@dataclass(frozen=True)
class FastDiagonalState:
    """Logical M50 state without dense diagonal matrices."""

    mean: np.ndarray
    raw_variance: np.ndarray
    regularized_variance: np.ndarray
    diagonal_scale: np.ndarray
    logical_state_sha256: str

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        return np.asarray((values - self.mean) * self.diagonal_scale, dtype=np.float64)


def _logical_state_hash(
    mean: np.ndarray, raw_variance: np.ndarray, diagonal_scale: np.ndarray
) -> str:
    digest = hashlib.sha256()
    digest.update(b"pooled_mad\0sample_weighted_rows\0gamma=0.5\0")
    for name, array in (
        (b"mean", mean),
        (b"raw_variance", raw_variance),
        (b"diagonal_scale", diagonal_scale),
    ):
        values = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
        digest.update(name + b"\0")
        digest.update(str(values.shape).encode("ascii") + b"\0")
        digest.update(values.tobytes())
    return digest.hexdigest()


def _fit_fast_m50_state_prevalidated(
    values: np.ndarray, target: np.ndarray
) -> FastDiagonalState:
    """Fit vector M50 state from canonical float64/validated inputs."""

    _classes, blocks = canonical._class_blocks(target)
    mean = np.asarray(np.mean(values, axis=0), dtype=np.float64)
    residuals = canonical._residual_stack(values, blocks)
    weights, uses_sample_path = canonical._weights_for_blocks(
        blocks, "sample_weighted_rows"
    )
    if not uses_sample_path:
        raise AssertionError("sample-weighted M50 must use the uniform-weight path")
    center = uniform_lower_median(residuals, weights)
    mad = canonical.MAD_CONSISTENCY * uniform_lower_median(
        np.abs(residuals - center), weights
    )
    raw_variance = np.asarray(mad * mad, dtype=np.float64)
    regularized, _effective_floor = canonical._regularize_variances(raw_variance)
    diagonal_scale = np.asarray(regularized ** (-0.25), dtype=np.float64)
    for array in (mean, raw_variance, regularized, diagonal_scale):
        array.setflags(write=False)
    return FastDiagonalState(
        mean=mean,
        raw_variance=raw_variance,
        regularized_variance=regularized,
        diagonal_scale=diagonal_scale,
        logical_state_sha256=_logical_state_hash(mean, raw_variance, diagonal_scale),
    )


def fit_fast_m50_state(X: np.ndarray, labels: np.ndarray) -> FastDiagonalState:
    """Fit the exact M50-SW numerical state using partition and vectors."""

    values = canonical._validate_dense_matrix(X, "X", preserve_dtype=False)
    target = canonical._validate_labels(labels, values.shape[0])
    return _fit_fast_m50_state_prevalidated(values, target)


def _canonical_snapshot(conditioner: Any, probe: np.ndarray) -> dict[str, Any]:
    return {
        "mean": np.asarray(conditioner.mean_, dtype=np.float64).copy(),
        "raw_variance": np.asarray(np.diag(conditioner.covariance_), dtype=np.float64),
        "diagonal_scale": np.asarray(np.diag(conditioner.transform_), dtype=np.float64),
        "transform": np.asarray(conditioner.transform(probe), dtype=np.float64),
        "canonical_state_sha256": str(conditioner.state_sha256_),
    }


def _fast_snapshot(state: FastDiagonalState, probe: np.ndarray) -> dict[str, Any]:
    return {
        "mean": np.asarray(state.mean, dtype=np.float64),
        "raw_variance": np.asarray(state.raw_variance, dtype=np.float64),
        "diagonal_scale": np.asarray(state.diagonal_scale, dtype=np.float64),
        "transform": np.asarray(state.transform(probe), dtype=np.float64),
        "logical_state_sha256": state.logical_state_sha256,
    }


def _exact_numerical_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        np.array_equal(np.asarray(left[key]), np.asarray(right[key]))
        for key in ("mean", "raw_variance", "diagonal_scale", "transform")
    )


def _fit_canonical(X: np.ndarray, labels: np.ndarray) -> tuple[Any, float]:
    started = time.perf_counter()
    state = canonical._fit_conditioner(
        X,
        labels,
        estimator="pooled_mad",
        weighting="sample_weighted_rows",
        gamma=0.5,
    )
    return state, float(time.perf_counter() - started)


def _fit_partition_dense(X: np.ndarray, labels: np.ndarray) -> tuple[Any, float]:
    original = canonical._weighted_median
    canonical._weighted_median = uniform_lower_median
    try:
        return _fit_canonical(X, labels)
    finally:
        canonical._weighted_median = original


def _fit_partition_vector(
    X: np.ndarray, labels: np.ndarray
) -> tuple[FastDiagonalState, float]:
    started = time.perf_counter()
    state = _fit_fast_m50_state_prevalidated(X, labels)
    return state, float(time.perf_counter() - started)


def benchmark_fold(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    repeats: int,
    warmup: bool = True,
) -> dict[str, Any]:
    """Benchmark one fold and require exact numerical state equivalence."""

    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    values = canonical._validate_dense_matrix(X, "X", preserve_dtype=False)
    target = canonical._validate_labels(labels, values.shape[0])
    probe = np.asarray(values[: min(16, values.shape[0])], dtype=np.float64)
    methods = {
        "canonical_sort_dense": _fit_canonical,
        "partition_dense": _fit_partition_dense,
        "partition_vector": _fit_partition_vector,
    }
    if warmup:
        for fit in methods.values():
            state, _elapsed = fit(values, target)
            del state
            gc.collect()

    timings: dict[str, list[float]] = {name: [] for name in methods}
    reference: dict[str, Any] | None = None
    parity = {"partition_dense": True, "partition_vector": True}
    names = tuple(methods)
    for repeat in range(repeats):
        order = names if repeat % 2 == 0 else tuple(reversed(names))
        snapshots: dict[str, dict[str, Any]] = {}
        for name in order:
            state, elapsed = methods[name](values, target)
            timings[name].append(elapsed)
            if name == "partition_vector":
                snapshots[name] = _fast_snapshot(state, probe)
            else:
                snapshots[name] = _canonical_snapshot(state, probe)
            del state
            gc.collect()
        reference = snapshots["canonical_sort_dense"]
        for name in parity:
            parity[name] = parity[name] and _exact_numerical_match(
                reference, snapshots[name]
            )
        if snapshots["partition_dense"].get("canonical_state_sha256") != reference.get(
            "canonical_state_sha256"
        ):
            parity["partition_dense"] = False

    if reference is None or not all(parity.values()):
        raise AssertionError(f"fast M50 numerical parity failed: {parity!r}")
    medians = {name: float(statistics.median(values)) for name, values in timings.items()}
    baseline = medians["canonical_sort_dense"]
    return {
        "row_count": int(values.shape[0]),
        "feature_count": int(values.shape[1]),
        "repeats": repeats,
        "warmup_excluded": bool(warmup),
        "exact_numerical_parity": parity,
        "median_wall_seconds": medians,
        "speedup_vs_canonical": {
            name: float(baseline / value) for name, value in medians.items() if name != "canonical_sort_dense"
        },
        "canonical_state_sha256": reference["canonical_state_sha256"],
    }


def _load_runtime_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 4_500:
        raise ValueError("runtime projection requires the complete frozen 4,500-row ledger")
    return rows


def project_frozen_runtime(
    runtime_rows: Sequence[Mapping[str, Any]], *, conditioning_speedup: float
) -> dict[str, Any]:
    """Project M1 wall ratios after only its conditioning component changes."""

    if not np.isfinite(conditioning_speedup) or conditioning_speedup <= 0.0:
        raise ValueError("conditioning_speedup must be positive and finite")
    lookup: dict[tuple[str, str, int, int, str], Mapping[str, Any]] = {}
    for row in runtime_rows:
        key = (
            str(row["backbone"]),
            str(row["arm"]),
            int(row["budget"]),
            int(row["repeat"]),
            str(row["candidate_id"]),
        )
        if key in lookup:
            raise ValueError(f"duplicate runtime row {key!r}")
        lookup[key] = row

    def summarize(speedup: float | None) -> tuple[dict[str, float], dict[str, float]]:
        per_call: dict[str, list[float]] = {
            arm: [] for arm in runtime_benchmark.manifest.RUNTIME_ARMS
        }
        panel_sums: dict[tuple[str, int, int], list[float]] = {}
        for arm in per_call:
            for model in food101.MODELS:
                for budget in runtime_benchmark.RUNTIME_BUDGETS:
                    for repeat in runtime_benchmark.RUNTIME_REPEATS:
                        prefix = (model, arm, int(budget), int(repeat))
                        m1 = lookup[prefix + ("M1-SW",)]
                        probe = lookup[prefix + ("LP-FULL",)]
                        current_total = float(m1["total_wall_seconds"])
                        current_conditioning = float(m1["conditioning_fit_wall_seconds"])
                        projected = current_total - current_conditioning
                        if speedup is not None:
                            projected += current_conditioning / speedup
                        probe_total = float(probe["total_wall_seconds"])
                        per_call[arm].append(projected / probe_total)
                        sums = panel_sums.setdefault(
                            (arm, int(budget), int(repeat)), [0.0, 0.0]
                        )
                        sums[0] += projected
                        sums[1] += probe_total

        full_panel: dict[str, list[float]] = {arm: [] for arm in per_call}
        for (arm, _budget, _repeat), (candidate_sum, probe_sum) in panel_sums.items():
            full_panel[arm].append(candidate_sum / probe_sum)
        return (
            {
                arm: float(statistics.median(values))
                for arm, values in per_call.items()
            },
            {
                arm: float(statistics.median(values))
                for arm, values in full_panel.items()
            },
        )

    per_call_median, full_panel_median = summarize(float(conditioning_speedup))
    zero_per_call, zero_full_panel = summarize(None)
    zero_target_met = bool(
        all(value <= 1.0 for value in zero_per_call.values())
        and all(value <= 1.0 for value in zero_full_panel.values())
    )
    return {
        "conditioning_speedup": float(conditioning_speedup),
        "per_call_median_ratio_vs_lp_full_by_arm": per_call_median,
        "full_panel_median_ratio_vs_lp_full_by_arm": full_panel_median,
        "point_target": "all per-call and full-panel medians <= 1.0",
        "point_target_met": bool(
            all(value <= 1.0 for value in per_call_median.values())
            and all(value <= 1.0 for value in full_panel_median.values())
        ),
        "zero_conditioning_time_lower_bound": {
            "per_call_median_ratio_vs_lp_full_by_arm": zero_per_call,
            "full_panel_median_ratio_vs_lp_full_by_arm": zero_full_panel,
            "point_target_met": zero_target_met,
        },
        "conditioning_optimization_alone_can_meet_point_target": zero_target_met,
        "frozen_gate_status": "not_evaluated_by_microbenchmark",
        "caveat": "No paired bootstrap upper bound or peak-memory gate is recomputed.",
    }


def run_benchmark(
    *,
    model: str,
    arm: str,
    budget: int,
    folds: int,
    repeats: int,
    cache_dir: Path,
    runtime_rows_path: Path | None,
) -> dict[str, Any]:
    thread_environment = _require_thread_environment()
    if model not in food101.MODELS:
        raise ValueError(f"unknown model {model!r}")
    if arm not in runtime_benchmark.manifest.RUNTIME_ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if budget not in runtime_benchmark.RUNTIME_BUDGETS:
        raise ValueError(f"unknown budget {budget!r}")
    if type(folds) is not int or not 1 <= folds <= food101.FOLDS:
        raise ValueError(f"folds must be in [1, {food101.FOLDS}]")

    factory = runtime_benchmark.food_dataset_factory(cache_dir)
    raw, labels = factory(model, arm, int(budget))
    values = food101._row_l2(raw)
    split_seed = int(food101.SEED + int(budget))
    splits = food101._stratified_folds(labels, seed=split_seed)
    fold_results = []
    for fold, (train, _holdout) in enumerate(splits[:folds]):
        result = benchmark_fold(values[train], labels[train], repeats=repeats)
        result["fold"] = int(fold)
        fold_results.append(result)
    speedups = [
        float(row["speedup_vs_canonical"]["partition_vector"]) for row in fold_results
    ]
    aggregate_speedup = float(statistics.median(speedups))
    output: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "m50_exact_partition_vector_probe",
        "outcomes_read": False,
        "model": model,
        "arm": arm,
        "budget_per_class": int(budget),
        "fold_count": int(folds),
        "repeat_count": int(repeats),
        "split_seed": split_seed,
        "thread_environment": thread_environment,
        "fold_results": fold_results,
        "aggregate_partition_vector_speedup": aggregate_speedup,
    }
    if runtime_rows_path is not None:
        output["runtime_projection"] = project_frozen_runtime(
            _load_runtime_rows(runtime_rows_path),
            conditioning_speedup=aggregate_speedup,
        )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=food101.MODELS, default="resnet50")
    parser.add_argument(
        "--arm",
        choices=runtime_benchmark.manifest.RUNTIME_ARMS,
        default="baseline",
    )
    parser.add_argument(
        "--budget",
        type=int,
        choices=runtime_benchmark.RUNTIME_BUDGETS,
        default=640,
    )
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path, default=food101.DEFAULT_CACHE)
    parser.add_argument(
        "--runtime-rows",
        type=Path,
        default=Path("artifacts/m50_backbone_ranking/runtime/runtime_rows.jsonl"),
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = run_benchmark(
        model=args.model,
        arm=args.arm,
        budget=int(args.budget),
        folds=int(args.folds),
        repeats=int(args.repeats),
        cache_dir=args.cache_dir,
        runtime_rows_path=args.runtime_rows,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(f"refuse to overwrite benchmark output {args.output}")
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FastDiagonalState",
    "benchmark_fold",
    "fit_fast_m50_state",
    "main",
    "project_frozen_runtime",
    "run_benchmark",
    "uniform_lower_median",
]
