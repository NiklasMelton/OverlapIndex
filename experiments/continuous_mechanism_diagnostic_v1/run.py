"""Frozen synthetic mechanism diagnostic for continuous OI.

This slice keeps the representation rows and targets fixed across selector
seeds, then changes only the training sample for a separate perturbation.  The
predictive reference is analytic: it is the R2 of the known conditional mean
on an independent evaluation draw.  No selector score defines truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import KFold

from experiments.continuous_rff_mean_ranking import run as rff_mean
from experiments.continuous_kernel_alignment import run as kernel
from experiments.continuous_null_diagnostic import run as null_parent
from experiments.continuous_save_ranking import run as save


PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
RUNNER_PATH = Path(__file__).resolve()
ROOT = PACKAGE_DIR.parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "continuous_mechanism_diagnostic_v1" / "results"
METHOD_IDS = ("observed_oi", "broad_rff", "save_oi")
FAMILY_IDS = ("linear_mean", "quadratic_mean", "periodic_mean", "variance_only", "null_noise")
PREDICTIVE_FAMILIES = ("linear_mean", "quadratic_mean", "periodic_mean")
CONTROL_FAMILIES = ("variance_only", "null_noise")
TIE_ATOL = 1.0e-12


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        + b"\n"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_bytes(value))
    os.replace(temporary, path)


def load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_exploratory_before_outcomes":
        raise ValueError("mechanism diagnostic protocol is not frozen")
    methods = tuple(row["method_id"] for row in protocol["selector"]["methods"])
    if methods != METHOD_IDS:
        raise ValueError("mechanism diagnostic method grid changed")
    if tuple(protocol["data"]["families"]) != FAMILY_IDS:
        raise ValueError("mechanism diagnostic family grid changed")
    if tuple(protocol["selector"]["algorithm_seeds"]) != (101, 202, 303):
        raise ValueError("mechanism diagnostic algorithm seeds changed")
    if tuple(protocol["data"]["training_sample_seeds"]) != (1001, 1002, 1003):
        raise ValueError("mechanism diagnostic sample seeds changed")
    if int(protocol["selector"]["folds"]) != 3 or int(protocol["data"]["train_rows"]) != 512:
        raise ValueError("mechanism diagnostic data grid changed")
    return protocol


def _stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return 1 + int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 4_000_000_000


def _family_position(family: str) -> int:
    try:
        return FAMILY_IDS.index(family)
    except ValueError as exc:
        raise ValueError(f"unknown mechanism family: {family!r}") from exc


def _quality_tau(quality: float) -> float:
    q = float(quality)
    if not 0.0 < q <= 1.0:
        raise ValueError("candidate quality must lie in (0, 1]")
    return float(np.sqrt((1.0 / (q * q)) - 1.0))


def _latent_rows(family: str, seed: int, n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_stable_seed("latent", family, seed))
    u = rng.normal(size=n_rows)
    epsilon = rng.normal(size=n_rows)
    independent_signal = rng.normal(size=n_rows)
    representation_noise = rng.normal(size=(n_rows, 8))
    return u, epsilon, independent_signal, representation_noise


def _target(family: str, u: np.ndarray, epsilon: np.ndarray, noise_sd: float) -> np.ndarray:
    if family == "linear_mean":
        return np.asarray(u + noise_sd * epsilon, dtype=np.float64)
    if family == "quadratic_mean":
        return np.asarray(np.square(u) - 1.0 + noise_sd * epsilon, dtype=np.float64)
    if family == "periodic_mean":
        return np.asarray(np.sin(u) + noise_sd * epsilon, dtype=np.float64)
    if family == "variance_only":
        return np.asarray((0.4 + 0.8 * np.abs(u)) * epsilon, dtype=np.float64)
    if family == "null_noise":
        return np.asarray(noise_sd * epsilon, dtype=np.float64)
    raise ValueError(f"unknown mechanism family: {family!r}")


def _representation(
    family: str,
    quality: float,
    u: np.ndarray,
    independent_signal: np.ndarray,
    representation_noise: np.ndarray,
) -> np.ndarray:
    tau = _quality_tau(quality)
    if family == "null_noise":
        source = independent_signal
    else:
        source = u
    # The first coordinate is a noisy measurement; the remaining coordinates
    # are irrelevant noise.  All candidate arrays have the same row order.
    z = source + tau * representation_noise[:, 0]
    values = np.column_stack([z, representation_noise[:, 1:]])
    if values.shape[1] != 8 or not np.isfinite(values).all():
        raise RuntimeError("synthetic representation is invalid")
    return np.asarray(values, dtype=np.float64)


def _conditional_mean(family: str, quality: float, z: np.ndarray) -> np.ndarray:
    q = float(quality)
    if family == "linear_mean":
        return q * q * z
    if family == "quadratic_mean":
        return q**4 * np.square(z) - q * q
    if family == "periodic_mean":
        return np.exp(-(1.0 - q * q) / 2.0) * np.sin(q * q * z)
    if family in CONTROL_FAMILIES:
        return np.zeros_like(z, dtype=np.float64)
    raise ValueError(f"unknown mechanism family: {family!r}")


def _truth_row(
    family: str,
    quality: float,
    X_evaluation: np.ndarray,
    y_evaluation: np.ndarray,
) -> dict[str, float]:
    conditional = _conditional_mean(family, quality, X_evaluation[:, 0])
    variance = float(np.var(y_evaluation))
    if variance <= 0.0:
        raise RuntimeError("synthetic evaluation target has zero variance")
    empirical_r2 = float(1.0 - np.mean(np.square(y_evaluation - conditional)) / variance)
    q = float(quality)
    noise_var = 0.35**2
    if family == "linear_mean":
        expected_r2 = q * q / (1.0 + noise_var)
    elif family == "quadratic_mean":
        expected_r2 = 2.0 * q**4 / (2.0 + noise_var)
    elif family == "periodic_mean":
        signal_var = 0.5 * (1.0 - np.exp(-2.0))
        explained = 0.5 * np.exp(-(1.0 - q * q)) * (1.0 - np.exp(-2.0 * q * q))
        expected_r2 = explained / (signal_var + noise_var)
    else:
        expected_r2 = 0.0
    return {
        "analytic_expected_r2": float(expected_r2),
        "evaluation_conditional_mean_r2": empirical_r2,
        "conditional_mean_mse": float(np.mean(np.square(y_evaluation - conditional))),
    }


def generate_panel(
    protocol: Mapping[str, Any], family: str, training_seed: int,
) -> dict[str, Any]:
    data = protocol["data"]
    train_rows = int(data["train_rows"])
    evaluation_rows = int(data["evaluation_rows"])
    noise_sd = float(data["target_noise_standard_deviation"])
    eval_u, eval_epsilon, eval_independent, eval_noise = _latent_rows(
        family, int(data["evaluation_seed"]), evaluation_rows
    )
    train_u, train_epsilon, train_independent, train_noise = _latent_rows(
        family, int(training_seed), train_rows
    )
    y_evaluation = _target(family, eval_u, eval_epsilon, noise_sd)
    y_training = _target(family, train_u, train_epsilon, noise_sd)
    candidates = []
    for position, quality in enumerate(data["candidate_quality"]):
        q = float(quality)
        X_train = _representation(family, q, train_u, train_independent, train_noise)
        X_evaluation = _representation(family, q, eval_u, eval_independent, eval_noise)
        truth = _truth_row(family, q, X_evaluation, y_evaluation)
        candidates.append(
            {
                "candidate_id": f"q{position}",
                "quality": q,
                "measurement_noise_tau": _quality_tau(q),
                "X_train": X_train,
                "X_evaluation": X_evaluation,
                "truth": truth,
            }
        )
    return {
        "family": family,
        "training_seed": int(training_seed),
        "evaluation_seed": int(data["evaluation_seed"]),
        "y_train": y_training,
        "y_evaluation": y_evaluation,
        "candidates": candidates,
    }


def _prepare_geometry(
    method_id: str,
    X_train: np.ndarray,
    X_holdout: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    primitive_train = X_train[:, :1]
    primitive_holdout = X_holdout[:, :1]
    if method_id == "observed_oi":
        return save._prepare_fold_geometry(
            "observed", X_train, X_holdout, primitive_train, primitive_holdout, y_train, seed
        )
    if method_id == "save_oi":
        return save._prepare_fold_geometry(
            "save", X_train, X_holdout, primitive_train, primitive_holdout, y_train, seed
        )
    if method_id == "broad_rff":
        return rff_mean._prepare_fold_geometry(
            {"geometry": "rff_mean", "rff_components": 64},
            X_train,
            X_holdout,
            primitive_train,
            primitive_holdout,
            y_train,
            seed,
        )
    raise ValueError(f"unknown selector method: {method_id!r}")


def _evaluate_method(
    method_id: str,
    X: np.ndarray,
    y: np.ndarray,
    algorithm_seed: int,
    protocol: Mapping[str, Any],
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    fold_rows = []
    started = time.perf_counter()
    for fold, (train, holdout) in enumerate(folds):
        fold_seed = _stable_seed("selector", int(algorithm_seed), int(fold))
        transformed_train, transformed_holdout, diagnostics = _prepare_geometry(
            method_id, X[train], X[holdout], y[train], fold_seed
        )
        bank = null_parent._permutation_bank(
            len(holdout),
            32,
            _stable_seed("null-bank", int(protocol["selector"]["fold_seed"]), int(fold)),
        )
        if method_id == "broad_rff":
            # The locked broad score is prototype-free kernel alignment.  The
            # geometry transform is fitted on training rows, then only the
            # held-out feature and target kernels are compared.
            target_scaler = clone(save._base_selector(fold_seed))
            y_train_scaled = target_scaler._scale_targets(
                target_scaler._validate_Y(y[train])
            )
            y_holdout_scaled = target_scaler._transform_targets(
                target_scaler._validate_Y(y[holdout])
            )
            components = kernel._kernel_fold_components(
                transformed_train,
                transformed_holdout,
                y_train_scaled,
                y_holdout_scaled,
                bank,
            )
            score = components["feature_bandwidths"]["4.0"]["score"]
            if score is None:
                raise RuntimeError("broad-RFF kernel alignment was undefined")
            fold_rows.append(
                {
                    "fold": int(fold),
                    "score": float(score),
                    "actual_loss": None,
                    "null_loss": None,
                    "loss_ratio": None,
                    "geometry": diagnostics,
                    "prototype_fit_elided": True,
                    "kernel_alignment": components,
                }
            )
        else:
            estimator = clone(save._base_selector(fold_seed))
            estimator._fit_geometry_only(transformed_train, y[train])
            components = null_parent._fixed_model_components(
                estimator, transformed_holdout, y[holdout], bank
            )
            null_loss = float(np.mean(components["null_loss_samples"]))
            derived = null_parent._score_from_losses(
                float(components["actual_loss"]), null_loss
            )
            fold_rows.append(
                {
                    "fold": int(fold),
                    "score": float(derived["score"]),
                    "actual_loss": float(components["actual_loss"]),
                    "null_loss": null_loss,
                    "loss_ratio": derived["loss_ratio"],
                    "geometry": diagnostics,
                    "prototype_fit_elided": False,
                }
            )
    return {
        "method_id": method_id,
        "algorithm_seed": int(algorithm_seed),
        "score": float(np.mean([row["score"] for row in fold_rows])),
        "folds": fold_rows,
        "wall_seconds": float(time.perf_counter() - started),
    }


def run_panel(
    protocol: Mapping[str, Any], family: str, training_seed: int,
    *, mode: str, algorithm_seeds: Sequence[int], fixed_folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    panel = generate_panel(protocol, family, training_seed)
    rows = []
    for candidate in panel["candidates"]:
        for algorithm_seed in algorithm_seeds:
            for method_id in METHOD_IDS:
                result = _evaluate_method(
                    method_id,
                    candidate["X_train"],
                    panel["y_train"],
                    int(algorithm_seed),
                    protocol,
                    fixed_folds,
                )
                rows.append(
                    {
                        "family": family,
                        "candidate_id": candidate["candidate_id"],
                        "quality": float(candidate["quality"]),
                        "training_seed": int(training_seed),
                        "evaluation_seed": int(panel["evaluation_seed"]),
                        "mode": mode,
                        "method_id": method_id,
                        "algorithm_seed": int(algorithm_seed),
                        "score": float(result["score"]),
                        "folds": result["folds"],
                        "wall_seconds": float(result["wall_seconds"]),
                        "truth": dict(candidate["truth"]),
                    }
                )
    return {
        "family": family,
        "training_seed": int(training_seed),
        "evaluation_seed": int(panel["evaluation_seed"]),
        "mode": mode,
        "rows": rows,
    }


def _mean_score(rows: Sequence[Mapping[str, Any]]) -> float:
    return float(np.mean([float(row["score"]) for row in rows]))


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    result = spearmanr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return None if not np.isfinite(result.statistic) else float(result.statistic)


def summarize(protocol: Mapping[str, Any], raw_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(str(row["mode"]), str(row["family"]), str(row["candidate_id"]), str(row["method_id"]))].append(row)

    algorithm_rows = [row for row in raw_rows if row["mode"] == "algorithm_seed"]
    sampling_rows = [row for row in raw_rows if row["mode"] == "training_sample"]
    method_summaries: dict[str, Any] = {}
    for method_id in METHOD_IDS:
        predictive = {}
        control = {}
        for family in PREDICTIVE_FAMILIES:
            candidate_rows = []
            for candidate_id in ("q0", "q1", "q2"):
                rows = grouped[("algorithm_seed", family, candidate_id, method_id)]
                truth = rows[0]["truth"]
                candidate_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "quality": float(rows[0]["quality"]),
                        "score": _mean_score(rows),
                        "truth_r2": float(truth["evaluation_conditional_mean_r2"]),
                    }
                )
            predictive[family] = {
                "candidates": candidate_rows,
                "spearman": _spearman(
                    [row["score"] for row in candidate_rows],
                    [row["truth_r2"] for row in candidate_rows],
                ),
                "top1_accuracy": float(
                    int(
                        max(candidate_rows, key=lambda row: row["score"])["candidate_id"]
                        == max(candidate_rows, key=lambda row: row["truth_r2"])["candidate_id"]
                    )
                ),
            }
        for family in CONTROL_FAMILIES:
            candidate_rows = []
            for candidate_id in ("q0", "q1", "q2"):
                rows = grouped[("algorithm_seed", family, candidate_id, method_id)]
                candidate_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "quality": float(rows[0]["quality"]),
                        "score": _mean_score(rows),
                        "truth_r2": float(rows[0]["truth"]["evaluation_conditional_mean_r2"]),
                    }
                )
            control[family] = {
                "candidates": candidate_rows,
                "score_range": float(max(row["score"] for row in candidate_rows) - min(row["score"] for row in candidate_rows)),
            }

        algorithm_groups = [
            rows for key, rows in grouped.items()
            if key[0] == "algorithm_seed" and key[3] == method_id
        ]
        sampling_groups = [
            rows for key, rows in grouped.items()
            if key[0] == "training_sample" and key[3] == method_id
        ]
        algorithm_sds = [float(np.std([row["score"] for row in rows], ddof=0)) for rows in algorithm_groups]
        # Sampling rows contain one fixed algorithm seed for each sample.
        sampling_sds = []
        for family in FAMILY_IDS:
            for candidate_id in ("q0", "q1", "q2"):
                rows = [
                    row for row in sampling_rows
                    if row["family"] == family and row["candidate_id"] == candidate_id and row["method_id"] == method_id
                ]
                sampling_sds.append(float(np.std([row["score"] for row in rows], ddof=0)))

        predictive_spearman = [value["spearman"] for value in predictive.values() if value["spearman"] is not None]
        predictive_top1 = [value["top1_accuracy"] for value in predictive.values()]
        control_ranges = [value["score_range"] for value in control.values()]
        gates = protocol["gates"]
        systematic = (
            min(predictive_spearman, default=-1.0) < float(gates["systematic_misalignment"]["predictive_mean_spearman_minimum"])
            or float(np.mean(predictive_top1)) < float(gates["systematic_misalignment"]["predictive_top1_accuracy_minimum"])
            or max(control_ranges, default=0.0) > float(gates["systematic_misalignment"]["control_score_range_maximum"])
        )
        algorithmic = float(np.mean(algorithm_sds)) > float(gates["algorithmic_variance"]["mean_score_sd_maximum"])
        sampling = float(np.mean(sampling_sds)) > float(gates["sampling_variance"]["mean_score_sd_maximum"])
        method_summaries[method_id] = {
            "predictive": predictive,
            "controls": control,
            "variance": {
                "algorithmic": {
                    "mean_score_sd": float(np.mean(algorithm_sds)),
                    "max_score_sd": float(max(algorithm_sds, default=0.0)),
                    "group_count": len(algorithm_sds),
                },
                "sampling": {
                    "mean_score_sd": float(np.mean(sampling_sds)),
                    "max_score_sd": float(max(sampling_sds, default=0.0)),
                    "group_count": len(sampling_sds),
                },
            },
            "classification": {
                "systematic_estimand_misalignment": bool(systematic),
                "algorithmic_variance": bool(algorithmic),
                "sampling_variance": bool(sampling),
            },
        }

    truth_errors = [
        abs(float(row["truth"]["evaluation_conditional_mean_r2"]) - float(row["truth"]["analytic_expected_r2"]))
        for row in raw_rows
    ]
    return {
        "schema_version": 1,
        "row_count": len(raw_rows),
        "method_summaries": method_summaries,
        "truth_validation": {
            "max_empirical_to_analytic_r2_error": float(max(truth_errors, default=0.0)),
            "mean_empirical_to_analytic_r2_error": float(np.mean(truth_errors)) if truth_errors else 0.0,
            "tolerance": float(protocol["predictive_truth"]["truth_tolerance"]),
            "within_tolerance": bool(max(truth_errors, default=0.0) <= float(protocol["predictive_truth"]["truth_tolerance"])),
        },
    }


def validate_raw(protocol: Mapping[str, Any], raw: Mapping[str, Any]) -> None:
    if raw.get("artifact_status") != "completed":
        raise ValueError("mechanism diagnostic raw artifact is incomplete")
    rows = raw.get("rows")
    if not isinstance(rows, list):
        raise ValueError("mechanism diagnostic raw rows are missing")
    expected = int(protocol["artifact_contract"]["expected_method_rows"])
    if len(rows) != expected:
        raise ValueError(f"mechanism diagnostic row count {len(rows)} != {expected}")
    for row in rows:
        if tuple(sorted(row)) != tuple(sorted((
            "family", "candidate_id", "quality", "training_seed", "evaluation_seed", "mode",
            "method_id", "algorithm_seed", "score", "folds", "wall_seconds", "truth",
        ))):
            raise ValueError("mechanism diagnostic row schema changed")
        if row["family"] not in FAMILY_IDS or row["method_id"] not in METHOD_IDS:
            raise ValueError("mechanism diagnostic row identity is invalid")
        if row["mode"] not in {"algorithm_seed", "training_sample"}:
            raise ValueError("mechanism diagnostic row mode is invalid")
        if len(row["folds"]) != int(protocol["selector"]["folds"]):
            raise ValueError("mechanism diagnostic fold count is invalid")
        if not np.isfinite(float(row["score"])) or not np.isfinite(float(row["wall_seconds"])):
            raise ValueError("mechanism diagnostic emitted a nonfinite score or runtime")


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    protocol = load_protocol()
    source_paths = {
        "protocol": PROTOCOL_PATH,
        "runner": RUNNER_PATH,
        "continuous_overlap_index": ROOT / "overlapindex" / "ContinuousOverlapIndex.py",
        "rff_mean_runner": rff_mean.RUNNER_PATH,
        "kernel_alignment_runner": kernel.RUNNER_PATH,
        "null_diagnostic_runner": null_parent.RUNNER_PATH,
        "save_runner": save.RUNNER_PATH,
        "save_geometry": save.GEOMETRY_PATH,
    }
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    output = output.resolve()
    if any((output / name).exists() for name in ("raw_results.json", "summary.json", "manifest.json")):
        raise FileExistsError(f"refusing to overwrite completed mechanism artifacts in {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    fold_indices = list(
        KFold(
            n_splits=int(protocol["selector"]["folds"]),
            shuffle=True,
            random_state=int(protocol["selector"]["fold_seed"]),
        ).split(np.arange(int(protocol["data"]["train_rows"])))
    )
    rows: list[dict[str, Any]] = []
    algorithm_seeds = [int(seed) for seed in protocol["selector"]["algorithm_seeds"]]
    fixed_algorithm_seed = [algorithm_seeds[0]]
    for family in FAMILY_IDS:
        algorithm_checkpoint = checkpoints / f"{family}__algorithm.json"
        if algorithm_checkpoint.exists():
            checkpoint = json.loads(algorithm_checkpoint.read_text(encoding="utf-8"))
            if checkpoint.get("source_hashes") != source_hashes:
                raise ValueError(f"stale mechanism checkpoint: {algorithm_checkpoint}")
            panel_rows = checkpoint["rows"]
        else:
            panel_rows = run_panel(
                protocol,
                family,
                int(protocol["data"]["training_sample_seeds"][0]),
                mode="algorithm_seed",
                algorithm_seeds=algorithm_seeds,
                fixed_folds=fold_indices,
            )["rows"]
            _atomic_json(algorithm_checkpoint, {"source_hashes": source_hashes, "rows": panel_rows})
        rows.extend(panel_rows)
        for training_seed in protocol["data"]["training_sample_seeds"]:
            checkpoint_path = checkpoints / f"{family}__sample_{int(training_seed)}.json"
            if checkpoint_path.exists():
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if checkpoint.get("source_hashes") != source_hashes:
                    raise ValueError(f"stale mechanism checkpoint: {checkpoint_path}")
                panel_rows = checkpoint["rows"]
            else:
                panel_rows = run_panel(
                    protocol,
                    family,
                    int(training_seed),
                    mode="training_sample",
                    algorithm_seeds=fixed_algorithm_seed,
                    fixed_folds=fold_indices,
                )["rows"]
                _atomic_json(checkpoint_path, {"source_hashes": source_hashes, "rows": panel_rows})
            rows.extend(panel_rows)
        print(f"completed family={family}", flush=True)

    raw = {
        "schema_version": 1,
        "artifact_status": "completed",
        "source_hashes": source_hashes,
        "protocol": protocol,
        "rows": rows,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "implementation_commit": os.environ.get("CONTINUOUS_OI_COMMIT"),
        },
    }
    validate_raw(protocol, raw)
    summary = summarize(protocol, rows)
    _atomic_json(output / "raw_results.json", raw)
    _atomic_json(output / "summary.json", summary)
    manifest = {
        "schema_version": 1,
        "artifact_status": "completed",
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "source_hashes": source_hashes,
        "raw_results_sha256": _sha256(output / "raw_results.json"),
        "summary_sha256": _sha256(output / "summary.json"),
        "row_count": len(rows),
    }
    _atomic_json(output / "manifest.json", manifest)
    return {"raw": raw, "summary": summary, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
