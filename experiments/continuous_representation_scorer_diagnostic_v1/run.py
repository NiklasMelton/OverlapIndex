"""Frozen representation-by-scorer diagnostic for continuous ranking."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from experiments.continuous_matched_geometry_diagnostic_v1 import run as matched


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[1]
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SIDECAR = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT = ROOT / "artifacts" / "continuous_representation_scorer_diagnostic_v1" / "results"
RELATIONS = ("linear", "product", "radial")
ORIENTATIONS = ("axis", "rotated_45")
VIEWS = ("observed", "signal", "lift")
HEADS = (
    "screened_linear_top1",
    "screened_pair_top1",
    "screened_linear_top2",
    "screened_quadratic_closure8",
)
METHODS = (
    "linear_cv",
    "screened_head_cv_max",
    "expanded_head_cv_max",
    "oi_observed",
    "oi_signal",
    "oi_lift",
    "save_observed",
    "save_signal",
    "save_lift",
)
ONE_THREAD = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "LOKY_MAX_CPU_COUNT": "1",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(_canonical(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SIDECAR.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or _sha256(PROTOCOL_PATH) != expected:
        raise RuntimeError("representation-scorer protocol sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_execution":
        raise RuntimeError("representation-scorer protocol is not frozen")
    if tuple(protocol["relations"]) != RELATIONS:
        raise RuntimeError("relation set changed")
    if tuple(protocol["orientations"]) != ORIENTATIONS or tuple(protocol["views"]) != VIEWS:
        raise RuntimeError("orientation or view set changed")
    if tuple(protocol["ranking_methods"]) != METHODS:
        raise RuntimeError("ranking method set changed")
    if tuple(protocol["head_models"]["reference_envelope"]) != HEADS:
        raise RuntimeError("reference head set changed")
    expected_panels = (
        len(protocol["scenarios"])
        * len(ORIENTATIONS)
        * len(protocol["data"]["dimensions"])
        * len(protocol["data"]["training_seeds"])
    )
    if expected_panels != int(protocol["execution"]["expected_panels"]):
        raise RuntimeError("frozen panel count is inconsistent")
    return protocol


def _check_threads() -> dict[str, str]:
    observed = {name: os.environ.get(name) for name in ONE_THREAD}
    if observed != ONE_THREAD:
        raise RuntimeError(f"one-thread environment required; got {observed!r}")
    return observed


def panel_keys(protocol: Mapping[str, Any]) -> tuple[tuple[str, str, int, int], ...]:
    return tuple(
        (scenario, orientation, int(dimension), int(seed))
        for scenario in protocol["scenarios"]
        for orientation in ORIENTATIONS
        for dimension in protocol["data"]["dimensions"]
        for seed in protocol["data"]["training_seeds"]
    )


def _quality_tau(quality: float) -> float:
    return float(np.sqrt((1.0 / quality**2) - 1.0))


def _rotation(orientation: str) -> np.ndarray:
    if orientation == "axis":
        return np.eye(2, dtype=np.float64)
    if orientation == "rotated_45":
        angle = np.pi / 4.0
        return np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float64,
        )
    raise ValueError(f"orientation must be one of {ORIENTATIONS!r}; got {orientation!r}")


def _relation_signal(
    relation: str,
    z: np.ndarray,
    sign: np.ndarray,
    radial_sign: np.ndarray,
    filler: np.ndarray,
    radial_weight: float,
    positive_floor: float,
) -> tuple[np.ndarray, float]:
    if relation == "linear":
        return np.column_stack([z, filler]), 0.0
    if relation == "product":
        return np.column_stack([sign, sign * z]), 0.0
    if relation == "radial":
        shift = float(max(0.0, positive_floor - float(np.min(z))))
        positive = z + shift
        radius = np.sqrt(positive / (1.0 + radial_weight))
        return np.column_stack(
            [sign * radius, radial_sign * radius]
        ), shift
    raise ValueError(f"relation must be one of {RELATIONS!r}; got {relation!r}")


def recover_signal(
    signal: np.ndarray,
    relation: str,
    orientation: str,
    shift: float,
    radial_weight: float,
) -> np.ndarray:
    primitive = np.asarray(signal, dtype=np.float64) @ _rotation(orientation).T
    if relation == "linear":
        return primitive[:, 0]
    if relation == "product":
        return primitive[:, 0] * primitive[:, 1] - float(shift)
    if relation == "radial":
        return np.square(primitive[:, 0]) + radial_weight * np.square(primitive[:, 1]) - float(shift)
    raise ValueError(f"relation must be one of {RELATIONS!r}; got {relation!r}")


def generate_panel(
    protocol: Mapping[str, Any],
    scenario: str,
    orientation: str,
    dimension: int,
    training_seed: int,
) -> dict[str, Any]:
    key = (scenario, orientation, int(dimension), int(training_seed))
    if key not in panel_keys(protocol):
        raise ValueError("requested panel is outside the frozen grid")
    n_train = int(protocol["data"]["training_rows"])
    total = n_train + int(protocol["data"]["evaluation_rows"])
    # Orientation is deliberately excluded so axis and rotated panels share
    # every random draw and differ only by the declared two-dimensional map.
    rng = np.random.default_rng(matched._seed("representation-scorer-v1", scenario, dimension, training_seed))
    latent = rng.normal(size=total)
    target_noise = rng.normal(size=total)
    measurement_noise = rng.normal(size=total)
    filler = rng.normal(size=total)
    sign = np.where(rng.integers(0, 2, size=total) == 0, -1.0, 1.0)
    radial_sign = np.where(rng.integers(0, 2, size=total) == 0, -1.0, 1.0)
    nuisance = rng.normal(size=(total, int(dimension) - 2))
    target_sd = float(protocol["data"]["target_noise_standard_deviation"])
    y = latent + target_sd * target_noise
    radial_weight = float(protocol["data"]["radial_weight"])
    positive_floor = float(protocol["data"]["positive_floor"])
    rotation = _rotation(orientation)
    scenario_spec = protocol["scenarios"][scenario]
    candidates = []
    for relation in RELATIONS:
        for role, quality in (
            ("candidate", float(scenario_spec[relation])),
            ("control", float(protocol["controls"]["quality"])),
        ):
            z = latent + _quality_tau(quality) * measurement_noise
            primitive, shift = _relation_signal(
                relation, z, sign, radial_sign, filler, radial_weight, positive_floor
            )
            signal = np.asarray(primitive @ rotation, dtype=np.float64)
            recovered = recover_signal(signal, relation, orientation, shift, radial_weight)
            if not np.allclose(recovered, z, rtol=1e-10, atol=1e-10):
                raise RuntimeError("declared relation failed exact recovery")
            observed = np.column_stack(
                [signal, float(protocol["data"]["nuisance_scale"]) * nuisance]
            )
            lifted = PolynomialFeatures(degree=2, include_bias=False).fit_transform(signal)
            views = {
                "observed": np.asarray(observed, dtype=np.float64),
                "signal": signal,
                "lift": np.asarray(lifted, dtype=np.float64),
            }
            candidates.append(
                {
                    "candidate_id": f"{relation}_{role}",
                    "relation": relation,
                    "role": role,
                    "quality": quality,
                    "analytic_r2": float(quality**2 / (1.0 + target_sd**2)),
                    "shift": shift,
                    "views_train": {name: value[:n_train] for name, value in views.items()},
                    "views_evaluation": {name: value[n_train:] for name, value in views.items()},
                }
            )
    return {
        "scenario": scenario,
        "orientation": orientation,
        "dimension": int(dimension),
        "training_seed": int(training_seed),
        "expected_winner": str(scenario_spec["expected_winner"]),
        "y_train": np.asarray(y[:n_train], dtype=np.float64),
        "y_evaluation": np.asarray(y[n_train:], dtype=np.float64),
        "candidates": candidates,
    }


def _top_indices(values: np.ndarray, count: int) -> np.ndarray:
    order = np.argsort(-np.abs(np.asarray(values)), kind="stable")
    return np.asarray(order[: min(int(count), len(order))], dtype=np.int64)


def _fit_head_features(head: str, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    scaler = StandardScaler().fit(matrix)
    standardized = scaler.transform(matrix)
    centered = target - np.mean(target)
    if head in {"screened_linear_top1", "screened_linear_top2"}:
        count = 1 if head.endswith("top1") else 2
        covariance = standardized.T @ centered / len(standardized)
        selected = _top_indices(covariance, count)
        return {"scaler": scaler, "kind": "linear", "selected_features": selected}
    if head == "screened_pair_top1":
        covariance = (standardized * centered[:, None]).T @ standardized / len(standardized)
        left, right = np.triu_indices(matrix.shape[1], k=1)
        selected = _top_indices(covariance[left, right], 1)
        pairs = np.column_stack([left[selected], right[selected]]).astype(np.int64, copy=False)
        return {"scaler": scaler, "kind": "quadratic", "selected_pairs": pairs}
    raw_centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    covariance = (raw_centered * centered[:, None]).T @ raw_centered / len(raw_centered)
    left, right = np.triu_indices(matrix.shape[1], k=0)
    seeds = _top_indices(covariance[left, right], 8)
    seed_pairs = np.column_stack([left[seeds], right[seeds]]).astype(np.int64, copy=False)
    coordinates = np.unique(seed_pairs)
    closure_left, closure_right = np.triu_indices(len(coordinates), k=0)
    pairs = np.column_stack(
        [coordinates[closure_left], coordinates[closure_right]]
    ).astype(np.int64, copy=False)
    return {
        "scaler": scaler,
        "kind": "quadratic",
        "selected_pairs": pairs,
        "seed_pairs": seed_pairs,
    }


def _transform_head_features(state: Mapping[str, Any], X: np.ndarray) -> np.ndarray:
    standardized = state["scaler"].transform(np.asarray(X, dtype=np.float64))
    if state["kind"] == "linear":
        return standardized[:, np.asarray(state["selected_features"], dtype=np.int64)]
    pairs = np.asarray(state["selected_pairs"], dtype=np.int64)
    return standardized[:, pairs[:, 0]] * standardized[:, pairs[:, 1]]


def evaluate_screened_head(
    protocol: Mapping[str, Any],
    head: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_evaluation: np.ndarray,
    y_evaluation: np.ndarray,
) -> dict[str, Any]:
    if head not in HEADS:
        raise ValueError(f"head must be one of {HEADS!r}; got {head!r}")
    head_config = protocol["head_models"]
    alphas = tuple(float(value) for value in head_config["alpha"])
    splitter = KFold(
        n_splits=int(head_config["inner_folds"]),
        shuffle=True,
        random_state=int(head_config["fold_seed"]),
    )
    started = time.perf_counter()
    fold_scores: dict[float, list[float]] = {alpha: [] for alpha in alphas}
    for train, holdout in splitter.split(X_train):
        state = _fit_head_features(head, X_train[train], y_train[train])
        features_train = _transform_head_features(state, X_train[train])
        features_holdout = _transform_head_features(state, X_train[holdout])
        for alpha in alphas:
            model = Ridge(alpha=alpha).fit(features_train, y_train[train])
            fold_scores[alpha].append(float(r2_score(y_train[holdout], model.predict(features_holdout))))
    means = {alpha: float(np.mean(values)) for alpha, values in fold_scores.items()}
    best_alpha = max(alphas, key=lambda alpha: (means[alpha], -alpha))
    final_state = _fit_head_features(head, X_train, y_train)
    final_train = _transform_head_features(final_state, X_train)
    final_evaluation = _transform_head_features(final_state, X_evaluation)
    final_model = Ridge(alpha=best_alpha).fit(final_train, y_train)
    output = {
        "head": head,
        "inner_best_r2": means[best_alpha],
        "evaluation_r2": float(r2_score(y_evaluation, final_model.predict(final_evaluation))),
        "best_alpha": best_alpha,
        "wall_seconds": float(time.perf_counter() - started),
    }
    if final_state["kind"] == "linear":
        output["selected_features"] = [int(value) for value in final_state["selected_features"]]
    else:
        output["selected_pairs"] = [
            [int(left), int(right)] for left, right in final_state["selected_pairs"]
        ]
    return output


def _linear_cv(protocol: Mapping[str, Any], X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    config = protocol["head_models"]
    splitter = KFold(
        n_splits=int(config["inner_folds"]),
        shuffle=True,
        random_state=int(config["fold_seed"]),
    )
    alphas = tuple(float(value) for value in config["alpha"])
    scores: dict[float, list[float]] = {alpha: [] for alpha in alphas}
    started = time.perf_counter()
    for train, holdout in splitter.split(X):
        scaler = StandardScaler().fit(X[train])
        train_values = scaler.transform(X[train])
        holdout_values = scaler.transform(X[holdout])
        for alpha in alphas:
            model = Ridge(alpha=alpha).fit(train_values, y[train])
            scores[alpha].append(float(r2_score(y[holdout], model.predict(holdout_values))))
    means = {alpha: float(np.mean(values)) for alpha, values in scores.items()}
    best_alpha = max(alphas, key=lambda alpha: (means[alpha], -alpha))
    return {
        "score": means[best_alpha],
        "best_alpha": best_alpha,
        "wall_seconds": float(time.perf_counter() - started),
    }


def evaluate_panel(
    protocol: Mapping[str, Any],
    scenario: str,
    orientation: str,
    dimension: int,
    training_seed: int,
) -> dict[str, Any]:
    panel = generate_panel(protocol, scenario, orientation, dimension, training_seed)
    folds = list(
        KFold(
            n_splits=int(protocol["oi"]["folds"]),
            shuffle=True,
            random_state=int(protocol["oi"]["fold_seed"]),
        ).split(np.arange(int(protocol["data"]["training_rows"])))
    )
    mechanism_protocol = {"selector": {"fold_seed": int(protocol["oi"]["fold_seed"])}}
    selector_rows = []
    reference_rows = []
    for candidate in panel["candidates"]:
        observed_train = candidate["views_train"]["observed"]
        observed_evaluation = candidate["views_evaluation"]["observed"]
        head_rows = [
            evaluate_screened_head(
                protocol,
                head,
                observed_train,
                panel["y_train"],
                observed_evaluation,
                panel["y_evaluation"],
            )
            for head in HEADS
        ]
        reference_rows.extend(
            {"candidate_id": candidate["candidate_id"], **row} for row in head_rows
        )
        linear = _linear_cv(protocol, observed_train, panel["y_train"])
        selector_rows.append(
            {"candidate_id": candidate["candidate_id"], "method_id": "linear_cv", "algorithm_seed": None, **linear}
        )
        for method_id, heads in (
            ("screened_head_cv_max", protocol["head_models"]["existing_heads"]),
            ("expanded_head_cv_max", protocol["head_models"]["expanded_heads"]),
        ):
            selected = [row for row in head_rows if row["head"] in heads]
            selector_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "method_id": method_id,
                    "algorithm_seed": None,
                    "score": float(max(row["inner_best_r2"] for row in selected)),
                    "wall_seconds": float(sum(row["wall_seconds"] for row in selected)),
                }
            )
        for view in VIEWS:
            for base_method, prefix in (("observed_oi", "oi"), ("save_oi", "save")):
                method_id = f"{prefix}_{view}"
                for algorithm_seed in protocol["oi"]["algorithm_seeds"]:
                    result = matched.mechanism._evaluate_method(
                        base_method,
                        candidate["views_train"][view],
                        panel["y_train"],
                        int(algorithm_seed),
                        mechanism_protocol,
                        folds,
                    )
                    selector_rows.append(
                        {
                            "candidate_id": candidate["candidate_id"],
                            "method_id": method_id,
                            "algorithm_seed": int(algorithm_seed),
                            "score": float(result["score"]),
                            "wall_seconds": float(result["wall_seconds"]),
                        }
                    )
    truths = [
        {key: candidate[key] for key in ("candidate_id", "relation", "role", "quality", "analytic_r2")}
        for candidate in panel["candidates"]
    ]
    return {
        "schema_version": 1,
        "artifact_status": "completed_panel",
        "scenario": scenario,
        "orientation": orientation,
        "dimension": int(dimension),
        "training_seed": int(training_seed),
        "expected_winner": panel["expected_winner"],
        "truths": truths,
        "selector_rows": selector_rows,
        "reference_rows": reference_rows,
    }


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def _material_pair_accuracy(
    scores: np.ndarray, truths: np.ndarray, threshold: float
) -> tuple[int, int]:
    correct = 0
    total = 0
    for left in range(len(scores)):
        for right in range(left + 1, len(scores)):
            difference = float(truths[left] - truths[right])
            if abs(difference) < threshold:
                continue
            total += 1
            score_difference = float(scores[left] - scores[right])
            correct += int(score_difference * difference > 0.0)
    return correct, total


def summarize(protocol: Mapping[str, Any], panels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    panel_rows = []
    reference_checks = []
    material_threshold = float(protocol["selector_gates"]["material_pair_r2_difference"])
    for panel in panels:
        candidate_ids = [row["candidate_id"] for row in panel["truths"]]
        truth = np.asarray([row["analytic_r2"] for row in panel["truths"]], dtype=float)
        reference = {
            candidate: max(
                float(row["evaluation_r2"])
                for row in panel["reference_rows"]
                if row["candidate_id"] == candidate
            )
            for candidate in candidate_ids
        }
        expected = str(panel["expected_winner"])
        runner_up = max(value for name, value in reference.items() if name != expected)
        max_gap = max(abs(reference[name] - truth[position]) for position, name in enumerate(candidate_ids))
        reference_checks.append(
            {
                "scenario": panel["scenario"],
                "orientation": panel["orientation"],
                "dimension": panel["dimension"],
                "training_seed": panel["training_seed"],
                "expected_winner": expected,
                "observed_winner": max(reference, key=reference.get),
                "winner_gap_r2": float(reference[expected] - runner_up),
                "maximum_absolute_r2_gap_to_analytic": float(max_gap),
                "pass": bool(
                    max(reference, key=reference.get) == expected
                    and reference[expected] - runner_up
                    >= float(protocol["reference_gates"]["minimum_expected_winner_gap_r2"])
                    and max_gap
                    <= float(protocol["reference_gates"]["maximum_absolute_r2_gap_to_analytic"])
                ),
            }
        )
        for method in METHODS:
            rows = [row for row in panel["selector_rows"] if row["method_id"] == method]
            scores = np.asarray(
                [
                    np.mean([row["score"] for row in rows if row["candidate_id"] == candidate])
                    for candidate in candidate_ids
                ],
                dtype=float,
            )
            selected = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0.0, atol=1e-12))
            correct, total = _material_pair_accuracy(scores, truth, material_threshold)
            quality_groups: dict[float, list[float]] = defaultdict(list)
            for position, row in enumerate(panel["truths"]):
                quality_groups[float(row["quality"])].append(float(scores[position]))
            equal_ranges = [max(values) - min(values) for values in quality_groups.values() if len(values) > 1]
            panel_rows.append(
                {
                    "scenario": panel["scenario"],
                    "orientation": panel["orientation"],
                    "dimension": panel["dimension"],
                    "training_seed": panel["training_seed"],
                    "method_id": method,
                    "selected_candidate_ids": [candidate_ids[int(index)] for index in selected],
                    "direction_correct": bool(candidate_ids.index(expected) in selected),
                    "regret_r2": float(truth.max() - truth[selected].max()),
                    "spearman": _spearman(scores, truth),
                    "material_pairs_correct": int(correct),
                    "material_pairs_total": int(total),
                    "equal_quality_score_range": float(max(equal_ranges, default=0.0)),
                    "mean_wall_seconds": float(np.mean([row["wall_seconds"] for row in rows])),
                }
            )
    reference_valid = all(row["pass"] for row in reference_checks)
    gates = protocol["selector_gates"]
    method_aggregates = []
    for method in METHODS:
        rows = [row for row in panel_rows if row["method_id"] == method]
        correct = sum(row["material_pairs_correct"] for row in rows)
        total = sum(row["material_pairs_total"] for row in rows)
        metrics = {
            "direction_accuracy": float(np.mean([row["direction_correct"] for row in rows])),
            "mean_regret_r2": float(np.mean([row["regret_r2"] for row in rows])),
            "mean_spearman": float(np.mean([row["spearman"] for row in rows if row["spearman"] is not None])),
            "material_pair_accuracy": float(correct / total),
            "mean_equal_quality_score_range": float(np.mean([row["equal_quality_score_range"] for row in rows])),
            "mean_wall_seconds": float(np.mean([row["mean_wall_seconds"] for row in rows])),
        }
        survives = bool(
            reference_valid
            and metrics["mean_regret_r2"] <= float(gates["maximum_mean_regret_r2"])
            and metrics["direction_accuracy"] >= float(gates["minimum_direction_accuracy"])
            and metrics["mean_spearman"] >= float(gates["minimum_mean_spearman"])
            and metrics["material_pair_accuracy"] >= float(gates["minimum_material_pair_accuracy"])
            and metrics["mean_equal_quality_score_range"]
            <= float(gates["maximum_mean_equal_quality_score_range"])
        )
        method_aggregates.append({"method_id": method, **metrics, "survives": survives})
    scenario_aggregates = []
    for scenario in protocol["scenarios"]:
        for method in METHODS:
            rows = [
                row
                for row in panel_rows
                if row["scenario"] == scenario and row["method_id"] == method
            ]
            scenario_aggregates.append(
                {
                    "scenario": scenario,
                    "method_id": method,
                    "direction_accuracy": float(np.mean([row["direction_correct"] for row in rows])),
                    "mean_regret_r2": float(np.mean([row["regret_r2"] for row in rows])),
                    "mean_spearman": float(np.mean([row["spearman"] for row in rows if row["spearman"] is not None])),
                }
            )
    contrast_rows = []
    for base in ("oi", "save"):
        lookup = {
            (row["scenario"], row["orientation"], row["dimension"], row["training_seed"], row["method_id"]): row
            for row in panel_rows
            if row["method_id"].startswith(f"{base}_")
        }
        for scenario, orientation, dimension, seed in panel_keys(protocol):
            observed = lookup[(scenario, orientation, dimension, seed, f"{base}_observed")]
            signal = lookup[(scenario, orientation, dimension, seed, f"{base}_signal")]
            lift = lookup[(scenario, orientation, dimension, seed, f"{base}_lift")]
            contrast_rows.append(
                {
                    "base_method": base,
                    "scenario": scenario,
                    "orientation": orientation,
                    "dimension": dimension,
                    "training_seed": seed,
                    "observed_to_signal_regret_change": float(signal["regret_r2"] - observed["regret_r2"]),
                    "signal_to_lift_regret_change": float(lift["regret_r2"] - signal["regret_r2"]),
                    "direction": {
                        "observed": observed["direction_correct"],
                        "signal": signal["direction_correct"],
                        "lift": lift["direction_correct"],
                    },
                }
            )
    return {
        "schema_version": 1,
        "reference_valid": reference_valid,
        "reference_checks": reference_checks,
        "panel_rows": panel_rows,
        "method_aggregates": method_aggregates,
        "scenario_aggregates": scenario_aggregates,
        "contrast_rows": contrast_rows,
    }


def _identity(protocol: Mapping[str, Any], threads: Mapping[str, str]) -> dict[str, Any]:
    source_paths = {
        "protocol": PROTOCOL_PATH,
        "runner": Path(__file__).resolve(),
        "mechanism_runner": matched.mechanism.RUNNER_PATH,
        "matched_runner": matched.RUNNER_PATH,
    }
    return {
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
        "threads": dict(threads),
        "expected_panels": int(protocol["execution"]["expected_panels"]),
    }


def _checkpoint_path(output: Path, key: Sequence[Any]) -> Path:
    panel_id = "__".join(str(value) for value in key)
    return output / "checkpoints" / f"{hashlib.sha256(panel_id.encode()).hexdigest()[:24]}.json"


def run_shard(output: Path, shard_index: int, shard_count: int) -> dict[str, Any]:
    protocol = load_protocol()
    threads = _check_threads()
    if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard index must be in [0, shard_count)")
    output = output.resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError("terminal artifact exists; refusing shard execution")
    identity = _identity(protocol, threads)
    selected = [key for position, key in enumerate(panel_keys(protocol)) if position % shard_count == shard_index]
    completed = 0
    for key in selected:
        path = _checkpoint_path(output, key)
        if path.exists():
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
            if checkpoint.get("identity") != identity:
                raise RuntimeError(f"stale representation-scorer checkpoint: {path}")
        else:
            panel = evaluate_panel(protocol, *key)
            _atomic_json(path, {"schema_version": 1, "identity": identity, "panel": panel})
        completed += 1
        print(f"shard {shard_index}: completed {completed}/{len(selected)}: {key}", flush=True)
    return {"shard_index": shard_index, "shard_count": shard_count, "completed": completed}


def finalize(output: Path) -> dict[str, Any]:
    protocol = load_protocol()
    threads = _check_threads()
    output = output.resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError("terminal artifact exists; refusing overwrite")
    identity = _identity(protocol, threads)
    panels = []
    for key in panel_keys(protocol):
        path = _checkpoint_path(output, key)
        if not path.exists():
            raise RuntimeError(f"missing representation-scorer checkpoint: {path}")
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity:
            raise RuntimeError(f"stale representation-scorer checkpoint: {path}")
        panels.append(checkpoint["panel"])
    selector_count = sum(len(panel["selector_rows"]) for panel in panels)
    reference_count = sum(len(panel["reference_rows"]) for panel in panels)
    if selector_count != int(protocol["execution"]["expected_selector_rows"]):
        raise RuntimeError("selector row count is incomplete")
    if reference_count != int(protocol["execution"]["expected_reference_rows"]):
        raise RuntimeError("reference row count is incomplete")
    summary = summarize(protocol, panels)
    raw = {
        "schema_version": 1,
        "artifact_status": "completed",
        "identity": identity,
        "panels": panels,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    _atomic_json(output / "raw_results.json", raw)
    _atomic_json(output / "summary.json", summary)
    manifest = {
        "schema_version": 1,
        "artifact_status": "completed",
        "identity": identity,
        "panel_count": len(panels),
        "selector_row_count": selector_count,
        "reference_row_count": reference_count,
        "raw_results_sha256": _sha256(output / "raw_results.json"),
        "summary_sha256": _sha256(output / "summary.json"),
    }
    _atomic_json(output / "manifest.json", manifest)
    return {"summary": summary, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        if args.shard_index is not None or args.shard_count is not None:
            parser.error("--finalize cannot be combined with shard arguments")
        result = finalize(args.output)
    else:
        if args.shard_index is None or args.shard_count is None:
            parser.error("shard execution requires --shard-index and --shard-count")
        result = run_shard(args.output, args.shard_index, args.shard_count)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
