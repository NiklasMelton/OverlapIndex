"""Frozen nonlinear-balanced residual-only continuous OI factorial."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import scipy
import sklearn
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from experiments.continuous_fisher_ranking import run as common
from overlapindex import ContinuousOverlapIndex


PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
RUNNER_PATH = Path(__file__).resolve()
COMMON_RUNNER_PATH = common.RUNNER_PATH


def _load_protocol() -> Dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_exploratory_before_outcomes":
        raise ValueError("protocol is not frozen for exploratory execution")
    return protocol


def _orthogonal_mix(rng: np.random.Generator, dimension: int) -> np.ndarray:
    basis, _triangular = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    return np.asarray(basis, dtype=np.float64)


def _latent_bank(
    seed: int,
    family: str,
    n_rows: int,
    dimension: int,
) -> Dict[str, np.ndarray]:
    families = tuple(_load_protocol()["families"])
    position = families.index(family)
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(position), 20260901])
    )
    u = rng.uniform(-1.0, 1.0, n_rows)
    v = rng.uniform(-1.0, 1.0, n_rows)
    signal_noise = rng.normal(size=(n_rows, 4))
    nuisance_width = dimension - 4
    iid = rng.normal(size=(n_rows, nuisance_width))
    latent = rng.normal(size=(n_rows, 3))
    loading = rng.normal(size=(3, nuisance_width))
    loading /= np.maximum(
        np.linalg.norm(loading, axis=0, keepdims=True),
        np.finfo(float).eps,
    )
    correlated = latent @ loading
    if family in {"correlated_nuisance", "rotated_nuisance"}:
        nuisance = 0.15 * iid + 2.80 * correlated
    else:
        nuisance = 0.55 * iid + 0.55 * correlated
    return {
        "u": u,
        "v": v,
        "signal_noise": signal_noise,
        "nuisance": nuisance,
        "mix": _orthogonal_mix(rng, dimension),
    }


def _target_and_signal(
    bank: Mapping[str, np.ndarray],
    family: str,
    quality: float,
) -> tuple[np.ndarray, np.ndarray]:
    u = bank["u"]
    v = bank["v"]
    noise = bank["signal_noise"]
    sigma = 0.025 + 0.65 * (1.0 - float(quality))
    heteroscedastic = False
    if family in {"linear", "correlated_nuisance"}:
        target = u
        primitive = np.column_stack([u, 0.55 * u])
    elif family == "heteroscedastic_linear":
        target = u
        primitive = np.column_stack([u, 0.55 * u])
        heteroscedastic = True
    elif family == "rotated_nuisance":
        target = 0.70 * u + 0.30 * v
        primitive = np.column_stack([u, v])
    elif family == "quadratic_product":
        target = u * v
        primitive = np.column_stack([u, v])
    elif family == "radial_quadratic":
        target = np.square(u) + 0.65 * np.square(v)
        primitive = np.column_stack([u, v])
    elif family == "sinusoid":
        target = np.sin(2.5 * np.pi * u)
        primitive = np.column_stack([u, 0.20 * u])
    elif family == "sawtooth":
        phase = 3.0 * (u + 1.0)
        target = 2.0 * (phase - np.floor(phase)) - 1.0
        primitive = np.column_stack([u, 0.20 * u])
    else:  # pragma: no cover - protocol closes this set
        raise ValueError(f"unknown family: {family!r}")
    if heteroscedastic:
        primitive = primitive + (
            sigma * (0.15 + np.abs(target))[:, None] * noise[:, :2]
        )
    else:
        primitive = primitive + sigma * noise[:, :2]
    signal = np.column_stack(
        [
            primitive[:, 0],
            primitive[:, 1],
            0.70 * primitive[:, 0] + 0.20 * primitive[:, 1],
            0.20 * primitive[:, 0] - 0.80 * primitive[:, 1],
        ]
    )
    return np.asarray(target, dtype=np.float64), signal


def generate_panel(
    protocol: Mapping[str, Any],
    seed: int,
    family: str,
) -> Dict[str, Any]:
    rows = protocol["rows"]
    n_train = int(rows["train"])
    n_test = int(rows["test"])
    dimension = int(rows["features"])
    bank = _latent_bank(seed, family, n_train + n_test, dimension)
    target_reference = None
    candidates = []
    for spec in protocol["candidate_specs"]:
        target, signal = _target_and_signal(bank, family, float(spec["quality"]))
        if target_reference is None:
            target_reference = target
        elif not np.array_equal(target_reference, target):
            raise RuntimeError("candidate targets diverged within a panel")
        values = np.column_stack(
            [signal, float(spec["nuisance_scale"]) * bank["nuisance"]]
        )
        values = np.asarray(values @ bank["mix"], dtype=np.float32)
        if values.shape != (n_train + n_test, dimension):
            raise RuntimeError("generated representation has the wrong shape")
        candidates.append(
            {
                **dict(spec),
                "X_train": values[:n_train],
                "X_test": values[n_train:],
            }
        )
    assert target_reference is not None
    return {
        "seed": int(seed),
        "family": family,
        "y_train": target_reference[:n_train],
        "y_test": target_reference[n_train:],
        "candidates": candidates,
    }


def _head_estimators() -> Dict[str, tuple[Any, Dict[str, Iterable[Any]]]]:
    return {
        "linear_ridge": (
            Pipeline([("scale", StandardScaler()), ("model", Ridge())]),
            {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "quadratic_ridge": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("quadratic", PolynomialFeatures(degree=2, include_bias=False)),
                    ("model", Ridge()),
                ]
            ),
            {"model__alpha": [0.01, 0.1, 1.0, 10.0]},
        ),
        "knn": (
            Pipeline([("scale", StandardScaler()), ("model", KNeighborsRegressor())]),
            {
                "model__n_neighbors": [3, 5, 9, 15, 31],
                "model__weights": ["distance"],
            },
        ),
        "rbf_svr": (
            Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf"))]),
            {
                "model__C": [1.0, 10.0, 100.0],
                "model__gamma": ["scale", 0.03, 0.1, 0.3],
                "model__epsilon": [0.05, 0.1],
            },
        ),
    }


def evaluate_heads(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    inner = KFold(n_splits=3, shuffle=True, random_state=int(seed))
    results: Dict[str, Any] = {}
    for head_id, (estimator, grid) in _head_estimators().items():
        search = GridSearchCV(
            clone(estimator),
            param_grid=grid,
            cv=inner,
            scoring="r2",
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        started = time.perf_counter()
        search.fit(X_train, y_train)
        prediction = search.predict(X_test)
        results[head_id] = {
            "test_r2": float(r2_score(y_test, prediction)),
            "inner_best_r2": float(search.best_score_),
            "best_params": dict(search.best_params_),
            "wall_seconds": float(time.perf_counter() - started),
        }
    return results


def method_specs() -> Dict[str, Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {
        "legacy_mbk": {
            "model_type": "MiniBatchKMeans",
            "feature_normalization": None,
            "feature_transform": None,
            "relevance_weighting": False,
        },
        "fused_l2_target_relevance": {
            "model_type": "FusedKMeans",
            "feature_normalization": "l2",
            "feature_transform": None,
            "relevance_weighting": True,
        },
        "fisher_full_no_relevance": {
            "model_type": "FusedKMeans",
            "feature_normalization": "l2",
            "feature_transform": "shared_fisher",
            "fisher_rank": "full",
            "relevance_weighting": False,
        },
        "fisher_full_target_relevance": {
            "model_type": "FusedKMeans",
            "feature_normalization": "l2",
            "feature_transform": "shared_fisher",
            "fisher_rank": "full",
            "relevance_weighting": True,
        },
    }
    for rank_name, rank in (("0", 0), ("8", 8), ("16", 16), ("full", "full")):
        method_prefix = (
            "residual_full" if rank_name == "full" else f"residual{rank_name}"
        )
        for suffix, relevance in (
            ("no_relevance", False),
            ("target_relevance", True),
        ):
            specs[f"{method_prefix}_{suffix}"] = {
                "model_type": "FusedKMeans",
                "feature_normalization": "l2",
                "feature_transform": "shared_residual",
                "fisher_rank": rank,
                "relevance_weighting": relevance,
            }
    return specs


def build_selector(method_id: str, seed: int) -> ContinuousOverlapIndex:
    spec = method_specs()[method_id]
    common_params: Dict[str, Any] = {
        "model_type": spec["model_type"],
        "kmeans_k": 4,
        "n_target_cells": 8,
        "adjacency_mode": "soft_topk",
        "top_k": 5,
        "feature_temperature": 1.0,
        "aggregation": "support_weighted",
        "null_mode": "refit_permutation",
        "n_null_permutations": 4,
        "random_state": int(seed),
        "feature_normalization": spec.get("feature_normalization"),
        "feature_transform": spec.get("feature_transform"),
        "fisher_rank": spec.get("fisher_rank", 32),
        "fisher_random_state": int(seed),
    }
    if spec["model_type"] == "MiniBatchKMeans":
        common_params["kmeans_kwargs"] = {
            "random_state": int(seed),
            "n_init": 1,
            "batch_size": 128,
        }
    else:
        common_params["fused_kmeans_kwargs"] = {
            "random_state": int(seed),
            "n_iter": 3,
            "min_samples_per_prototype": 5,
            "relevance_weighting": bool(spec["relevance_weighting"]),
            "margin_rows_per_class": 32,
            "weight_floor": 0.05,
        }
    return ContinuousOverlapIndex(**common_params)


def evaluate_selector(
    method_id: str,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    selector = build_selector(method_id, seed)
    started = time.perf_counter()
    score = selector.cross_fit_score(X, y, n_splits=4, random_state=int(seed))
    return {
        "score": float(score),
        "fold_scores": [float(value) for value in selector.cross_fit_scores_],
        "wall_seconds": float(time.perf_counter() - started),
    }


def run_panel(protocol: Mapping[str, Any], seed: int, family: str) -> Dict[str, Any]:
    panel = generate_panel(protocol, seed, family)
    method_ids = list(protocol["continuous_oi"]["methods"])
    candidates = []
    for position, candidate in enumerate(panel["candidates"]):
        candidate_seed = int(seed) * 100 + int(position)
        heads = evaluate_heads(
            candidate["X_train"],
            panel["y_train"],
            candidate["X_test"],
            panel["y_test"],
            candidate_seed,
        )
        selectors = {
            method_id: evaluate_selector(
                method_id,
                candidate["X_train"],
                panel["y_train"],
                candidate_seed,
            )
            for method_id in method_ids
        }
        head_scores = {
            head_id: float(result["test_r2"])
            for head_id, result in heads.items()
        }
        envelope = max(head_scores.values())
        winners = sorted(
            head_id
            for head_id, value in head_scores.items()
            if np.isclose(value, envelope, atol=1.0e-12, rtol=0.0)
        )
        candidates.append(
            {
                "candidate_id": candidate["candidate_id"],
                "quality": float(candidate["quality"]),
                "nuisance_scale": float(candidate["nuisance_scale"]),
                "heads": heads,
                "best_head_envelope_r2": float(envelope),
                "best_head_ids": winners,
                "selectors": selectors,
            }
        )
    return {
        "seed": int(seed),
        "family": family,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def summarize(protocol: Mapping[str, Any], panels: list[Mapping[str, Any]]) -> Dict[str, Any]:
    summary = common.summarize(protocol, panels)
    unique_counts = {head_id: 0 for head_id in protocol["reference_heads"]["families"]}
    for panel in panels:
        for candidate in panel["candidates"]:
            winners = candidate["best_head_ids"]
            if len(winners) == 1:
                unique_counts[winners[0]] += 1
    minimums = protocol["head_diversity_gate"]["minimum_unique_wins"]
    checks = {
        head_id: {
            "observed": int(unique_counts[head_id]),
            "minimum": int(minimums[head_id]),
            "pass": bool(unique_counts[head_id] >= int(minimums[head_id])),
        }
        for head_id in minimums
    }
    summary["head_diversity_gate"] = {
        "status": "pass" if all(row["pass"] for row in checks.values()) else "fail",
        "checks": checks,
    }
    return summary


def _validate_protocol_against_code(protocol: Mapping[str, Any]) -> None:
    if list(protocol["continuous_oi"]["methods"]) != list(method_specs()):
        raise ValueError("protocol method order does not match code")
    if len(protocol["candidate_specs"]) != 8:
        raise ValueError("frozen candidate grid must contain eight candidates")
    if len(protocol["families"]) != 8 or len(protocol["seeds"]) != 3:
        raise ValueError("frozen panel grid must contain 3 seeds by 8 families")
    head_ids = set(protocol["reference_heads"]["families"])
    if set(protocol["head_diversity_gate"]["minimum_unique_wins"]) != head_ids:
        raise ValueError("head diversity gate must cover every reference head")


def run(output: Path) -> Dict[str, Any]:
    protocol = _load_protocol()
    _validate_protocol_against_code(protocol)
    source_hashes = {
        "protocol": common._sha256(PROTOCOL_PATH),
        "runner": common._sha256(RUNNER_PATH),
        "common_runner": common._sha256(COMMON_RUNNER_PATH),
    }
    output = output.resolve()
    for terminal in ("raw_results.json", "summary.json", "manifest.json"):
        if (output / terminal).exists():
            raise FileExistsError(f"refusing to overwrite {output / terminal}")
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    panels = []
    total = len(protocol["seeds"]) * len(protocol["families"])
    completed = 0
    for seed in protocol["seeds"]:
        for family in protocol["families"]:
            path = checkpoints / f"seed_{seed}__{family}.json"
            if path.exists():
                checkpoint = json.loads(path.read_text(encoding="utf-8"))
                if checkpoint["source_hashes"] != source_hashes:
                    raise ValueError(f"stale source checkpoint: {path}")
                panel = checkpoint["panel"]
            else:
                started = time.perf_counter()
                panel = run_panel(protocol, int(seed), str(family))
                common._atomic_json(
                    path,
                    {
                        "schema_version": 1,
                        "source_hashes": source_hashes,
                        "panel_wall_seconds": float(time.perf_counter() - started),
                        "panel": panel,
                    },
                )
            panels.append(panel)
            completed += 1
            print(f"completed {completed}/{total}: seed={seed} family={family}", flush=True)
    raw = {
        "schema_version": 1,
        "artifact_status": "completed",
        "source_hashes": source_hashes,
        "implementation_commit": os.environ.get("CONTINUOUS_OI_COMMIT"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "protocol": protocol,
        "panels": panels,
    }
    summary = summarize(protocol, panels)
    common._atomic_json(output / "raw_results.json", raw)
    common._atomic_json(output / "summary.json", summary)
    manifest = {
        "schema_version": 1,
        "artifact_status": "completed",
        "source_hashes": source_hashes,
        "files": {
            name: {
                "sha256": common._sha256(output / name),
                "bytes": (output / name).stat().st_size,
            }
            for name in ("raw_results.json", "summary.json")
        },
        "checkpoint_count": len(panels),
    }
    common._atomic_json(output / "manifest.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.output)
    print(
        json.dumps(
            {
                "head_diversity_gate": summary["head_diversity_gate"],
                "methods": {
                    method_id: {
                        "mean_spearman": row["mean_spearman"],
                        "mean_regret_r2": row["mean_regret_r2"],
                        "within_0_01_rate": row["within_0_01_rate"],
                        "median_candidate_wall_seconds": row[
                            "median_candidate_wall_seconds"
                        ],
                    }
                    for method_id, row in summary["methods"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
