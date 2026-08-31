"""Frozen oracle bottleneck diagnostic for continuous OI."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

from experiments.continuous_fisher_ranking import run as common
from experiments.continuous_residual_ranking import run as residual
from overlapindex import ContinuousOverlapIndex


PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
RUNNER_PATH = Path(__file__).resolve()
COMMON_RUNNER_PATH = common.RUNNER_PATH
RESIDUAL_RUNNER_PATH = residual.RUNNER_PATH


def _load_protocol() -> Dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_exploratory_before_outcomes":
        raise ValueError("protocol is not frozen for exploratory execution")
    return protocol


def _method_specs(protocol: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        row["method_id"]: dict(row)
        for row in protocol["continuous_oi"]["methods"]
    }


def _orthogonal_mix(rng: np.random.Generator, dimension: int) -> np.ndarray:
    basis, _triangular = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    return np.asarray(basis, dtype=np.float64)


def _latent_bank(
    protocol: Mapping[str, Any],
    seed: int,
    family: str,
    n_rows: int,
) -> Dict[str, np.ndarray]:
    position = list(protocol["families"]).index(family)
    dimension = int(protocol["rows"]["observed_features"])
    signal_dimension = int(protocol["rows"]["signal_features"])
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(position), 20260902])
    )
    u = rng.uniform(-1.0, 1.0, n_rows)
    v = rng.uniform(-1.0, 1.0, n_rows)
    signal_noise = rng.normal(size=(n_rows, 2))
    nuisance_width = dimension - signal_dimension
    iid = rng.normal(size=(n_rows, nuisance_width))
    latent = rng.normal(size=(n_rows, 3))
    loading = rng.normal(size=(3, nuisance_width))
    loading /= np.maximum(
        np.linalg.norm(loading, axis=0, keepdims=True),
        np.finfo(float).eps,
    )
    nuisance = 0.45 * iid + 0.85 * (latent @ loading)
    return {
        "u": u,
        "v": v,
        "signal_noise": signal_noise,
        "nuisance": nuisance,
        "mix": _orthogonal_mix(rng, dimension),
    }


def _target_and_primitive(
    bank: Mapping[str, np.ndarray],
    family: str,
    quality: float,
) -> tuple[np.ndarray, np.ndarray]:
    u = bank["u"]
    v = bank["v"]
    sigma = 0.02 + 0.65 * (1.0 - float(quality))
    if family == "linear":
        target = u
        latent = np.column_stack([u, 0.55 * u])
    elif family == "quadratic_product":
        target = u * v
        latent = np.column_stack([u, v])
    elif family == "radial_quadratic":
        target = np.square(u) + 0.65 * np.square(v)
        latent = np.column_stack([u, v])
    elif family == "sinusoid":
        target = np.sin(2.5 * np.pi * u)
        latent = np.column_stack([u, 0.20 * u])
    elif family == "sawtooth":
        phase = 3.0 * (u + 1.0)
        target = 2.0 * (phase - np.floor(phase)) - 1.0
        latent = np.column_stack([u, 0.20 * u])
    else:  # pragma: no cover - protocol closes this set
        raise ValueError(f"unknown family: {family!r}")
    primitive = latent + sigma * bank["signal_noise"]
    return np.asarray(target, dtype=np.float64), primitive


def generate_panel(
    protocol: Mapping[str, Any],
    seed: int,
    family: str,
) -> Dict[str, Any]:
    n_train = int(protocol["rows"]["train"])
    n_test = int(protocol["rows"]["test"])
    n_rows = n_train + n_test
    bank = _latent_bank(protocol, seed, family, n_rows)
    candidates = []
    target_reference = None
    for quality_position, quality in enumerate(protocol["candidate_grid"]["quality"]):
        target, primitive = _target_and_primitive(bank, family, float(quality))
        if target_reference is None:
            target_reference = target
        elif not np.array_equal(target_reference, target):
            raise RuntimeError("candidate targets diverged within a panel")
        signal = np.column_stack(
            [
                primitive[:, 0],
                primitive[:, 1],
                0.70 * primitive[:, 0] + 0.20 * primitive[:, 1],
                0.20 * primitive[:, 0] - 0.80 * primitive[:, 1],
            ]
        )
        lifted = PolynomialFeatures(
            degree=2,
            include_bias=False,
        ).fit_transform(primitive)
        for nuisance_position, nuisance_scale in enumerate(
            protocol["candidate_grid"]["nuisance_scale"]
        ):
            observed = np.column_stack(
                [signal, float(nuisance_scale) * bank["nuisance"]]
            )
            observed = np.asarray(observed @ bank["mix"], dtype=np.float32)
            views = {
                "observed": observed,
                "oracle_signal": np.asarray(signal, dtype=np.float32),
                "oracle_sufficient": np.asarray(primitive, dtype=np.float32),
                "oracle_quadratic_lift": np.asarray(lifted, dtype=np.float32),
            }
            candidates.append(
                {
                    "candidate_id": f"Q{quality_position}N{nuisance_position}",
                    "quality": float(quality),
                    "nuisance_scale": float(nuisance_scale),
                    "views_train": {
                        key: value[:n_train]
                        for key, value in views.items()
                    },
                    "views_test": {
                        key: value[n_train:]
                        for key, value in views.items()
                    },
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


def build_selector(method_id: str, seed: int) -> ContinuousOverlapIndex:
    protocol = _load_protocol()
    spec = _method_specs(protocol)[method_id]
    return ContinuousOverlapIndex(
        model_type="FusedKMeans",
        kmeans_k=int(spec["prototypes_per_cell"]),
        fused_kmeans_kwargs={
            "random_state": int(seed),
            "n_iter": 3,
            "min_samples_per_prototype": 5,
            "relevance_weighting": True,
            "margin_rows_per_class": 32,
            "weight_floor": 0.05,
        },
        feature_normalization="l2",
        feature_transform=None,
        n_target_cells=8,
        adjacency_mode="soft_topk",
        top_k=5,
        feature_temperature=1.0,
        aggregation="support_weighted",
        null_mode="refit_permutation",
        n_null_permutations=4,
        random_state=int(seed),
    )


def evaluate_selector(
    method_id: str,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    base = build_selector(method_id, seed)
    splitter = KFold(n_splits=4, shuffle=True, random_state=int(seed))
    folds = []
    started = time.perf_counter()
    for fold, (train, holdout) in enumerate(splitter.split(X)):
        estimator = clone(base)
        fold_seed = int(seed) + int(fold)
        fused_kwargs = dict(estimator.fused_kmeans_kwargs or {})
        fused_kwargs["random_state"] = fold_seed
        estimator.set_params(
            random_state=fold_seed,
            fused_kmeans_kwargs=fused_kwargs,
        )
        estimator._fit_geometry_only(X[train], y[train])
        score = estimator.score_fixed(X[holdout], y[holdout])
        ratio = float(estimator.loss_ratio_)
        folds.append(
            {
                "fold": int(fold),
                "score": float(score),
                "actual_loss": float(estimator.actual_loss_),
                "null_loss": float(estimator.null_loss_),
                "loss_ratio": ratio if np.isfinite(ratio) else None,
                "prototype_count": int(estimator._model.n_clusters_total),
            }
        )
    ratios = [row["loss_ratio"] for row in folds]
    return {
        "score": float(np.mean([row["score"] for row in folds])),
        "mean_actual_loss": float(np.mean([row["actual_loss"] for row in folds])),
        "mean_null_loss": float(np.mean([row["null_loss"] for row in folds])),
        "mean_loss_ratio": (
            float(np.mean(ratios)) if all(value is not None for value in ratios) else None
        ),
        "folds": folds,
        "wall_seconds": float(time.perf_counter() - started),
    }


def run_panel(protocol: Mapping[str, Any], seed: int, family: str) -> Dict[str, Any]:
    panel = generate_panel(protocol, seed, family)
    specs = _method_specs(protocol)
    candidates = []
    for position, candidate in enumerate(panel["candidates"]):
        candidate_seed = int(seed) * 100 + int(position)
        heads = residual.evaluate_heads(
            candidate["views_train"]["observed"],
            panel["y_train"],
            candidate["views_test"]["observed"],
            panel["y_test"],
            candidate_seed,
        )
        selectors = {
            method_id: evaluate_selector(
                method_id,
                candidate["views_train"][spec["view"]],
                panel["y_train"],
                candidate_seed,
            )
            for method_id, spec in specs.items()
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
                "quality": candidate["quality"],
                "nuisance_scale": candidate["nuisance_scale"],
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


def _defined_spearman(left: list[float], right: list[float]) -> float | None:
    if len(set(left)) < 2 or len(set(right)) < 2:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def summarize(protocol: Mapping[str, Any], panels: list[Mapping[str, Any]]) -> Dict[str, Any]:
    normalized = deepcopy(protocol)
    method_ids = list(_method_specs(protocol))
    normalized["continuous_oi"]["methods"] = method_ids
    summary = common.summarize(normalized, panels)
    component_metrics = []
    for panel in panels:
        envelope = [
            float(row["best_head_envelope_r2"])
            for row in panel["candidates"]
        ]
        for method_id in method_ids:
            selectors = [row["selectors"][method_id] for row in panel["candidates"]]
            actual = [float(row["mean_actual_loss"]) for row in selectors]
            null = [float(row["mean_null_loss"]) for row in selectors]
            ratios = [row["mean_loss_ratio"] for row in selectors]
            component_metrics.append(
                {
                    "seed": int(panel["seed"]),
                    "family": panel["family"],
                    "method_id": method_id,
                    "negative_actual_loss_spearman": _defined_spearman(
                        [-value for value in actual], envelope
                    ),
                    "negative_loss_ratio_spearman": (
                        _defined_spearman(
                            [-float(value) for value in ratios],
                            envelope,
                        )
                        if all(value is not None for value in ratios)
                        else None
                    ),
                    "null_loss_spearman": _defined_spearman(null, envelope),
                }
            )
    summary["component_metrics"] = component_metrics
    summary["component_summary"] = {
        method_id: {
            key: common._mean_defined(
                row[key]
                for row in component_metrics
                if row["method_id"] == method_id
            )
            for key in (
                "negative_actual_loss_spearman",
                "negative_loss_ratio_spearman",
                "null_loss_spearman",
            )
        }
        for method_id in method_ids
    }
    axis_metrics = []
    for panel in panels:
        for method_id in method_ids:
            for axis_name, fixed_key in (
                ("quality_at_fixed_nuisance", "nuisance_scale"),
                ("nuisance_at_fixed_quality", "quality"),
            ):
                fixed_values = sorted(
                    {float(row[fixed_key]) for row in panel["candidates"]}
                )
                for fixed_value in fixed_values:
                    rows = [
                        row
                        for row in panel["candidates"]
                        if float(row[fixed_key]) == fixed_value
                    ]
                    envelope = [
                        float(row["best_head_envelope_r2"])
                        for row in rows
                    ]
                    scores = [
                        float(row["selectors"][method_id]["score"])
                        for row in rows
                    ]
                    selected = np.flatnonzero(
                        np.isclose(
                            scores,
                            np.max(scores),
                            atol=1.0e-12,
                            rtol=0.0,
                        )
                    )
                    regret = float(
                        np.max(envelope)
                        - np.mean(np.asarray(envelope, dtype=float)[selected])
                    )
                    axis_metrics.append(
                        {
                            "seed": int(panel["seed"]),
                            "family": panel["family"],
                            "method_id": method_id,
                            "axis": axis_name,
                            "fixed_key": fixed_key,
                            "fixed_value": fixed_value,
                            "spearman": _defined_spearman(scores, envelope),
                            "selection_regret_r2": regret,
                        }
                    )
    summary["axis_metrics"] = axis_metrics
    summary["axis_summary"] = {
        method_id: {
            axis: {
                "mean_spearman": common._mean_defined(
                    row["spearman"]
                    for row in axis_metrics
                    if row["method_id"] == method_id and row["axis"] == axis
                ),
                "mean_regret_r2": float(
                    np.mean(
                        [
                            row["selection_regret_r2"]
                            for row in axis_metrics
                            if row["method_id"] == method_id
                            and row["axis"] == axis
                        ]
                    )
                ),
            }
            for axis in (
                "quality_at_fixed_nuisance",
                "nuisance_at_fixed_quality",
            )
        }
        for method_id in method_ids
    }
    return summary


def _validate_protocol_against_code(protocol: Mapping[str, Any]) -> None:
    specs = _method_specs(protocol)
    if len(specs) != 7 or len(specs) != len(protocol["continuous_oi"]["methods"]):
        raise ValueError("frozen diagnostic must contain seven unique methods")
    expected_views = {
        "observed",
        "oracle_signal",
        "oracle_sufficient",
        "oracle_quadratic_lift",
    }
    if {row["view"] for row in specs.values()} != expected_views:
        raise ValueError("frozen oracle view set does not match code")
    if len(protocol["seeds"]) != 3 or len(protocol["families"]) != 5:
        raise ValueError("frozen panel grid must contain 3 seeds by 5 families")
    quality = protocol["candidate_grid"]["quality"]
    nuisance = protocol["candidate_grid"]["nuisance_scale"]
    if len(quality) != 4 or len(nuisance) != 4:
        raise ValueError("candidate diagnostic must be a 4 by 4 factorial")


def run(output: Path) -> Dict[str, Any]:
    protocol = _load_protocol()
    _validate_protocol_against_code(protocol)
    source_hashes = {
        "protocol": common._sha256(PROTOCOL_PATH),
        "runner": common._sha256(RUNNER_PATH),
        "common_runner": common._sha256(COMMON_RUNNER_PATH),
        "residual_runner": common._sha256(RESIDUAL_RUNNER_PATH),
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
                "method_order_by_mean_regret": summary["method_order_by_mean_regret"],
                "methods": {
                    method_id: {
                        "mean_spearman": row["mean_spearman"],
                        "mean_regret_r2": row["mean_regret_r2"],
                        **summary["component_summary"].get(method_id, {}),
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
