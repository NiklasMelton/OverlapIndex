"""Frozen mean/covariance target-geometry factorial for continuous OI."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import scipy
import sklearn
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

from experiments.continuous_fisher_ranking import run as common
from experiments.continuous_oracle_diagnostic import run as oracle
from experiments.continuous_residual_ranking import run as residual
from experiments.continuous_save_ranking.geometry import fit_target_moment_transform
from overlapindex import ContinuousOverlapIndex
from overlapindex._shared_fisher import fit_shared_fisher, fit_shared_residual


PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
RUNNER_PATH = Path(__file__).resolve()
GEOMETRY_PATH = Path(__file__).with_name("geometry.py")


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
        np.random.SeedSequence([int(seed), int(position), 20260903])
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
    sigma = 0.015 + 0.55 * (1.0 - float(quality))
    if family == "linear":
        target = u
        latent = np.column_stack([u, 0.55 * u])
    elif family == "quadratic_product":
        target = u * v
        latent = np.column_stack([u, v])
    elif family == "radial_quadratic":
        target = np.square(u) + 0.65 * np.square(v)
        latent = np.column_stack([u, v])
    elif family == "absolute_value":
        target = np.abs(u)
        latent = np.column_stack([u, 0.20 * u])
    elif family == "sinusoid":
        target = np.sin(1.5 * np.pi * u)
        latent = np.column_stack([u, 0.20 * u])
    elif family == "triangle_wave":
        phase = 1.5 * (u + 1.0)
        target = 2.0 * np.abs(2.0 * (phase - np.floor(phase + 0.5))) - 1.0
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
    bank = _latent_bank(protocol, seed, family, n_train + n_test)
    local_multiplier = (
        float(protocol["candidate_grid"]["local_family_nuisance_multiplier"])
        if family in {"sinusoid", "triangle_wave"}
        else 1.0
    )
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
        for nuisance_position, nuisance_scale in enumerate(
            protocol["candidate_grid"]["nuisance_scale"]
        ):
            effective_nuisance = float(nuisance_scale) * local_multiplier
            observed = np.column_stack(
                [signal, effective_nuisance * bank["nuisance"]]
            )
            observed = np.asarray(observed @ bank["mix"], dtype=np.float32)
            candidates.append(
                {
                    "candidate_id": f"Q{quality_position}N{nuisance_position}",
                    "quality": float(quality),
                    "nuisance_scale": float(nuisance_scale),
                    "effective_nuisance_scale": effective_nuisance,
                    "X_train": observed[:n_train],
                    "X_test": observed[n_train:],
                    "primitive_train": np.asarray(primitive[:n_train], dtype=np.float32),
                    "primitive_test": np.asarray(primitive[n_train:], dtype=np.float32),
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


def _row_l2(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


def _target_cells(y: np.ndarray, n_cells: int) -> np.ndarray:
    values = np.asarray(y, dtype=float).reshape(-1)
    order = np.argsort(values, kind="stable")
    labels = np.empty(len(values), dtype=np.int64)
    labels[order] = np.minimum(
        n_cells - 1,
        np.floor(np.arange(len(values)) * n_cells / len(values)).astype(np.int64),
    )
    return labels


def _prepare_fold_geometry(
    geometry: str,
    X_train: np.ndarray,
    X_holdout: np.ndarray,
    primitive_train: np.ndarray,
    primitive_holdout: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if geometry == "oracle_quadratic_lift":
        polynomial = PolynomialFeatures(degree=2, include_bias=False)
        train = polynomial.fit_transform(primitive_train)
        holdout = polynomial.transform(primitive_holdout)
        return _row_l2(train), _row_l2(holdout), {
            "geometry": geometry,
            "output_dimension": int(train.shape[1]),
        }

    normalized_train = _row_l2(X_train)
    normalized_holdout = _row_l2(X_holdout)
    if geometry == "observed":
        return normalized_train, normalized_holdout, {
            "geometry": geometry,
            "output_dimension": int(normalized_train.shape[1]),
        }
    cells = _target_cells(y_train, 8)
    if geometry == "residual8":
        transform = fit_shared_residual(
            normalized_train,
            cells,
            rank=8,
            random_state=seed,
        )
        diagnostics = {
            "geometry": geometry,
            "output_dimension": transform.output_dimension,
            "effective_residual_rank": transform.effective_nuisance_rank,
        }
    elif geometry == "fisher_full":
        transform = fit_shared_fisher(
            normalized_train,
            cells,
            rank="full",
            random_state=seed,
        )
        diagnostics = {
            "geometry": geometry,
            "output_dimension": transform.output_dimension,
            "effective_residual_rank": transform.effective_nuisance_rank,
        }
    elif geometry in {"mean", "save", "mean_save"}:
        transform = fit_target_moment_transform(
            normalized_train,
            cells,
            mode=geometry,
            residual_rank=8,
            projection_rank=8,
            random_state=seed,
        )
        diagnostics = {
            "geometry": geometry,
            "output_dimension": transform.output_dimension,
            "effective_residual_rank": (
                transform.residual_transform.effective_nuisance_rank
            ),
            "mean_trace": transform.mean_trace,
            "save_trace": transform.save_trace,
            "moment_eigenvalues": transform.eigenvalues.tolist(),
        }
    else:  # pragma: no cover - protocol closes this set
        raise ValueError(f"unknown geometry: {geometry!r}")
    return (
        transform.transform(normalized_train),
        transform.transform(normalized_holdout),
        diagnostics,
    )


def _base_selector(seed: int) -> ContinuousOverlapIndex:
    return ContinuousOverlapIndex(
        model_type="FusedKMeans",
        kmeans_k=4,
        fused_kmeans_kwargs={
            "random_state": int(seed),
            "n_iter": 3,
            "min_samples_per_prototype": 5,
            "relevance_weighting": True,
            "margin_rows_per_class": 32,
            "weight_floor": 0.05,
        },
        feature_normalization=None,
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
    primitive: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    protocol = _load_protocol()
    geometry = _method_specs(protocol)[method_id]["geometry"]
    splitter = KFold(n_splits=4, shuffle=True, random_state=int(seed))
    folds = []
    started = time.perf_counter()
    for fold, (train, holdout) in enumerate(splitter.split(X)):
        fold_seed = int(seed) + int(fold)
        transformed_train, transformed_holdout, geometry_diagnostics = (
            _prepare_fold_geometry(
                geometry,
                X[train],
                X[holdout],
                primitive[train],
                primitive[holdout],
                y[train],
                fold_seed,
            )
        )
        estimator = clone(_base_selector(fold_seed))
        estimator._fit_geometry_only(transformed_train, y[train])
        score = estimator.score_fixed(transformed_holdout, y[holdout])
        ratio = float(estimator.loss_ratio_)
        folds.append(
            {
                "fold": int(fold),
                "score": float(score),
                "actual_loss": float(estimator.actual_loss_),
                "null_loss": float(estimator.null_loss_),
                "loss_ratio": ratio if np.isfinite(ratio) else None,
                "prototype_count": int(estimator._model.n_clusters_total),
                "geometry_diagnostics": geometry_diagnostics,
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
    methods = _method_specs(protocol)
    candidates = []
    for position, candidate in enumerate(panel["candidates"]):
        candidate_seed = int(seed) * 100 + int(position)
        heads = residual.evaluate_heads(
            candidate["X_train"],
            panel["y_train"],
            candidate["X_test"],
            panel["y_test"],
            candidate_seed,
        )
        X = candidate["X_train"]
        primitive = candidate["primitive_train"]
        y = panel["y_train"]
        selectors = {
            method_id: evaluate_selector(
                method_id,
                X,
                primitive,
                y,
                candidate_seed,
            )
            for method_id in methods
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
                "effective_nuisance_scale": candidate["effective_nuisance_scale"],
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
    summary = oracle.summarize(protocol, panels)
    floors = protocol["reference_performance_gate"]["minimum_panel_best_r2"]
    panel_checks = []
    for panel in panels:
        observed = max(
            float(row["best_head_envelope_r2"])
            for row in panel["candidates"]
        )
        minimum = float(floors[panel["family"]])
        panel_checks.append(
            {
                "seed": int(panel["seed"]),
                "family": panel["family"],
                "observed": observed,
                "minimum": minimum,
                "pass": bool(observed >= minimum),
            }
        )
    unique = {head: 0 for head in protocol["reference_heads"]["families"]}
    for panel in panels:
        for candidate in panel["candidates"]:
            if len(candidate["best_head_ids"]) == 1:
                unique[candidate["best_head_ids"][0]] += 1
    minimums = protocol["reference_performance_gate"]["minimum_unique_wins"]
    win_checks = {
        "quadratic_ridge": {
            "observed": int(unique["quadratic_ridge"]),
            "minimum": int(minimums["quadratic_ridge"]),
            "pass": bool(unique["quadratic_ridge"] >= int(minimums["quadratic_ridge"])),
        },
        "knn_or_rbf_svr": {
            "observed": int(unique["knn"] + unique["rbf_svr"]),
            "minimum": int(minimums["knn_or_rbf_svr"]),
            "pass": bool(
                unique["knn"] + unique["rbf_svr"]
                >= int(minimums["knn_or_rbf_svr"])
            ),
        },
    }
    summary["reference_performance_gate"] = {
        "status": (
            "pass"
            if all(row["pass"] for row in panel_checks)
            and all(row["pass"] for row in win_checks.values())
            else "fail"
        ),
        "panel_checks": panel_checks,
        "unique_head_wins": unique,
        "win_checks": win_checks,
    }
    return summary


def _validate_protocol_against_code(protocol: Mapping[str, Any]) -> None:
    specs = _method_specs(protocol)
    if len(specs) != 7 or len(specs) != len(protocol["continuous_oi"]["methods"]):
        raise ValueError("frozen SAVE experiment must contain seven methods")
    expected = {
        "observed",
        "residual8",
        "fisher_full",
        "mean",
        "save",
        "mean_save",
        "oracle_quadratic_lift",
    }
    if {row["geometry"] for row in specs.values()} != expected:
        raise ValueError("frozen geometry set does not match code")
    if len(protocol["seeds"]) != 3 or len(protocol["families"]) != 6:
        raise ValueError("frozen panel grid must contain 3 seeds by 6 families")
    if len(protocol["candidate_grid"]["quality"]) != 4 or len(
        protocol["candidate_grid"]["nuisance_scale"]
    ) != 4:
        raise ValueError("candidate grid must be a 4 by 4 factorial")


def run(output: Path) -> Dict[str, Any]:
    protocol = _load_protocol()
    _validate_protocol_against_code(protocol)
    source_paths = {
        "protocol": PROTOCOL_PATH,
        "runner": RUNNER_PATH,
        "geometry": GEOMETRY_PATH,
        "common_runner": common.RUNNER_PATH,
        "oracle_runner": oracle.RUNNER_PATH,
        "residual_runner": residual.RUNNER_PATH,
    }
    source_hashes = {
        key: common._sha256(path)
        for key, path in source_paths.items()
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
                "reference_performance_gate": summary["reference_performance_gate"],
                "methods": {
                    method_id: {
                        "mean_spearman": row["mean_spearman"],
                        "mean_regret_r2": row["mean_regret_r2"],
                        "quality_at_fixed_nuisance": summary["axis_summary"].get(
                            method_id, {}
                        ).get("quality_at_fixed_nuisance"),
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
