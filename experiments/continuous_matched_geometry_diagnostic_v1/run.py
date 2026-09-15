"""Frozen matched-predictive-utility geometry diagnostic."""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from experiments.continuous_mechanism_diagnostic_v1 import run as mechanism


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[1]
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SIDECAR = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT = ROOT / "artifacts" / "continuous_matched_geometry_diagnostic_v1" / "results"
METHODS = mechanism.METHOD_IDS
CANDIDATES = ("direct_q85", "rotated_q85", "anisotropic_q85", "signed_q85", "signed_q55")
MATCHED = CANDIDATES[:4]
HEADS = ("linear_ridge", "quadratic_svr", "knn", "rbf_svr")


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
        raise ValueError("matched-geometry protocol sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_exploratory_before_outcomes":
        raise ValueError("matched-geometry protocol is not frozen")
    if tuple(row["candidate_id"] for row in protocol["data"]["candidates"]) != CANDIDATES:
        raise ValueError("matched-geometry candidate set changed")
    if tuple(protocol["selectors"]["methods"]) != METHODS:
        raise ValueError("matched-geometry selector set changed")
    return protocol


def _seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return 1 + int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 4_000_000_000


def _quality_tau(quality: float) -> float:
    return float(np.sqrt((1.0 / float(quality) ** 2) - 1.0))


@lru_cache(maxsize=None)
def _rotation(dimension: int) -> np.ndarray:
    rotation_rng = np.random.default_rng(_seed("rotation", int(dimension)))
    raw = rotation_rng.normal(size=(int(dimension), int(dimension)))
    rotation, _ = np.linalg.qr(raw)
    rotation.setflags(write=False)
    return rotation


def generate_panel(protocol: Mapping[str, Any], dimension: int, train_rows: int,
                   training_seed: int) -> dict[str, Any]:
    if int(dimension) not in protocol["data"]["dimensions"] or int(train_rows) not in protocol["data"]["training_rows"]:
        raise ValueError("dimension/sample-size cell is outside the frozen grid")
    total = int(train_rows) + int(protocol["data"]["evaluation_rows"])
    rng = np.random.default_rng(_seed("rows", dimension, train_rows, training_seed))
    u = rng.normal(size=total)
    target_noise = rng.normal(size=total)
    measurement_noise = rng.normal(size=total)
    nuisance = rng.normal(size=(total, int(dimension)))
    signs = np.where(rng.integers(0, 2, size=total) == 0, -1.0, 1.0)
    rotation = _rotation(int(dimension))
    y = u + float(protocol["data"]["target_noise_standard_deviation"]) * target_noise
    specs = {row["candidate_id"]: row for row in protocol["data"]["candidates"]}
    candidates = []
    for candidate_id in CANDIDATES:
        spec = specs[candidate_id]
        quality = float(spec["quality"])
        z = u + _quality_tau(quality) * measurement_noise
        base = nuisance.copy()
        base[:, 0] = z
        geometry = spec["geometry"]
        if geometry == "direct":
            values = base
            recovered_z = values[:, 0]
        elif geometry == "rotated":
            values = base @ rotation
            recovered_z = (values @ rotation.T)[:, 0]
        elif geometry == "anisotropic":
            values = base.copy()
            values[:, 1:] *= float(protocol["data"]["anisotropic_nuisance_scale"])
            recovered_z = values[:, 0]
        elif geometry == "signed_product":
            values = nuisance.copy()
            values[:, 0] = signs
            values[:, 1] = signs * z
            recovered_z = values[:, 0] * values[:, 1]
        else:
            raise ValueError(f"unknown geometry {geometry!r}")
        if not np.allclose(recovered_z, z, rtol=1e-10, atol=1e-10):
            raise RuntimeError("geometry failed exact information recovery")
        noise_var = float(protocol["data"]["target_noise_standard_deviation"]) ** 2
        analytic = quality * quality / (1.0 + noise_var)
        conditional = quality * quality * recovered_z
        candidates.append({"candidate_id": candidate_id, "geometry": geometry, "quality": quality,
                           "analytic_r2": float(analytic),
                           "evaluation_conditional_mean_r2": float(r2_score(y[train_rows:], conditional[train_rows:])),
                           "X_train": np.asarray(values[:train_rows], dtype=np.float64),
                           "X_evaluation": np.asarray(values[train_rows:], dtype=np.float64)})
    return {"dimension": int(dimension), "train_rows": int(train_rows), "training_seed": int(training_seed),
            "y_train": np.asarray(y[:train_rows], dtype=np.float64),
            "y_evaluation": np.asarray(y[train_rows:], dtype=np.float64), "candidates": candidates}


def _head_estimators(protocol: Mapping[str, Any]) -> dict[str, tuple[Any, dict[str, Sequence[Any]]]]:
    config = protocol["reference_heads"]["families"]
    return {
        "linear_ridge": (Pipeline([("scale", StandardScaler()), ("model", Ridge())]),
                         {"model__alpha": config["linear_ridge"]["alpha"]}),
        "quadratic_svr": (Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="poly", degree=2,
            gamma="scale", coef0=1.0, epsilon=0.1))]), {"model__C": config["quadratic_svr"]["C"]}),
        "knn": (Pipeline([("scale", StandardScaler()), ("model", KNeighborsRegressor(weights="distance", n_jobs=1))]),
                {"model__n_neighbors": config["knn"]["n_neighbors"]}),
        "rbf_svr": (Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf", gamma="scale", epsilon=0.1))]),
                    {"model__C": config["rbf_svr"]["C"]}),
    }


def evaluate_heads(protocol: Mapping[str, Any], candidate: Mapping[str, Any],
                   y_train: np.ndarray, y_evaluation: np.ndarray, *, seed: int) -> dict[str, Any]:
    inner = KFold(n_splits=3, shuffle=True, random_state=int(seed))
    rows = {}
    for head, (estimator, grid) in _head_estimators(protocol).items():
        started = time.perf_counter()
        search = GridSearchCV(clone(estimator), grid, cv=inner, scoring="r2", n_jobs=1,
                              refit=True, error_score="raise")
        search.fit(candidate["X_train"], y_train)
        prediction = search.predict(candidate["X_evaluation"])
        rows[head] = {"evaluation_r2": float(r2_score(y_evaluation, prediction)),
                      "inner_best_r2": float(search.best_score_), "best_params": dict(search.best_params_),
                      "wall_seconds": float(time.perf_counter() - started)}
    return rows


def evaluate_panel(protocol: Mapping[str, Any], dimension: int, train_rows: int,
                   training_seed: int) -> dict[str, Any]:
    panel = generate_panel(protocol, dimension, train_rows, training_seed)
    folds = list(KFold(n_splits=3, shuffle=True, random_state=int(protocol["selectors"]["fold_seed"]))
                 .split(np.arange(train_rows)))
    selector_rows = []
    mechanism_protocol = {"selector": {"fold_seed": int(protocol["selectors"]["fold_seed"])}}
    for candidate in panel["candidates"]:
        for algorithm_seed in protocol["selectors"]["algorithm_seeds"]:
            for method in METHODS:
                result = mechanism._evaluate_method(method, candidate["X_train"], panel["y_train"],
                                                    int(algorithm_seed), mechanism_protocol, folds)
                selector_rows.append({"candidate_id": candidate["candidate_id"], "method_id": method,
                                      "algorithm_seed": int(algorithm_seed), "score": float(result["score"]),
                                      "wall_seconds": float(result["wall_seconds"]), "folds": result["folds"]})
    references = []
    if training_seed == int(protocol["reference_heads"]["evaluation_training_seed"]):
        for position, candidate in enumerate(panel["candidates"]):
            heads = evaluate_heads(protocol, candidate, panel["y_train"], panel["y_evaluation"],
                                   seed=_seed("heads", dimension, train_rows, position))
            references.append({"candidate_id": candidate["candidate_id"], "heads": heads,
                               "envelope_r2": float(max(row["evaluation_r2"] for row in heads.values())),
                               "winning_heads": [head for head, row in heads.items() if np.isclose(row["evaluation_r2"],
                                   max(item["evaluation_r2"] for item in heads.values()), rtol=0, atol=1e-12)]})
    truths = [{key: candidate[key] for key in ("candidate_id", "geometry", "quality", "analytic_r2",
               "evaluation_conditional_mean_r2")} for candidate in panel["candidates"]]
    return {"dimension": dimension, "train_rows": train_rows, "training_seed": training_seed,
            "truths": truths, "selector_rows": selector_rows, "reference_rows": references}


def _pair_accuracy(scores: Mapping[str, float]) -> float:
    weak = scores["signed_q55"]
    return float(np.mean([1.0 if scores[name] > weak else 0.5 if scores[name] == weak else 0.0 for name in MATCHED]))


def summarize(protocol: Mapping[str, Any], panels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    panel_metrics = []
    for panel in panels:
        truths = {row["candidate_id"]: float(row["analytic_r2"]) for row in panel["truths"]}
        for method in METHODS:
            method_rows = [row for row in panel["selector_rows"] if row["method_id"] == method]
            scores = {candidate: float(np.mean([row["score"] for row in method_rows if row["candidate_id"] == candidate]))
                      for candidate in CANDIDATES}
            selected = [name for name in CANDIDATES if np.isclose(scores[name], max(scores.values()), rtol=0, atol=1e-12)]
            panel_metrics.append({"dimension": panel["dimension"], "train_rows": panel["train_rows"],
                                  "training_seed": panel["training_seed"], "method_id": method,
                                  "matched_score_range": float(max(scores[name] for name in MATCHED) - min(scores[name] for name in MATCHED)),
                                  "material_pair_accuracy": _pair_accuracy(scores),
                                  "selected_candidate_ids": selected,
                                  "weak_candidate_selected": "signed_q55" in selected,
                                  "analytic_selection_regret_r2": float(max(truths.values()) - max(truths[name] for name in selected)),
                                  "algorithm_score_sd_mean": float(np.mean([np.std([row["score"] for row in method_rows if row["candidate_id"] == candidate]) for candidate in CANDIDATES]))})
    method_summary = []
    thresholds = protocol["diagnostic_thresholds"]
    for method in METHODS:
        rows = [row for row in panel_metrics if row["method_id"] == method]
        biased = np.mean([row["matched_score_range"] > thresholds["material_score_range"] for row in rows])
        weak_rate = np.mean([row["weak_candidate_selected"] for row in rows])
        pair_accuracy = np.mean([row["material_pair_accuracy"] for row in rows])
        method_summary.append({"method_id": method, "mean_matched_score_range": float(np.mean([row["matched_score_range"] for row in rows])),
                               "geometry_bias_panel_fraction": float(biased), "weak_candidate_selection_rate": float(weak_rate),
                               "mean_material_pair_accuracy": float(pair_accuracy),
                               "mean_analytic_selection_regret_r2": float(np.mean([row["analytic_selection_regret_r2"] for row in rows])),
                               "score_geometry_bias": bool(biased >= thresholds["geometry_bias_panel_fraction"]),
                               "harmful_geometry_failure": bool(weak_rate > thresholds["maximum_acceptable_weak_candidate_selection_rate"] or pair_accuracy < thresholds["minimum_material_pair_accuracy"])})
    reference_metrics = []
    for panel in panels:
        if not panel["reference_rows"]:
            continue
        envelope = {row["candidate_id"]: float(row["envelope_r2"]) for row in panel["reference_rows"]}
        reference_metrics.append({"dimension": panel["dimension"], "train_rows": panel["train_rows"],
                                  "training_seed": panel["training_seed"],
                                  "matched_envelope_spread_r2": float(max(envelope[name] for name in MATCHED) - min(envelope[name] for name in MATCHED)),
                                  "finite_head_accessibility_difference": bool(max(envelope[name] for name in MATCHED) - min(envelope[name] for name in MATCHED) > thresholds["material_head_envelope_spread_r2"]),
                                  "envelope": envelope,
                                  "winning_heads": {row["candidate_id"]: row["winning_heads"] for row in panel["reference_rows"]}})
    return {"schema_version": 1, "panel_metrics": panel_metrics, "method_summary": method_summary,
            "reference_metrics": reference_metrics}


def run(output: Path = DEFAULT_OUTPUT, *, resume: bool = False) -> dict[str, Any]:
    protocol = load_protocol()
    source_paths = {"protocol": PROTOCOL_PATH, "runner": Path(__file__), "mechanism_runner": Path(mechanism.__file__),
                    "continuous_overlap_index": ROOT / "overlapindex" / "ContinuousOverlapIndex.py"}
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    output = output.resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError("terminal matched-geometry diagnostic exists")
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    panels = []
    for dimension in protocol["data"]["dimensions"]:
        for train_rows in protocol["data"]["training_rows"]:
            for training_seed in protocol["data"]["training_sample_seeds"]:
                path = checkpoints / f"d{dimension}__n{train_rows}__s{training_seed}.json"
                if path.exists():
                    checkpoint = json.loads(path.read_text(encoding="utf-8"))
                    if not resume or checkpoint.get("source_hashes") != source_hashes:
                        raise ValueError(f"stale matched-geometry checkpoint: {path}")
                    panel = checkpoint["panel"]
                else:
                    panel = evaluate_panel(protocol, int(dimension), int(train_rows), int(training_seed))
                    _atomic_json(path, {"source_hashes": source_hashes, "panel": panel})
                panels.append(panel)
                print(f"completed d={dimension} n={train_rows} seed={training_seed}", flush=True)
    summary = summarize(protocol, panels)
    raw = {"schema_version": 1, "artifact_status": "completed", "source_hashes": source_hashes,
           "protocol": protocol, "panels": panels, "environment": {"python": platform.python_version(), "numpy": np.__version__}}
    _atomic_json(output / "raw_results.json", raw)
    _atomic_json(output / "summary.json", summary)
    manifest = {"schema_version": 1, "artifact_status": "completed", "source_hashes": source_hashes,
                "panel_count": len(panels), "selector_row_count": sum(len(row["selector_rows"]) for row in panels),
                "reference_candidate_row_count": sum(len(row["reference_rows"]) for row in panels),
                "raw_results_sha256": _sha256(output / "raw_results.json"), "summary_sha256": _sha256(output / "summary.json")}
    contract = protocol["artifact_contract"]
    if (manifest["panel_count"], manifest["selector_row_count"], manifest["reference_candidate_row_count"]) != (
        contract["panel_count"], contract["total_selector_rows"], contract["total_reference_candidate_rows"]):
        raise RuntimeError("matched-geometry artifact counts are incomplete")
    _atomic_json(output / "manifest.json", manifest)
    return {"summary": summary, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, resume=args.resume)["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
