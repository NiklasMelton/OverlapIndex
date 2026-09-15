"""Frozen linear-versus-nonlinear continuous ranking crossover."""

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
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.continuous_matched_geometry_diagnostic_v1 import run as matched
from experiments.continuous_reference_head_adequacy_v1 import run as adequacy


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[1]
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SIDECAR = PACKAGE_DIR / "protocol.sha256"
ADEQUACY_RESULTS = ROOT / "artifacts" / "continuous_reference_head_adequacy_v1" / "results"
DEFAULT_OUTPUT = ROOT / "artifacts" / "continuous_nonlinear_crossover_v1" / "results"
CANDIDATES = ("axis_candidate", "signed_candidate", "axis_control", "signed_control")
OI_METHODS = matched.METHODS
METHODS = ("linear_cv", *OI_METHODS, "screened_head_cv_max")
ONE_THREAD = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "LOKY_MAX_CPU_COUNT": "1"}


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
        raise RuntimeError("continuous crossover protocol sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_execution":
        raise RuntimeError("continuous crossover protocol is not frozen")
    if tuple(protocol["candidates"]) != CANDIDATES or tuple(protocol["ranking_methods"]) != METHODS:
        raise RuntimeError("continuous crossover method or candidate set changed")
    return protocol


def _check_threads() -> dict[str, str]:
    observed = {name: os.environ.get(name) for name in ONE_THREAD}
    if observed != ONE_THREAD:
        raise RuntimeError(f"one-thread environment required; got {observed!r}")
    return observed


def panel_keys(protocol: Mapping[str, Any]) -> tuple[tuple[str, int, int, int], ...]:
    return tuple((scenario, int(dimension), int(rows), int(seed))
                 for scenario in protocol["scenarios"]
                 for dimension in protocol["data"]["dimensions"]
                 for rows in protocol["data"]["training_rows"]
                 for seed in protocol["data"]["training_seeds"])


def _quality_tau(quality: float) -> float:
    return float(np.sqrt((1.0 / quality ** 2) - 1.0))


def generate_panel(protocol: Mapping[str, Any], scenario: str, dimension: int,
                   train_rows: int, training_seed: int) -> dict[str, Any]:
    if scenario not in protocol["scenarios"] or (scenario, dimension, train_rows, training_seed) not in panel_keys(protocol):
        raise ValueError("requested panel is outside the frozen crossover grid")
    total = int(train_rows) + int(protocol["data"]["evaluation_rows"])
    rng = np.random.default_rng(matched._seed("continuous-crossover-v1", scenario, dimension, train_rows, training_seed))
    u = rng.normal(size=total)
    target_noise = rng.normal(size=total)
    measurement_noise = rng.normal(size=total)
    nuisance = rng.normal(size=(total, int(dimension)))
    signs = np.where(rng.integers(0, 2, size=total) == 0, -1.0, 1.0)
    y = u + float(protocol["data"]["target_noise_standard_deviation"]) * target_noise
    scenario_spec = protocol["scenarios"][scenario]
    specs = (
        ("axis_candidate", "axis", float(scenario_spec["axis_quality"])),
        ("signed_candidate", "signed_pair", float(scenario_spec["signed_quality"])),
        ("axis_control", "axis", float(protocol["controls"]["axis_quality"])),
        ("signed_control", "signed_pair", float(protocol["controls"]["signed_quality"])),
    )
    candidates = []
    for candidate_id, geometry, quality in specs:
        z = u + _quality_tau(quality) * measurement_noise
        values = nuisance.copy()
        if geometry == "axis":
            values[:, 0] = z
            recovered = values[:, 0]
        else:
            values[:, 0] = signs
            values[:, 1] = signs * z
            recovered = values[:, 0] * values[:, 1]
        if not np.allclose(recovered, z, rtol=1e-10, atol=1e-10):
            raise RuntimeError("crossover geometry failed exact recovery")
        candidates.append({"candidate_id": candidate_id, "geometry": geometry, "quality": quality,
                           "analytic_r2": float(quality ** 2 / (1.0 + protocol["data"]["target_noise_standard_deviation"] ** 2)),
                           "X_train": np.asarray(values[:train_rows], dtype=np.float64),
                           "X_evaluation": np.asarray(values[train_rows:], dtype=np.float64)})
    return {"scenario": scenario, "dimension": dimension, "train_rows": train_rows,
            "training_seed": training_seed, "expected_winner": scenario_spec["expected_winner"],
            "y_train": np.asarray(y[:train_rows]), "y_evaluation": np.asarray(y[train_rows:]),
            "candidates": candidates}


def _linear_cv(protocol: Mapping[str, Any], X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    estimator = make_pipeline(StandardScaler(), Ridge())
    inner = KFold(n_splits=3, shuffle=True, random_state=int(protocol["selectors"]["fold_seed"]))
    started = time.perf_counter()
    search = GridSearchCV(estimator, {"ridge__alpha": protocol["selectors"]["linear_ridge_alpha"]},
                          cv=inner, scoring="r2", n_jobs=1, refit=False, error_score="raise").fit(X, y)
    return {"score": float(search.best_score_), "wall_seconds": float(time.perf_counter() - started),
            "best_alpha": float(search.best_params_["ridge__alpha"])}


def evaluate_panel(protocol: Mapping[str, Any], scenario: str, dimension: int,
                   train_rows: int, training_seed: int) -> dict[str, Any]:
    panel = generate_panel(protocol, scenario, dimension, train_rows, training_seed)
    folds = list(KFold(n_splits=int(protocol["selectors"]["folds"]), shuffle=True,
                       random_state=int(protocol["selectors"]["fold_seed"])).split(np.arange(train_rows)))
    mechanism_protocol = {"selector": {"fold_seed": int(protocol["selectors"]["fold_seed"])}}
    selector_rows = []
    reference_rows = []
    for candidate in panel["candidates"]:
        head_rows = [adequacy.evaluate_head(protocol={
                "validation": {"inner_folds": 3, "inner_seed": int(protocol["selectors"]["fold_seed"])},
                "heads": {head: {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]} for head in adequacy.HEADS}},
            head=head, X_train=candidate["X_train"], y_train=panel["y_train"],
            X_evaluation=candidate["X_evaluation"], y_evaluation=panel["y_evaluation"])
            for head in adequacy.HEADS]
        reference_rows.extend({"candidate_id": candidate["candidate_id"], **row} for row in head_rows)
        linear = _linear_cv(protocol, candidate["X_train"], panel["y_train"])
        selector_rows.append({"candidate_id": candidate["candidate_id"], "method_id": "linear_cv",
                              "algorithm_seed": None, **linear})
        selector_rows.append({"candidate_id": candidate["candidate_id"], "method_id": "screened_head_cv_max",
                              "algorithm_seed": None, "score": float(max(row["inner_best_r2"] for row in head_rows)),
                              "wall_seconds": float(sum(row["wall_seconds"] for row in head_rows))})
        for algorithm_seed in protocol["selectors"]["algorithm_seeds"]:
            for method in OI_METHODS:
                result = matched.mechanism._evaluate_method(method, candidate["X_train"], panel["y_train"],
                                                            int(algorithm_seed), mechanism_protocol, folds)
                selector_rows.append({"candidate_id": candidate["candidate_id"], "method_id": method,
                                      "algorithm_seed": int(algorithm_seed), "score": float(result["score"]),
                                      "wall_seconds": float(result["wall_seconds"])})
    truths = [{key: row[key] for key in ("candidate_id", "geometry", "quality", "analytic_r2")} for row in panel["candidates"]]
    return {"schema_version": 1, "artifact_status": "completed_panel", "scenario": scenario,
            "dimension": dimension, "train_rows": train_rows, "training_seed": training_seed,
            "expected_winner": panel["expected_winner"], "truths": truths,
            "selector_rows": selector_rows, "reference_rows": reference_rows}


def summarize(protocol: Mapping[str, Any], panels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    panel_rows = []
    reference_checks = []
    for panel in panels:
        envelope = {candidate: max(float(row["evaluation_r2"]) for row in panel["reference_rows"] if row["candidate_id"] == candidate)
                    for candidate in CANDIDATES}
        expected = panel["expected_winner"]
        runner_up = max(value for name, value in envelope.items() if name != expected)
        reference_checks.append({"scenario": panel["scenario"], "dimension": panel["dimension"],
                                 "train_rows": panel["train_rows"], "training_seed": panel["training_seed"],
                                 "expected_winner": expected, "observed_winner": max(envelope, key=envelope.get),
                                 "winner_gap_r2": float(envelope[expected] - runner_up),
                                 "pass": bool(envelope[expected] == max(envelope.values()) and
                                              envelope[expected] - runner_up >= protocol["reference"]["minimum_expected_winner_gap_r2"])})
        target = np.asarray([envelope[name] for name in CANDIDATES])
        for method in METHODS:
            rows = [row for row in panel["selector_rows"] if row["method_id"] == method]
            scores = {candidate: float(np.mean([row["score"] for row in rows if row["candidate_id"] == candidate])) for candidate in CANDIDATES}
            vector = np.asarray([scores[name] for name in CANDIDATES])
            selected = np.flatnonzero(np.isclose(vector, vector.max(), rtol=0, atol=1e-12))
            rho = spearmanr(vector, target).statistic
            panel_rows.append({"scenario": panel["scenario"], "dimension": panel["dimension"],
                               "train_rows": panel["train_rows"], "training_seed": panel["training_seed"],
                               "method_id": method, "selected_candidate_ids": [CANDIDATES[int(i)] for i in selected],
                               "direction_correct": bool(CANDIDATES.index(expected) in selected),
                               "exact_best": bool(np.any(np.isclose(target[selected], target.max(), rtol=0, atol=1e-12))),
                               "regret_r2": float(target.max() - target[selected].max()),
                               "spearman": None if not np.isfinite(rho) else float(rho),
                               "mean_wall_seconds": float(np.mean([row["wall_seconds"] for row in rows]))})
    aggregates = []
    gates = protocol["gates"]
    for method in METHODS:
        rows = [row for row in panel_rows if row["method_id"] == method]
        cells = defaultdict(list)
        for row in rows:
            cells[(row["dimension"], row["train_rows"])].append(row["direction_correct"])
        metrics = {"mean_regret_r2": float(np.mean([row["regret_r2"] for row in rows])),
                   "mean_spearman": float(np.mean([row["spearman"] for row in rows if row["spearman"] is not None])),
                   "direction_accuracy": float(np.mean([row["direction_correct"] for row in rows])),
                   "exact_best_rate": float(np.mean([row["exact_best"] for row in rows])),
                   "minimum_cell_direction_accuracy": float(min(np.mean(value) for value in cells.values())),
                   "mean_wall_seconds": float(np.mean([row["mean_wall_seconds"] for row in rows]))}
        survives = (metrics["mean_regret_r2"] <= gates["maximum_mean_regret_r2"] and
                    metrics["mean_spearman"] >= gates["minimum_mean_spearman"] and
                    metrics["direction_accuracy"] >= gates["minimum_direction_accuracy"] and
                    metrics["minimum_cell_direction_accuracy"] >= gates["minimum_cell_direction_accuracy"] and
                    all(row["pass"] for row in reference_checks))
        aggregates.append({"method_id": method, **metrics, "survives": bool(survives)})
    scenario_aggregates = []
    for scenario in protocol["scenarios"]:
        for method in METHODS:
            rows = [row for row in panel_rows if row["scenario"] == scenario and row["method_id"] == method]
            scenario_aggregates.append({"scenario": scenario, "method_id": method,
                                        "mean_regret_r2": float(np.mean([row["regret_r2"] for row in rows])),
                                        "direction_accuracy": float(np.mean([row["direction_correct"] for row in rows])),
                                        "mean_spearman": float(np.mean([row["spearman"] for row in rows if row["spearman"] is not None]))})
    return {"schema_version": 1, "reference_valid": all(row["pass"] for row in reference_checks),
            "reference_checks": reference_checks, "panel_rows": panel_rows,
            "method_aggregates": aggregates, "scenario_aggregates": scenario_aggregates}


def run(output: Path = DEFAULT_OUTPUT, *, resume: bool = False) -> dict[str, Any]:
    protocol = load_protocol()
    threads = _check_threads()
    adequacy_manifest = ADEQUACY_RESULTS / "manifest.json"
    adequacy_summary = ADEQUACY_RESULTS / "summary.json"
    if _sha256(adequacy_manifest) != protocol["source_adequacy_manifest_sha256"]:
        raise RuntimeError("source adequacy manifest changed")
    if json.loads(adequacy_summary.read_text())["decision"] != "reference_adequate":
        raise RuntimeError("source reference-head adequacy gate did not pass")
    identity = {"protocol_sha256": _sha256(PROTOCOL_PATH), "runner_sha256": _sha256(Path(__file__)),
                "adequacy_manifest_sha256": _sha256(adequacy_manifest), "threads": threads}
    if (output / "manifest.json").exists():
        raise FileExistsError("terminal crossover artifact exists; refusing overwrite")
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    panels = []
    for key in panel_keys(protocol):
        panel_id = "__".join(str(value) for value in key)
        path = checkpoints / f"{hashlib.sha256(panel_id.encode()).hexdigest()[:24]}.json"
        if path.exists():
            panel = json.loads(path.read_text())
            if not resume or panel.get("identity") != identity:
                raise RuntimeError(f"stale crossover checkpoint: {path}")
        else:
            panel = evaluate_panel(protocol, *key)
            panel["identity"] = identity
            _atomic_json(path, panel)
        panels.append(panel)
        print(f"completed {panel_id}", flush=True)
    summary = summarize(protocol, panels)
    raw = {"schema_version": 1, "artifact_status": "completed", "identity": identity, "panels": panels,
           "environment": {"python": platform.python_version(), "numpy": np.__version__}}
    _atomic_json(output / "raw_results.json", raw)
    _atomic_json(output / "summary.json", summary)
    selector_count = sum(len(panel["selector_rows"]) for panel in panels)
    reference_count = sum(len(panel["reference_rows"]) for panel in panels)
    expected = protocol["execution"]
    if len(panels) != expected["expected_panels"] or selector_count != expected["expected_selector_rows"] or reference_count != expected["expected_reference_rows"]:
        raise RuntimeError("crossover artifact counts are incomplete")
    manifest = {"schema_version": 1, "artifact_status": "completed", "identity": identity,
                "panel_count": len(panels), "selector_row_count": selector_count,
                "reference_row_count": reference_count, "raw_results_sha256": _sha256(output / "raw_results.json"),
                "summary_sha256": _sha256(output / "summary.json")}
    _atomic_json(output / "manifest.json", manifest)
    return {"summary": summary, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, resume=args.resume), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

