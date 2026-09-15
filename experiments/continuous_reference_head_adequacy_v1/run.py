"""Frozen calibration of sparse reference heads on known continuous signals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

from experiments.continuous_matched_geometry_diagnostic_v1 import run as source


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[1]
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SIDECAR = PACKAGE_DIR / "protocol.sha256"
SOURCE_RESULTS = ROOT / "artifacts" / "continuous_matched_geometry_diagnostic_v1" / "results"
DEFAULT_OUTPUT = ROOT / "artifacts" / "continuous_reference_head_adequacy_v1" / "results"
HEADS = ("screened_linear_ridge", "screened_pair_ridge")


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
        raise RuntimeError("reference-head adequacy protocol sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_execution" or tuple(protocol["heads"]) != HEADS:
        raise RuntimeError("reference-head adequacy protocol changed")
    return protocol


class ScreenedLinearRidge(RegressorMixin, BaseEstimator):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: Any, y: Any) -> "ScreenedLinearRidge":
        matrix = np.asarray(X, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        self.scaler_ = StandardScaler().fit(matrix)
        standardized = self.scaler_.transform(matrix)
        centered = target - np.mean(target)
        covariance = standardized.T @ centered / len(standardized)
        self.selected_feature_ = int(np.argmax(np.abs(covariance)))
        self.model_ = Ridge(alpha=float(self.alpha)).fit(standardized[:, [self.selected_feature_]], target)
        self.n_features_in_ = matrix.shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        standardized = self.scaler_.transform(np.asarray(X, dtype=np.float64))
        return self.model_.predict(standardized[:, [self.selected_feature_]])


class ScreenedPairRidge(RegressorMixin, BaseEstimator):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: Any, y: Any) -> "ScreenedPairRidge":
        matrix = np.asarray(X, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        self.scaler_ = StandardScaler().fit(matrix)
        standardized = self.scaler_.transform(matrix)
        centered = target - np.mean(target)
        covariance = (standardized * centered[:, None]).T @ standardized / len(standardized)
        covariance[np.tril_indices_from(covariance)] = 0.0
        self.selected_pair_ = tuple(int(value) for value in np.unravel_index(np.argmax(np.abs(covariance)), covariance.shape))
        features = self._features(standardized)
        self.model_ = Ridge(alpha=float(self.alpha)).fit(features, target)
        self.n_features_in_ = matrix.shape[1]
        return self

    def _features(self, standardized: np.ndarray) -> np.ndarray:
        left, right = self.selected_pair_
        return (standardized[:, left] * standardized[:, right])[:, None]

    def predict(self, X: Any) -> np.ndarray:
        standardized = self.scaler_.transform(np.asarray(X, dtype=np.float64))
        return self.model_.predict(self._features(standardized))


def evaluate_head(protocol: Mapping[str, Any], head: str, X_train: np.ndarray, y_train: np.ndarray,
                  X_evaluation: np.ndarray, y_evaluation: np.ndarray) -> dict[str, Any]:
    if head == "screened_linear_ridge":
        estimator: Any = ScreenedLinearRidge()
    elif head == "screened_pair_ridge":
        estimator = ScreenedPairRidge()
    else:
        raise ValueError(f"head must be one of {HEADS!r}; got {head!r}")
    inner = KFold(n_splits=int(protocol["validation"]["inner_folds"]), shuffle=True,
                  random_state=int(protocol["validation"]["inner_seed"]))
    started = time.perf_counter()
    search = GridSearchCV(estimator, {"alpha": protocol["heads"][head]["alpha"]},
                          scoring="r2", cv=inner, n_jobs=1, refit=True, error_score="raise")
    search.fit(X_train, y_train)
    prediction = search.predict(X_evaluation)
    output = {"head": head, "inner_best_r2": float(search.best_score_),
              "evaluation_r2": float(r2_score(y_evaluation, prediction)),
              "best_alpha": float(search.best_params_["alpha"]),
              "wall_seconds": float(time.perf_counter() - started)}
    fitted = search.best_estimator_
    if head == "screened_linear_ridge":
        output["selected_feature"] = int(fitted.selected_feature_)
    else:
        output["selected_pair"] = list(fitted.selected_pair_)
    return output


def summarize(protocol: Mapping[str, Any], source_raw: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    truth = {}
    old_envelope = {}
    selectors = {}
    for panel in source_raw["panels"]:
        if not panel["reference_rows"]:
            continue
        key = (int(panel["dimension"]), int(panel["train_rows"]))
        truth[key] = {row["candidate_id"]: float(row["analytic_r2"]) for row in panel["truths"]}
        old_envelope[key] = {row["candidate_id"]: float(row["envelope_r2"]) for row in panel["reference_rows"]}
        selectors[key] = panel["selector_rows"]
    new_lookup = {(int(row["dimension"]), int(row["train_rows"]), row["candidate_id"], row["head"]): row for row in rows}
    required_checks = []
    for required in protocol["gates"]["required_cells"]:
        for key in sorted(truth):
            row = new_lookup[(*key, required["candidate_id"], required["head"])]
            gap = abs(float(row["evaluation_r2"]) - truth[key][required["candidate_id"]])
            known = row.get("selected_feature") == 0 if required["head"] == "screened_linear_ridge" else set(row.get("selected_pair", ())) == {0, 1}
            required_checks.append({"dimension": key[0], "train_rows": key[1], **required,
                                    "absolute_r2_gap": gap, "known_signal_selected": bool(known),
                                    "pass": bool(gap <= protocol["gates"]["maximum_absolute_r2_gap_to_analytic"] and known)})
    calibrated = {}
    for key in sorted(truth):
        calibrated[key] = {}
        for candidate in source.CANDIDATES:
            additions = [float(new_lookup[(*key, candidate, head)]["evaluation_r2"]) for head in HEADS]
            calibrated[key][candidate] = max(old_envelope[key][candidate], *additions)
    ranking_rows = []
    for key in sorted(calibrated):
        for method in source.METHODS:
            method_rows = [row for row in selectors[key] if row["method_id"] == method]
            scores = {candidate: float(np.mean([row["score"] for row in method_rows if row["candidate_id"] == candidate])) for candidate in source.CANDIDATES}
            score_values = np.asarray([scores[name] for name in source.CANDIDATES])
            target_values = np.asarray([calibrated[key][name] for name in source.CANDIDATES])
            selected = np.flatnonzero(np.isclose(score_values, score_values.max(), rtol=0, atol=1e-12))
            rho = spearmanr(score_values, target_values).statistic
            ranking_rows.append({"dimension": key[0], "train_rows": key[1], "method_id": method,
                                 "regret_r2": float(target_values.max() - target_values[selected].max()),
                                 "spearman": None if not np.isfinite(rho) else float(rho)})
    gate_pass = all(row["pass"] for row in required_checks)
    return {"schema_version": 1, "required_checks": required_checks, "reference_adequate": gate_pass,
            "decision": "reference_adequate" if gate_pass else "reference_inadequate",
            "calibrated_envelopes": [{"dimension": key[0], "train_rows": key[1], "envelope": value} for key, value in calibrated.items()],
            "selector_ranking_rows": ranking_rows}


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    protocol = load_protocol()
    source_raw_path = SOURCE_RESULTS / "raw_results.json"
    if _sha256(source_raw_path) != protocol["source_raw_sha256"] or _sha256(source.PROTOCOL_PATH) != protocol["source_protocol_sha256"]:
        raise RuntimeError("source matched-geometry artifact changed")
    if (output / "manifest.json").exists():
        raise FileExistsError("terminal head-adequacy artifact exists; refusing overwrite")
    source_raw = json.loads(source_raw_path.read_text(encoding="utf-8"))
    source_protocol = source.load_protocol()
    rows = []
    for dimension in source_protocol["data"]["dimensions"]:
        for train_rows in source_protocol["data"]["training_rows"]:
            panel = source.generate_panel(source_protocol, dimension, train_rows, protocol["reference_training_seed"])
            for candidate in panel["candidates"]:
                for head in HEADS:
                    rows.append({"dimension": int(dimension), "train_rows": int(train_rows),
                                 "candidate_id": candidate["candidate_id"],
                                 **evaluate_head(protocol, head, candidate["X_train"], panel["y_train"],
                                                 candidate["X_evaluation"], panel["y_evaluation"])})
    expected = int(protocol["artifact_contract"]["expected_head_rows"])
    if len(rows) != expected:
        raise RuntimeError("head-adequacy row count is incomplete")
    summary = summarize(protocol, source_raw, rows)
    identity = {"protocol_sha256": _sha256(PROTOCOL_PATH), "runner_sha256": _sha256(Path(__file__)),
                "source_raw_sha256": _sha256(source_raw_path)}
    raw = {"schema_version": 1, "artifact_status": "completed", "identity": identity, "head_rows": rows,
           "environment": {"python": platform.python_version(), "numpy": np.__version__}}
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "raw_results.json", raw)
    _atomic_json(output / "summary.json", summary)
    manifest = {"schema_version": 1, "artifact_status": "completed", "identity": identity,
                "head_row_count": len(rows), "raw_results_sha256": _sha256(output / "raw_results.json"),
                "summary_sha256": _sha256(output / "summary.json")}
    _atomic_json(output / "manifest.json", manifest)
    return {"summary": summary, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

