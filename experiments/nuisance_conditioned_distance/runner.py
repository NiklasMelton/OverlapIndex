"""Research-only staged runner for nuisance-conditioned OverlapIndex.

The module has no import-time experiment side effects.  Use the CLI explicitly:

.. code-block:: console

   python3 -m experiments.nuisance_conditioned_distance.runner smoke \
       --output artifacts/nuisance_conditioned_distance/smoke

``screen`` is the development-like candidate screen and ``full`` is the
frozen Stage-2 battery.  Long-running stages are intentionally absent from the
repository's default unit-test command.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

# Enforce the frozen serial execution policy before importing NumPy, scipy,
# scikit-learn, or joblib.  LOKY_MAX_CPU_COUNT also avoids joblib's known
# physical-core detection fallback on this macOS environment.
_THREAD_ENVIRONMENT = {
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "LOKY_MAX_CPU_COUNT": "1",
}
for _thread_name, _thread_value in _THREAD_ENVIRONMENT.items():
    os.environ[_thread_name] = _thread_value

import numpy as np

from .fixtures import (
    ARCHIVED_STAGE2_CONFIG_PATH,
    ARCHIVED_STAGE2_CONFIG_SHA256,
    ARCHIVED_STAGE2_CONDITIONS,
    ARCHIVED_STAGE2_FAMILIES,
    ARCHIVED_STAGE2_GENERATOR_PATH,
    ARCHIVED_STAGE2_GENERATOR_SHA256,
    ARCHIVED_STAGE2_K_VALUES,
    ARCHIVED_STAGE2_N_CLASSES,
    ARCHIVED_STAGE2_N_PER_CLASS,
    ARCHIVED_STAGE2_NUISANCE_DIM,
    ARCHIVED_STAGE2_NUISANCE_STRENGTHS,
    ARCHIVED_STAGE2_SEEDS,
    ARCHIVED_STAGE2_SEED_ROLE,
    ARCHIVED_STAGE2_SIGNAL_DIM,
    ConfirmConfig,
    MechanisticConfig,
    SyntheticDataset,
    archived_stage2_case_grid,
    generate_confirm_dataset,
    generate_mechanistic_dataset,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
# Verified before experiment work began from the task worktree's develop
# branch.  Keep this historical value separate from the later experiment
# commit recorded in each manifest.
EXPERIMENT_STARTING_COMMIT = "165fa344653a72dd2abc04a20387d90b891b9acc"
STAGES = ("smoke", "screen", "full")
FULL_STAGE_CASE_COUNT = (
    len(ARCHIVED_STAGE2_FAMILIES)
    * len(ARCHIVED_STAGE2_CONDITIONS)
    * len(ARCHIVED_STAGE2_NUISANCE_STRENGTHS)
    * len(ARCHIVED_STAGE2_SEEDS)
    * len(ARCHIVED_STAGE2_K_VALUES)
)
BACKEND = "MiniBatchKMeans"
FIT_SPLIT = "train"
EVALUATION_SPLIT = "evaluation"
REJECTED_METHODS = (
    "margin-aware adjacency",
    "radius calibration",
    "reciprocal adjacency",
    "Wilson shrinkage",
)


@dataclass(frozen=True)
class CandidateSpec:
    """One predeclared candidate row in the experiment protocol."""

    candidate_id: str
    name: str
    mode: str | None
    prototype_refinement: bool
    covariance_scope: str | None = None
    structure: str | None = None
    covariance_estimator: str | None = None
    residual_weighting: str | None = None
    kind: str = "oi"
    diagnostic_only: bool = False


CANDIDATE_SPECS: tuple[CandidateSpec, ...] = (
    CandidateSpec("A", "oi_unrefined_raw", "none", False, structure="identity"),
    CandidateSpec("B", "oi_refined_raw", "none", True, structure="identity"),
    CandidateSpec(
        "C", "oi_refined_global_isotropy", "global_isotropy", True,
        covariance_scope="global", structure="full", covariance_estimator="oas",
    ),
    CandidateSpec(
        "D", "oi_refined_pooled_diagonal", "pooled_diagonal", True,
        covariance_scope="pooled_within_class", structure="diagonal",
        covariance_estimator="oas", residual_weighting="sample_weighted_rows",
    ),
    CandidateSpec(
        "E", "oi_refined_pooled_full", "pooled_full", True,
        covariance_scope="pooled_within_class", structure="full",
        covariance_estimator="oas", residual_weighting="sample_weighted_rows",
    ),
    CandidateSpec(
        "F",
        "raw_conditioned_disagreement",
        None,
        True,
        kind="diagnostic",
        diagnostic_only=True,
    ),
)
CANDIDATE_BY_ID = {candidate.candidate_id: candidate for candidate in CANDIDATE_SPECS}
CONDITIONED_MODES = frozenset({"none", "global_isotropy", "pooled_diagonal", "pooled_full"})
PRODUCT_POLICY_COMPARATOR = {
    "candidate_id": "G",
    "name": "panel_selective_capped_probe_guardrail",
    "kind": "product_policy_comparator",
    "oi_candidate": False,
    "trigger": {
        "any_model_b_refinement_applied_rate_gte": 0.50,
        "any_model_abs_b_minus_e_score_gte": 0.05,
    },
    "probe": {
        "kind": "five_fold_oof_logistic",
        "maximum_rows_per_model": 2048,
        "subsample": "deterministic_stratified_without_replacement",
        "standardization": "fit_within_each_training_fold",
    },
}

# These are the exact Stage-2 backend settings recovered from the archived
# runner.  ``_oi_kwargs`` deep-copies and injects a fresh per-row random_state.
_ARCHIVED_KMEANS_KWARGS: Mapping[str, Any] = {
    "batch_size": 256,
    "compute_labels": False,
    "init": "random",
    "max_no_improvement": 5,
    "n_init": 1,
}


@dataclass(frozen=True)
class RunCase:
    """One paired dataset/candidate comparison unit."""

    case_id: str
    stage: str
    family: str
    condition: str
    seed: int
    k: int
    config: ConfirmConfig | MechanisticConfig
    nuisance_strength: float
    overlap_severity: float
    balance: str
    nuisance_shift: bool
    geometry: str | None = None
    execution_offset: int = 0


def _counterbalanced_cases(cases: Sequence[RunCase]) -> tuple[RunCase, ...]:
    """Assign exact cyclic Latin positions without consulting outcomes."""

    width = 5
    return tuple(
        replace(case, execution_offset=int(position % width))
        for position, case in enumerate(cases)
    )


def _validate_stage(stage: str) -> str:
    value = str(stage).strip().lower()
    if value not in STAGES:
        raise ValueError(f"stage must be one of {STAGES!r}; got {stage!r}")
    return value


def validate_stage(stage: str) -> str:
    """Public stage validation used by tests and wrappers."""

    return _validate_stage(stage)


def _case_from_archived(
    stage: str,
    family: str,
    condition: Mapping[str, Any],
    strength: float,
    seed: int,
    k: int,
    *,
    n_per_class: int,
    n_classes: int,
    signal_dim: int,
    nuisance_dim: int,
) -> RunCase:
    condition_name = str(condition["name"])
    config = ConfirmConfig(
        family=family,
        signal_design="selective_multiclass",
        n_per_class=int(n_per_class),
        n_classes=int(n_classes),
        signal_dim=int(signal_dim),
        nuisance_dim=int(nuisance_dim),
        overlap_severity=float(condition["overlap_severity"]),
        signal_gap=float(condition["signal_gap"]),
        nuisance_strength=float(strength),
        balance=str(condition["balance"]),
        nuisance_shift=bool(condition["nuisance_shift"]),
        train_seed=int(seed),
        eval_seed=int(seed) + 10000,
        overlap_pairs=((0, 1),),
    )
    case_id = (
        f"{family}__{condition_name}__nuis-{float(strength):g}__"
        f"k-{int(k)}__seed-{int(seed)}"
    )
    return RunCase(
        case_id=case_id,
        stage=stage,
        family=family,
        condition=condition_name,
        seed=int(seed),
        k=int(k),
        config=config,
        nuisance_strength=float(strength),
        overlap_severity=float(condition["overlap_severity"]),
        balance=str(condition["balance"]),
        nuisance_shift=bool(condition["nuisance_shift"]),
    )


def _smoke_cases() -> tuple[RunCase, ...]:
    """Return a bounded mechanistic smoke matrix plus archived smoke points."""

    cases: list[RunCase] = []
    smoke_id = 0
    for geometry in ("linear", "rings", "xor"):
        for nuisance_kind in ("isotropic", "anisotropic", "correlated_low_rank"):
            config = MechanisticConfig(
                geometry=geometry,
                nuisance_kind=nuisance_kind,
                n_per_class=12,
                n_classes=2,
                signal_dim=2,
                nuisance_dim=4,
                signal_gap=1.0,
                overlap_severity=0.5 if geometry == "linear" else 0.0,
                nuisance_strength=1.0,
                imbalance=(nuisance_kind == "correlated_low_rank"),
                train_seed=100 + smoke_id,
                eval_seed=10100 + smoke_id,
                nuisance_shift=(nuisance_kind == "anisotropic"),
            )
            cases.append(
                RunCase(
                    case_id=f"mechanistic-{geometry}-{nuisance_kind}-{smoke_id}",
                    stage="smoke",
                    family="mechanistic",
                    condition=geometry,
                    seed=100 + smoke_id,
                    k=2,
                    config=config,
                    nuisance_strength=1.0,
                    overlap_severity=float(config.overlap_severity),
                    balance="imbalanced" if config.imbalance else "balanced",
                    nuisance_shift=bool(config.nuisance_shift),
                    geometry=geometry,
                )
            )
            smoke_id += 1
    # A dimensionality/n/sample/k corner and a zero-nuisance clean parity
    # point are part of smoke so baseline parity is exercised before screen.
    for index, (dim, n, k) in enumerate(((1, 8, 1), (3, 20, 2), (2, 16, 1))):
        config = MechanisticConfig(
            geometry="linear",
            nuisance_kind="isotropic",
            n_per_class=n,
            signal_dim=dim,
            nuisance_dim=2,
            signal_gap=1.5,
            overlap_severity=0.0,
            nuisance_strength=0.0,
            train_seed=200 + index,
            eval_seed=10200 + index,
        )
        cases.append(
            RunCase(
                case_id=f"mechanistic-corner-{index}",
                stage="smoke",
                family="mechanistic",
                condition="clean_linear",
                seed=200 + index,
                k=k,
                config=config,
                nuisance_strength=0.0,
                overlap_severity=0.0,
                balance="balanced",
                nuisance_shift=False,
                geometry="linear",
            )
        )
    # Include one exact archived-family point per mechanism at both clean and
    # nuisance strength, matching the archived runner's train/eval streams.
    for family in ARCHIVED_STAGE2_FAMILIES:
        for condition in ARCHIVED_STAGE2_CONDITIONS[:2]:
            for strength in (0.0, 1.0):
                cases.append(
                    _case_from_archived(
                        "smoke",
                        family,
                        condition,
                        strength,
                        1000,
                        2,
                        n_per_class=12,
                        n_classes=4,
                        signal_dim=2,
                        nuisance_dim=4,
                    )
                )
    return tuple(cases)


def stage_cases(stage: str) -> tuple[RunCase, ...]:
    """Build the exact case grid for ``smoke``, ``screen``, or ``full``."""

    stage = _validate_stage(stage)
    if stage == "smoke":
        return _counterbalanced_cases(_smoke_cases())
    if stage == "screen":
        cases: list[RunCase] = []
        # Development-like seeds are deliberately 0..9 and the grid is
        # smoke-sized (n=12) so no evaluation outcome is used to tune a mode.
        for family in ARCHIVED_STAGE2_FAMILIES:
            for condition in ARCHIVED_STAGE2_CONDITIONS[:4]:
                for strength in ARCHIVED_STAGE2_NUISANCE_STRENGTHS:
                    for seed in range(10):
                        for k in ARCHIVED_STAGE2_K_VALUES:
                            cases.append(
                                _case_from_archived(
                                    "screen",
                                    family,
                                    condition,
                                    strength,
                                    seed,
                                    k,
                                    n_per_class=12,
                                    n_classes=4,
                                    signal_dim=2,
                                    nuisance_dim=4,
                                )
                            )
        return _counterbalanced_cases(cases)
    cases = []
    for row in archived_stage2_case_grid():
        condition = {key: row[key] for key in ("name", "overlap_severity", "signal_gap", "balance", "nuisance_shift")}
        cases.append(
            _case_from_archived(
                "full",
                str(row["family"]),
                condition,
                float(row["nuisance_strength"]),
                int(row["seed"]),
                int(row["k"]),
                n_per_class=ARCHIVED_STAGE2_N_PER_CLASS,
                n_classes=ARCHIVED_STAGE2_N_CLASSES,
                signal_dim=ARCHIVED_STAGE2_SIGNAL_DIM,
                nuisance_dim=ARCHIVED_STAGE2_NUISANCE_DIM,
            )
        )
    if len(cases) != FULL_STAGE_CASE_COUNT:
        raise AssertionError("full Stage-2 grid count changed unexpectedly")
    return _counterbalanced_cases(cases)


def build_stage_cases(stage: str) -> tuple[RunCase, ...]:
    """Alias for :func:`stage_cases` used by external orchestration."""

    return stage_cases(stage)


def _oi_kwargs(k: int, seed: int) -> dict[str, Any]:
    """Create a fresh exact archived OI kwargs mapping for one row."""

    kmeans_kwargs = copy.deepcopy(dict(_ARCHIVED_KMEANS_KWARGS))
    kmeans_kwargs["random_state"] = int(seed)
    return {
        "model_type": BACKEND,
        "kmeans_k": int(k),
        "kmeans_kwargs": kmeans_kwargs,
    }


def oi_kwargs(k: int, seed: int) -> dict[str, Any]:
    """Public copy of the exact per-case backend controls."""

    return _oi_kwargs(k, seed)


def _conditioning_adapter() -> Any:
    try:
        from .conditioning_adapter import ConditionedOverlapIndex
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "ConditionedOverlapIndex is unavailable; conditioned candidates "
            "cannot run until experiments.nuisance_conditioned_distance."
            "conditioning_adapter is present"
        ) from exc
    return ConditionedOverlapIndex


def _estimator(spec: CandidateSpec, k: int, seed: int) -> Any:
    if type(spec.prototype_refinement) is not bool:
        raise TypeError("candidate prototype_refinement must be a strict bool")
    kwargs = _oi_kwargs(k, seed)
    if spec.mode == "none":
        from overlapindex import OverlapIndex

        # Do not route A/B through the adapter: exact public baseline parity is
        # an explicit experiment gate.
        return OverlapIndex(
            **copy.deepcopy(kwargs),
            prototype_refinement=spec.prototype_refinement,
        )
    if spec.mode not in CONDITIONED_MODES - {"none"}:
        raise ValueError(f"unknown conditioning mode {spec.mode!r}")
    ConditionedOverlapIndex = _conditioning_adapter()
    # This is the frozen private wrapper signature.  Keep all public OI
    # controls in one mapping and never duplicate prototype_refinement there.
    return ConditionedOverlapIndex(
        mode=spec.mode,
        prototype_refinement=spec.prototype_refinement,
        conditioning_kwargs=None,
        overlap_index_kwargs=copy.deepcopy(kwargs),
    )


def _clock() -> tuple[float, float]:
    return float(time.perf_counter()), float(time.process_time())


def _elapsed(start: tuple[float, float]) -> tuple[float, float]:
    wall, cpu = _clock()
    return wall - start[0], cpu - start[1]


def _jsonable(value: Any) -> Any:
    """Convert diagnostics into deterministic JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _snapshot_model(model: Any) -> dict[str, Any]:
    """Capture fitted state needed to prove score_fixed did not refit."""

    candidates: list[Any] = [model]
    for attr in ("overlap_index_", "estimator_", "_estimator", "model_"):
        nested = getattr(model, attr, None)
        if nested is not None:
            candidates.append(nested)
    for candidate in candidates:
        backend = getattr(candidate, "_model", None)
        centers = getattr(backend, "centers", None)
        if centers is not None:
            return {
                "centers_sha256": hashlib.sha256(np.asarray(centers).tobytes()).hexdigest(),
                "prototype_count": int(np.asarray(centers).shape[0]),
            }
    return {"centers_sha256": None, "prototype_count": None}


def _refinement_summary(model: Any) -> dict[str, Any]:
    value = getattr(model, "prototype_refinement_", None)
    return _jsonable(value) if value is not None else {}


def _pairwise_evidence(model: Any, labels: np.ndarray) -> dict[str, Any]:
    mapping = getattr(model, "pairwise_index", None)
    if mapping is None:
        nested = getattr(model, "overlap_index_", None)
        if nested is None:
            nested = getattr(model, "estimator_", None)
        mapping = getattr(nested, "pairwise_index", None) if nested is not None else None
    if mapping is None:
        return {}
    result: dict[str, Any] = {}
    classes = [int(value) if isinstance(value, (np.integer, int)) else value for value in np.unique(labels)]
    for source in classes:
        for target in classes:
            if source == target:
                continue
            try:
                value = float(mapping[(source, target)])
            except (KeyError, TypeError, ValueError):
                continue
            result[f"{source}->{target}"] = {
                "pairwise_index": value,
                "overlap_evidence": 1.0 - value if math.isfinite(value) else None,
            }
    return result


def _predict(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict through the candidate's public selector geometry.

    Conditioned candidates must transform internally before predicting.  Do
    not reach through ``estimator_``: passing raw rows to an inner OI would
    silently evaluate a different geometry than ``score_fixed``.
    """

    predict = getattr(model, "predict", None)
    if predict is None:
        raise RuntimeError(
            "candidate selector must expose predict(X) in its public selector geometry"
        )
    try:
        return np.asarray(predict(X))
    except (NotImplementedError, TypeError, ValueError) as exc:
        raise RuntimeError("candidate selector predict(X) failed in its fitted geometry") from exc


def _reference_models(dataset: SyntheticDataset, seed: int) -> tuple[dict[str, float], dict[str, float]]:
    """Fit fixed downstream references on the same train/evaluation rows."""

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.svm import SVC
    except ImportError as exc:  # pragma: no cover - sklearn is package dependency
        return {}, {"error": f"sklearn unavailable: {exc}"}
    X_train, y_train = dataset.train.X, dataset.train.y
    X_eval, y_eval = dataset.evaluation.X, dataset.evaluation.y
    models: list[tuple[str, Any]] = [
        (
            "linear_logistic",
            LogisticRegression(max_iter=500, solver="lbfgs", random_state=int(seed)),
        ),
        (
            "quadratic_logistic",
            make_pipeline(
                PolynomialFeatures(degree=2, include_bias=False),
                LogisticRegression(max_iter=500, solver="lbfgs", random_state=int(seed)),
            ),
        ),
        (
            "knn",
            KNeighborsClassifier(n_neighbors=min(5, max(1, X_train.shape[0] // 4))),
        ),
        ("rbf_svc", SVC(kernel="rbf", C=1.0, gamma="scale")),
    ]
    accuracies: dict[str, float] = {}
    timings: dict[str, float] = {}
    for name, estimator in models:
        started = time.perf_counter()
        started_cpu = time.process_time()
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_eval)
        timings[name] = float(time.perf_counter() - started)
        timings[f"{name}_cpu_seconds"] = float(time.process_time() - started_cpu)
        accuracies[name] = float(np.mean(np.asarray(predictions) == y_eval))
    return accuracies, timings


def _conditioning_diagnostics(model: Any) -> dict[str, Any]:
    value = getattr(model, "conditioning_diagnostics_", None)
    return _jsonable(value) if value is not None else {}


def _conditioning_runtime_diagnostics(model: Any) -> dict[str, Any]:
    value = getattr(model, "runtime_diagnostics_", None)
    return _jsonable(value) if value is not None else {}


def _determinism_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fitted/scoring state that must repeat bit-for-bit.

    Wall/CPU measurements, process RSS, reference timings, and execution
    position are deliberately excluded. The real first-case warm-up and the
    first measured case share the same frozen data and algorithm seeds, so
    their structural signatures form an outcome-independent determinism
    preflight without adding a row to any score or timing summary.
    """

    return _jsonable(
        {
            "candidate_id": row.get("candidate_id"),
            "status": row.get("status"),
            "fit_score": row.get("fit_score"),
            "candidate_score": row.get("candidate_score"),
            "fit_state": row.get("fit_state"),
            "score_fixed_state": row.get("score_fixed_state"),
            "pairwise": row.get("pairwise"),
            "refinement": row.get("refinement"),
            "refinement_after_score_fixed": row.get(
                "refinement_after_score_fixed"
            ),
            "conditioning": row.get("conditioning"),
            "conditioning_after_score_fixed": row.get(
                "conditioning_after_score_fixed"
            ),
            "evaluation_prediction_accuracy": row.get(
                "evaluation_prediction_accuracy"
            ),
        }
    )


def _determinism_verification(
    warmup_rows: Sequence[Mapping[str, Any]],
    measured_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare the excluded warm-up with the identical first measured case."""

    warmup = {
        str(row.get("candidate_id")): row
        for row in warmup_rows
        if row.get("candidate_id") in {"A", "B", "C", "D", "E"}
    }
    measured = {
        str(row.get("candidate_id")): row
        for row in measured_rows
        if row.get("candidate_id") in {"A", "B", "C", "D", "E"}
    }
    candidates: dict[str, Any] = {}
    for candidate_id in ("A", "B", "C", "D", "E"):
        first = warmup.get(candidate_id)
        second = measured.get(candidate_id)
        exact = (
            first is not None
            and second is not None
            and _canonical_json(_determinism_signature(first))
            == _canonical_json(_determinism_signature(second))
        )
        candidates[candidate_id] = {
            "exact": bool(exact),
            "warmup_signature_sha256": (
                _digest(_determinism_signature(first)) if first is not None else None
            ),
            "measured_signature_sha256": (
                _digest(_determinism_signature(second)) if second is not None else None
            ),
        }
    exact = all(value["exact"] for value in candidates.values())
    return {
        "status": "pass" if exact else "fail",
        "exact": exact,
        "basis": "excluded first-case warmup versus identical first measured case",
        "runtime_fields_excluded": True,
        "candidates": candidates,
    }


def _run_candidate(
    spec: CandidateSpec,
    case: RunCase,
    dataset: SyntheticDataset,
    references: Mapping[str, float],
    reference_timings: Mapping[str, float],
    execution_order: Sequence[str],
) -> tuple[dict[str, Any], Any | None]:
    """Fit one candidate on train and call score_fixed exactly once on eval."""

    row: dict[str, Any] = {
        "case_id": case.case_id,
        "stage": case.stage,
        "family": case.family,
        "condition": case.condition,
        "seed": int(case.seed),
        "k": int(case.k),
        "candidate_id": spec.candidate_id,
        "candidate_name": spec.name,
        "candidate_kind": spec.kind,
        "mode": spec.mode,
        "prototype_refinement": spec.prototype_refinement,
        "diagnostic_only": bool(spec.diagnostic_only),
        "nuisance_strength": case.nuisance_strength,
        "overlap_severity": case.overlap_severity,
        "balance": case.balance,
        "nuisance_shift": case.nuisance_shift,
        "fit_split": FIT_SPLIT,
        "evaluation_split": EVALUATION_SPLIT,
        "execution_order": list(execution_order),
        "references": dict(references),
        "reference_timings_seconds": dict(reference_timings),
        "status": "ok",
        "error": None,
        "stop_reason": None,
        "conditioning_fit_split": "train",
        "overlap_index_kwargs": _oi_kwargs(case.k, case.seed),
        "n_train": int(dataset.train.n_samples),
        "n_evaluation": int(dataset.evaluation.n_samples),
        "n_features": int(dataset.train.n_features),
        "truth": dataset.ground_truth,
        "fixture_metadata": {
            key: dataset.metadata[key]
            for key in (
                "family",
                "geometry",
                "signal_design",
                "class_counts",
                "nuisance_label_independent",
                "train_eval_independent",
                "train_seed",
                "eval_seed",
                "archived_generator_sha256",
                "archived_config_sha256",
            )
            if key in dataset.metadata
        },
    }
    if spec.diagnostic_only:
        row["candidate_score"] = None
        row["evaluation_score"] = None
        return row, None
    model: Any | None = None
    try:
        model = _estimator(spec, case.k, case.seed)
        fit_started = _clock()
        model.fit(dataset.train.X, dataset.train.y)
        fit_wall, fit_cpu = _elapsed(fit_started)
        fit_state = _snapshot_model(model)
        row["fit_wall_seconds"] = fit_wall
        row["fit_cpu_seconds"] = fit_cpu
        # ``getattr(..., default)`` would evaluate ``model.score()`` eagerly
        # and accidentally refit a fitted OI.  Prefer the fit-time ``index``
        # attribute; only wrappers that do not expose it may provide a
        # non-refitting ``fit_score_`` diagnostic.
        fit_score_value = getattr(model, "index", None)
        if fit_score_value is None:
            fit_score_value = getattr(model, "fit_score_", None)
        row["fit_score"] = float(fit_score_value) if fit_score_value is not None else None
        row["fit_state"] = fit_state
        refinement_before = _refinement_summary(model)
        conditioning_before = _conditioning_diagnostics(model)
        row["refinement"] = refinement_before
        row["conditioning"] = conditioning_before
        row["conditioning_runtime"] = _conditioning_runtime_diagnostics(model)

        score_started = _clock()
        score = float(model.score_fixed(dataset.evaluation.X, dataset.evaluation.y))
        score_wall, score_cpu = _elapsed(score_started)
        eval_state = _snapshot_model(model)
        if fit_state.get("centers_sha256") != eval_state.get("centers_sha256"):
            raise RuntimeError("score_fixed changed fitted prototypes; possible refit")
        row["score_fixed_wall_seconds"] = score_wall
        row["score_fixed_cpu_seconds"] = score_cpu
        row["candidate_score"] = score
        row["evaluation_score"] = score
        row["score_fixed_state"] = eval_state
        row["pairwise"] = _pairwise_evidence(model, dataset.evaluation.y)
        refinement_after = _refinement_summary(model)
        conditioning_after = _conditioning_diagnostics(model)
        row["refinement_after_score_fixed"] = refinement_after
        row["conditioning_after_score_fixed"] = conditioning_after
        if _canonical_json(refinement_after) != _canonical_json(refinement_before):
            raise RuntimeError("score_fixed changed prototype-refinement diagnostics")
        if _canonical_json(conditioning_after) != _canonical_json(conditioning_before):
            raise RuntimeError("score_fixed changed fitted conditioning diagnostics")
        predictions = _predict(model, dataset.evaluation.X)
        if predictions is not None and predictions.shape[0] == dataset.evaluation.y.shape[0]:
            row["evaluation_prediction_accuracy"] = float(np.mean(predictions == dataset.evaluation.y))
        row["peak_rss_bytes"] = _peak_rss_bytes()
    except Exception as exc:
        row.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "stop_reason": _stop_reason(exc),
                "candidate_score": None,
                "evaluation_score": None,
            }
        )
    return _jsonable(row), model


def _stop_reason(exc: Exception) -> str:
    text = str(exc).lower()
    if "refit" in text or "prototype" in text:
        return "baseline_mismatch_or_refit"
    if "random" in text or "determin" in text:
        return "nondeterminism"
    if "train" in text or "evaluation" in text or "leak" in text:
        return "leakage_or_split_mismatch"
    if "memory" in text or "resource" in text:
        return "runtime_or_resource"
    return "candidate_error"


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return raw if sys.platform == "darwin" else raw * 1024
    except (ImportError, OSError, AttributeError):
        return None


def _execution_order(case: RunCase) -> tuple[str, ...]:
    """Deterministic cyclic-Latin counterbalance for fit candidates A--E."""

    ids = tuple(
        candidate.candidate_id
        for candidate in CANDIDATE_SPECS
        if not candidate.diagnostic_only
    )
    offset = int(case.execution_offset) % len(ids)
    return ids[offset:] + ids[:offset]


def run_case(case: RunCase) -> list[dict[str, Any]]:
    """Run all candidates for one paired case and return raw rows."""

    if not isinstance(case, RunCase):
        raise TypeError("case must be a RunCase")
    generated = _clock()
    if isinstance(case.config, ConfirmConfig):
        dataset = generate_confirm_dataset(case.config)
    else:
        dataset = generate_mechanistic_dataset(case.config)
    generation_wall, generation_cpu = _elapsed(generated)
    references, reference_timings = _reference_models(dataset, case.seed)
    order = _execution_order(case)
    result_by_id: dict[str, tuple[dict[str, Any], Any | None]] = {}
    for candidate_id in order:
        spec = CANDIDATE_BY_ID[candidate_id]
        result_by_id[candidate_id] = _run_candidate(
            spec, case, dataset, references, reference_timings, order
        )
    # F is diagnostic-only: it uses already-fit refined raw B and conditioned E
    # models, emits no selector score, and never refits either model.
    diagnostic_spec = CANDIDATE_BY_ID["F"]
    f_row, _ = _run_candidate(
        diagnostic_spec, case, dataset, references, reference_timings, order
    )
    b_row, b_model = result_by_id.get("B", ({"status": "error"}, None))
    e_row, e_model = result_by_id.get("E", ({"status": "error"}, None))
    if b_model is not None and e_model is not None and b_row.get("status") == "ok" and e_row.get("status") == "ok":
        try:
            raw_prediction = _predict(b_model, dataset.evaluation.X)
            conditioned_prediction = _predict(e_model, dataset.evaluation.X)
            f_row["raw_candidate_id"] = "B"
            f_row["conditioned_candidate_id"] = "E"
            f_row["raw_vs_conditioned_bmu_disagreement"] = float(
                np.mean(raw_prediction != conditioned_prediction)
            )
            if b_row.get("candidate_score") is not None and e_row.get("candidate_score") is not None:
                f_row["raw_vs_conditioned_score_disagreement"] = abs(
                    float(e_row["candidate_score"]) - float(b_row["candidate_score"])
                )
            f_row["status"] = "ok"
        except Exception as exc:
            f_row["status"] = "error"
            f_row["error"] = f"{type(exc).__name__}: {exc}"
            f_row["stop_reason"] = _stop_reason(exc)
    else:
        f_row["status"] = "skipped"
        f_row["stop_reason"] = "missing_raw_or_conditioned_candidate"
    rows: list[dict[str, Any]] = []
    for candidate_id in ("A", "B", "C", "D", "E"):
        row, _ = result_by_id[candidate_id]
        row = dict(row)
        row["generation_wall_seconds"] = generation_wall
        row["generation_cpu_seconds"] = generation_cpu
        rows.append(_jsonable(row))
    f_row = dict(f_row)
    f_row["generation_wall_seconds"] = generation_wall
    f_row["generation_cpu_seconds"] = generation_cpu
    rows.append(_jsonable(f_row))
    return rows


def _environment() -> dict[str, Any]:
    package_versions: dict[str, str] = {}
    for package in ("numpy", "scikit-learn", "scipy"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "unavailable"
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_executable": sys.executable,
        "package_versions": package_versions,
        "thread_environment": dict(_THREAD_ENVIRONMENT),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_output(*args: str) -> str:
    return subprocess.check_output(list(args), cwd=ROOT, text=True).strip()


def verify_starting_commit() -> str:
    """Verify this branch still descends from the requested develop state."""

    try:
        develop_commit = _git_output("git", "rev-parse", "develop")
        merge_base = _git_output("git", "merge-base", "develop", "HEAD")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot verify the develop starting commit") from exc
    if develop_commit != EXPERIMENT_STARTING_COMMIT or merge_base != EXPERIMENT_STARTING_COMMIT:
        raise RuntimeError(
            "experiment starting commit mismatch: expected "
            f"{EXPERIMENT_STARTING_COMMIT}, develop={develop_commit}, "
            f"merge_base={merge_base}"
        )
    return merge_base


def _git_dirty_status() -> str:
    try:
        return _git_output("git", "status", "--porcelain=v1", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError):
        return ""


def _source_hashes() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for relative in (
        "experiments/nuisance_conditioned_distance/fixtures.py",
        "experiments/nuisance_conditioned_distance/runner.py",
        "experiments/nuisance_conditioned_distance/conditioning_adapter.py",
        "experiments/nuisance_conditioned_distance/protocol.json",
        "experiments/nuisance_conditioned_distance/protocol.sha256",
        "experiments/nuisance_conditioned_distance/analysis.py",
        "experiments/nuisance_conditioned_distance/reporting.py",
    ):
        path = ROOT / relative
        if path.exists():
            result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            result[relative] = None
    return result


def _code_identity_hash(source_hashes: Mapping[str, str | None]) -> str:
    """Hash only the declared experiment recipe, never generated artifacts.

    Individual source hashes are the canonical identities.  This aggregate is
    deliberately independent of output/checkpoint/cache roots so a completed
    run or resume cannot mutate its own recorded code identity.
    """

    payload = {
        "schema": "declared_experiment_sources_v1",
        "starting_commit": EXPERIMENT_STARTING_COMMIT,
        "source_hashes": dict(sorted(source_hashes.items())),
    }
    return _digest(payload)


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compute a compact candidate/stage summary without selecting a winner."""

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("stage", "")), str(row.get("candidate_name", "")))
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (stage, candidate), group in sorted(groups.items()):
        scores = np.asarray(
            [float(row["candidate_score"]) for row in group if row.get("candidate_score") is not None],
            dtype=float,
        )
        fit = np.asarray(
            [float(row["fit_wall_seconds"]) for row in group if row.get("fit_wall_seconds") is not None],
            dtype=float,
        )
        score_fixed = np.asarray(
            [float(row["score_fixed_wall_seconds"]) for row in group if row.get("score_fixed_wall_seconds") is not None],
            dtype=float,
        )
        output.append(
            {
                "stage": stage,
                "candidate_name": candidate,
                "candidate_id": CANDIDATE_BY_ID.get(str(group[0].get("candidate_id")), CandidateSpec("", "", "", False)).candidate_id,
                "n_rows": len(group),
                "n_ok": sum(row.get("status") == "ok" for row in group),
                "n_error": sum(row.get("status") == "error" for row in group),
                "score_mean": float(np.mean(scores)) if scores.size else None,
                "score_median": float(np.median(scores)) if scores.size else None,
                "fit_wall_median_seconds": float(np.median(fit)) if fit.size else None,
                "score_fixed_wall_median_seconds": float(np.median(score_fixed)) if score_fixed.size else None,
            }
        )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(_jsonable(row.get(field)), sort_keys=True)
                    if isinstance(row.get(field), (dict, list, tuple))
                    else _jsonable(row.get(field))
                    for field in fields
                }
            )


def run_stage(
    stage: str,
    output: str | Path,
    *,
    limit: int | None = None,
    warmup: bool = True,
) -> dict[str, Path]:
    """Run one explicit stage and write raw/summary/manifest artifacts."""

    stage = _validate_stage(stage)
    verified_starting_commit = verify_starting_commit()
    cases = stage_cases(stage)
    if limit is not None:
        if stage == "full":
            raise ValueError("full stage is frozen and does not accept --limit")
        if int(limit) <= 0:
            raise ValueError("limit must be positive")
        cases = cases[: int(limit)]
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    warmup_info: dict[str, Any] = {
        "enabled": bool(warmup),
        "excluded_from_stage_timing": True,
        "case_id": None,
        "wall_seconds": 0.0,
        "cpu_seconds": 0.0,
        "candidate_statuses": {},
    }
    warmup_rows: list[dict[str, Any]] = []
    if warmup and cases:
        warmup_started = _clock()
        warmup_rows = run_case(cases[0])
        warmup_wall, warmup_cpu = _elapsed(warmup_started)
        warmup_info.update(
            {
                "case_id": cases[0].case_id,
                "wall_seconds": warmup_wall,
                "cpu_seconds": warmup_cpu,
                "candidate_statuses": {
                    str(row.get("candidate_id")): str(row.get("status"))
                    for row in warmup_rows
                },
            }
        )
    stage_started = _clock()
    deviations: list[str] = []
    if not Path(ARCHIVED_STAGE2_GENERATOR_PATH).exists():
        deviations.append("archived generator path unavailable; fixture constants were embedded")
    if not Path(ARCHIVED_STAGE2_CONFIG_PATH).exists():
        deviations.append("archived config path unavailable; fixture constants were embedded")
    source_hashes = _source_hashes()
    dirty_status = _git_dirty_status()
    code_identity_sha256 = _code_identity_hash(source_hashes)
    provenance = {
        "starting_commit": verified_starting_commit,
        "experiment_commit": _git_commit(),
        "git_dirty": bool(dirty_status),
        "git_status_sha256": hashlib.sha256(dirty_status.encode("utf-8")).hexdigest(),
        "code_identity_sha256": code_identity_sha256,
        "source_hashes": source_hashes,
    }
    early_stop: dict[str, Any] | None = None
    determinism_verification: dict[str, Any] = {
        "status": "inconclusive",
        "exact": None,
        "basis": "warmup disabled or no case available",
        "runtime_fields_excluded": True,
        "candidates": {},
    }
    for case_position, case in enumerate(cases):
        case_rows = run_case(case)
        rows.extend(case_rows)
        if case_position == 0 and warmup_rows:
            determinism_verification = _determinism_verification(
                warmup_rows, case_rows
            )
            if determinism_verification["status"] != "pass":
                early_stop = {
                    "case_id": case.case_id,
                    "failures": [
                        {
                            "candidate_id": candidate_id,
                            "stop_reason": "nondeterminism",
                            "error": "excluded warmup and first measured structural signatures differ",
                        }
                        for candidate_id, value in determinism_verification[
                            "candidates"
                        ].items()
                        if not value["exact"]
                    ],
                }
                deviations.append(
                    "Stage stopped because the deterministic-repeat preflight failed."
                )
                break
        failed = [
            row
            for row in case_rows
            if row.get("candidate_id") in {"A", "B", "C", "D", "E"}
            and row.get("status") != "ok"
        ]
        if failed:
            early_stop = {
                "case_id": case.case_id,
                "failures": [
                    {
                        "candidate_id": row.get("candidate_id"),
                        "stop_reason": row.get("stop_reason"),
                        "error": row.get("error"),
                    }
                    for row in failed
                ],
            }
            deviations.append(
                "Stage stopped at the first candidate correctness/runtime error."
            )
            break
    stage_wall, stage_cpu = _elapsed(stage_started)
    raw_path = output_dir / "raw_results.json"
    csv_path = output_dir / "raw_results.csv"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"
    raw_payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "provenance": provenance,
        "rows": rows,
    }
    raw_path.write_text(
        json.dumps(_jsonable(raw_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "n_cases": len(cases),
        "n_rows": len(rows),
        "artifact_status": "partial" if early_stop is not None else "completed",
        "early_stop": early_stop,
        "stage_wall_seconds": stage_wall,
        "stage_cpu_seconds": stage_cpu,
        "warmup": warmup_info,
        "warmup_excluded_from_summaries": True,
        "determinism_verification": determinism_verification,
        "provenance": provenance,
        "candidates": summarize_rows(rows),
    }
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        **provenance,
        "environment": _environment(),
        "candidate_table": _jsonable([asdict(candidate) for candidate in CANDIDATE_SPECS]),
        "product_policy_comparator": PRODUCT_POLICY_COMPARATOR,
        "candidate_modes": sorted(CONDITIONED_MODES),
        "backend": BACKEND,
        "fit_split": FIT_SPLIT,
        "evaluation_split": EVALUATION_SPLIT,
        "stage_case_count": len(cases),
        "full_stage_case_count": FULL_STAGE_CASE_COUNT,
        "stage2_constants": {
            "families": list(ARCHIVED_STAGE2_FAMILIES),
            "conditions": _jsonable(ARCHIVED_STAGE2_CONDITIONS),
            "seeds": list(ARCHIVED_STAGE2_SEEDS),
            "nuisance_strengths": list(ARCHIVED_STAGE2_NUISANCE_STRENGTHS),
            "k_values": list(ARCHIVED_STAGE2_K_VALUES),
            "n_per_class": ARCHIVED_STAGE2_N_PER_CLASS,
            "n_classes": ARCHIVED_STAGE2_N_CLASSES,
            "signal_dim": ARCHIVED_STAGE2_SIGNAL_DIM,
            "nuisance_dim": ARCHIVED_STAGE2_NUISANCE_DIM,
            "seed_role": ARCHIVED_STAGE2_SEED_ROLE,
            "generator_path": ARCHIVED_STAGE2_GENERATOR_PATH,
            "generator_sha256": ARCHIVED_STAGE2_GENERATOR_SHA256,
            "config_path": ARCHIVED_STAGE2_CONFIG_PATH,
            "config_sha256": ARCHIVED_STAGE2_CONFIG_SHA256,
        },
        "rejected_methods_not_run": list(REJECTED_METHODS),
        "timing": "generation, fit, score_fixed, and reference stages are recorded separately; a real first-case warm-up is excluded from stage timing",
        "warmup": warmup_info,
        "determinism_verification": determinism_verification,
        "starting_commit_verified": verified_starting_commit,
        "execution_order": "deterministic counterbalanced A-E permutation per paired case, recorded in every row",
        "deviations": deviations,
        "artifact_status": "partial" if early_stop is not None else "completed",
        "early_stop": early_stop,
        "artifacts": {
            "raw_results": str(raw_path),
            "raw_results_csv": str(csv_path),
            "summary": str(summary_path),
        },
    }
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if early_stop is not None:
        raise RuntimeError(
            f"{stage} stopped early after {early_stop['case_id']}; partial artifacts: {output_dir}"
        )
    return {"raw": raw_path, "raw_csv": csv_path, "summary": summary_path, "manifest": manifest_path}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="artifact directory (defaults to artifacts/nuisance_conditioned_distance/<stage>)",
    )
    parser.add_argument("--limit", type=int, default=None, help="bounded case limit for local verification")
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip the real first-case warm-up (only for focused verification)",
    )
    args = parser.parse_args(argv)
    output = args.output or Path("artifacts") / "nuisance_conditioned_distance" / args.stage
    paths = run_stage(args.stage, output, limit=args.limit, warmup=not args.no_warmup)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2, sort_keys=True))
    return 0


# Familiar orchestration name used by the archived research runners.  It is
# intentionally an alias for the explicit staged API, so calling it still
# requires a stage and never launches the full suite implicitly.
run_experiment = run_stage


if __name__ == "__main__":  # pragma: no cover - explicit CLI only
    raise SystemExit(main())


__all__ = [
    "BACKEND",
    "CANDIDATE_SPECS",
    "CandidateSpec",
    "EXPERIMENT_STARTING_COMMIT",
    "FULL_STAGE_CASE_COUNT",
    "RunCase",
    "build_stage_cases",
    "oi_kwargs",
    "run_case",
    "run_experiment",
    "run_stage",
    "stage_cases",
    "summarize_rows",
    "validate_stage",
    "verify_starting_commit",
]
