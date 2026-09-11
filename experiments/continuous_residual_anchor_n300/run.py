"""Frozen n=300 raw-linear-anchor plus nonlinear-residual diagnostic.

The implementation is experiment-local.  It reuses the already validated
information-ladder source loader and geometry helpers, but does not alter any
public OverlapIndex/Vertebrae code.  The important leakage boundary is the
outer fold: the base Ridge prediction for every outer-training row is
out-of-fold, and only those residuals are exposed to target-aware geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from experiments.continuous_information_ladder_n300 import run as ladder
from experiments.continuous_stable_prototype_readout_n300 import run as stable
from experiments.continuous_structured_head_discovery import run as geometry


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_residual_anchor_n300"

OUTER_LINEAR = "outer_linear_ridge"
RESIDUAL_SAVE = "linear_plus_save_residual"
RESIDUAL_RFF = "linear_plus_rff_mean_residual"
RESIDUAL_EQUAL = "linear_plus_save_rff_equal_residual"
RESIDUAL_SQUARED = "linear_plus_prototype_squared_distance_residual"
RESIDUAL_ILR = "linear_plus_prototype_orthogonal_log_contrast_residual"
METHODS = (OUTER_LINEAR, RESIDUAL_SAVE, RESIDUAL_RFF, RESIDUAL_EQUAL, RESIDUAL_SQUARED, RESIDUAL_ILR)
RESIDUAL_METHODS = METHODS[1:]
GEOMETRY_FEATURES = {
    RESIDUAL_SAVE: ladder.SAVE_ONLY,
    RESIDUAL_RFF: ladder.RFF_MEAN_ONLY,
    RESIDUAL_EQUAL: ladder.SAVE_RFF,
    RESIDUAL_SQUARED: ladder.PROTO_SQUARED_DISTANCE,
    RESIDUAL_ILR: ladder.PROTO_ILR,
}
TARGETS = tuple(ladder.TARGETS)
BACKBONES = tuple(ladder.BACKBONES)
STRATA = dict(ladder.STRATA)
N300_BUDGET_ID = ladder.N300_BUDGET_ID
N300_TRAIN_ROWS = ladder.N300_TRAIN_ROWS
N300_EVALUATION_ROWS = ladder.N300_EVALUATION_ROWS
OUTER_FOLDS = ladder.OUTER_FOLDS
OUTER_TRAIN_ROWS = ladder.OUTER_TRAIN_ROWS
OUTER_HOLDOUT_ROWS = ladder.OUTER_HOLDOUT_ROWS
OUTER_SEED_BASE = ladder.OUTER_SEED_BASE
INNER_FOLDS = 3
# Seed offsets are named by the stage that consumes them.  The outer anchor
# deliberately follows the parent information ladder's inner-CV seed, while
# residual OOF and nested calibration stages retain this experiment's frozen
# offsets.  Keeping these separate prevents a calibration offset from silently
# changing the production anchor.
OUTER_ANCHOR_CV_SEED_OFFSET = int(stable.INNER_SEED_OFFSET)
RESIDUAL_OOF_SEED_OFFSET = 1103
CALIBRATION_SELECTION_SPLIT_SEED_OFFSET = 1103
CALIBRATION_OOF_SEED_OFFSET = 701
CALIBRATION_ANCHOR_SEED_OFFSET = 907
CALIBRATION_GEOMETRY_SEED_OFFSET = 1301
FINAL_GEOMETRY_SEED_OFFSET = 1301
TARGET_PERMUTATION_GEOMETRY_SEED_OFFSET = 1601
ORIGINAL_Y_GEOMETRY_SEED_OFFSET = 1701
# Compatibility alias for callers of the pre-freeze helper; new code should
# use the stage-specific names above.
INNER_SEED_OFFSET = RESIDUAL_OOF_SEED_OFFSET
SHRINKAGE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
FIXED_ALPHA = 1.0
TIE_ATOL = ladder.TIE_ATOL
MATERIAL_PAIR_DIFFERENCE_R2 = ladder.MATERIAL_PAIR_DIFFERENCE_R2
CONTROL_NULL_SEMANTICS = "conditional_association_null_after_residual_geometry"
TARGET_NULL_SEMANTICS = "target_permutation_residual_geometry_null"
NEGATIVE_PERMUTATION_VERSION = "sha256_sort_residual_v1"
TARGET_PERMUTATION_VERSION = "sha256_sort_residual_target_geometry_v1"
THREAD_KEYS = ladder.THREAD_KEYS
RUNTIME_CONTRACT = dict(ladder.RUNTIME_CONTRACT)
GROUPS = tuple(ladder.GROUPS)

RAW_KEYS = {
    "schema_version", "status", "protocol_sha256", "source_hashes", "environment",
    "checkpoint_identity", "selected_targets", "reserved_backbones_evaluated", "panels",
}
SUMMARY_KEYS = {
    "schema_version", "status", "selected_targets", "panel_count", "candidate_count",
    "ranking_rows", "ranking_summaries", "target_ranking_summaries", "runtime_summary",
    "diagnosis", "truth_is_independent", "reserved_backbones_evaluated", "method_promotion",
}
MANIFEST_KEYS = {
    "schema_version", "status", "decision_status", "protocol_sha256", "source_hashes",
    "checkpoint_identity", "artifacts", "counts", "environment", "method_promotion",
}
PANEL_KEYS = {
    "dataset_id", "target_id", "target", "stratum", "budget_id", "outer_seed",
    "training_rows", "evaluation_rows", "training_sample_ids", "evaluation_sample_ids",
    "candidate_count", "candidates",
}
CANDIDATE_KEYS = {
    "candidate_id", "truth", "archived_source_linear_cv_r2", "archived_source_linear_runtime_seconds",
    "source_pca", "scores", "runtime_seconds", "pipeline_runtime_seconds", "folds",
}
FOLD_KEYS = {
    "fold", "fit_seed", "pca_seed", "train_row_indices", "holdout_row_indices",
    "train_sample_ids", "holdout_sample_ids", "base_oof_prediction_sha256",
    "residual_target_sha256", "selected_shrinkage", "shrinkage_selection",
    "readouts", "permutation_controls", "target_permutation_geometry_control",
    "fitted_state_before_sha256", "fitted_state_after_sha256", "fitted_state_unchanged",
    "geometry_seconds", "feature_seconds", "readout_seconds", "fixed_lambda_one_readouts",
    "original_y_geometry_control", "geometry_diagnostics",
}
READOUT_KEYS = {
    "selected_shrinkage", "correction_coefficient_l2", "wall_seconds", "heldout_r2",
    "prediction", "prediction_sha256", "base_prediction_sha256", "feature_dimension",
    "train_feature_sha256", "holdout_feature_sha256", "selection_r2",
}
CONTROL_KEYS = {
    "permutation_version", "permutation_sha256", "null_semantics", "heldout_target_access",
    "heldout_r2", "prediction_sha256",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError("residual-anchor %s schema is not exact: %r" % (label, observed))


def _assert_finite(value: Any, label: str = "artifact") -> None:
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("residual-anchor %s contains nonfinite JSON number" % label)
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, "%s.%s" % (label, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, "%s[%d]" % (label, index))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_payload(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return sha256_payload({
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "bytes_sha256": sha256_bytes(array.tobytes(order="C")),
    })


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".%s." % path.name, dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(path))
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Any) -> None:
    _assert_finite(value)
    _atomic_bytes(path, (canonical_json(value) + "\n").encode("utf-8"))


def runtime_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_major_minor": "%d.%d" % (sys.version_info.major, sys.version_info.minor),
        "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "thread_environment": {key: os.environ.get(key) for key in THREAD_KEYS},
    }


def validate_runtime_contract() -> dict[str, Any]:
    observed = runtime_environment()
    for key, expected in RUNTIME_CONTRACT.items():
        if observed.get(key) != expected:
            raise RuntimeError("residual-anchor runtime contract mismatch at %s: %r != %r" % (key, observed.get(key), expected))
    return observed


def _parent_protocol_hash() -> str:
    return sha256_path(ladder.PROTOCOL_PATH)


def _source_hashes() -> dict[str, str]:
    paths = {
        "parent_ladder_runner": Path(ladder.__file__).resolve(),
        "parent_ladder_protocol": ladder.PROTOCOL_PATH,
        "parent_ladder_protocol_sidecar": ladder.PROTOCOL_SHA256_PATH,
        "parent_stable_runner": Path(stable.__file__).resolve(),
        "parent_geometry_runner": Path(geometry.__file__).resolve(),
        "residual_runner": Path(__file__).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("residual-anchor provenance paths missing: %r" % missing)
    return {name: sha256_path(path) for name, path in paths.items()}


def load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("residual-anchor protocol SHA-256 sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_continuous_residual_anchor_n300_before_outcomes":
        raise ValueError("residual-anchor protocol is not frozen before outcomes")
    if protocol.get("runtime_contract") != RUNTIME_CONTRACT:
        raise ValueError("residual-anchor runtime contract changed")
    if tuple(protocol.get("methods", ())) != METHODS:
        raise ValueError("residual-anchor method roster changed")
    if protocol.get("parent_protocol") != "continuous_information_ladder_n300":
        raise ValueError("residual-anchor parent protocol changed")
    if protocol.get("inner_residual_protocol", {}).get("shrinkage_grid") != list(SHRINKAGE_GRID):
        raise ValueError("residual-anchor shrinkage grid changed")
    expected_seed_offsets = {
        "outer_anchor_cv": OUTER_ANCHOR_CV_SEED_OFFSET,
        "residual_oof": RESIDUAL_OOF_SEED_OFFSET,
        "calibration_selection_split": CALIBRATION_SELECTION_SPLIT_SEED_OFFSET,
        "calibration_oof": CALIBRATION_OOF_SEED_OFFSET,
        "calibration_anchor": CALIBRATION_ANCHOR_SEED_OFFSET,
        "calibration_geometry": CALIBRATION_GEOMETRY_SEED_OFFSET,
        "final_geometry": FINAL_GEOMETRY_SEED_OFFSET,
        "target_permutation_geometry": TARGET_PERMUTATION_GEOMETRY_SEED_OFFSET,
        "original_y_geometry": ORIGINAL_Y_GEOMETRY_SEED_OFFSET,
    }
    if protocol.get("inner_residual_protocol", {}).get("seed_offsets") != expected_seed_offsets:
        raise ValueError("residual-anchor seed offsets changed")
    expected_grid = {
        "target_count": len(TARGETS), "backbone_count": len(BACKBONES), "panel_count": len(TARGETS),
        "candidate_count": len(TARGETS) * len(BACKBONES), "outer_fold_count": len(TARGETS) * len(BACKBONES) * OUTER_FOLDS,
        "ranking_method_count": len(METHODS), "ranking_row_count": len(TARGETS) * len(METHODS), "reserved_backbone_count": 0,
    }
    if protocol.get("expected_grid") != expected_grid:
        raise ValueError("residual-anchor expected grid changed")
    if protocol.get("roster", {}).get("reserved_backbones") != []:
        raise ValueError("residual-anchor reserved backbone roster changed")
    if protocol.get("source_hashes") is None:
        raise ValueError("residual-anchor source hashes are not frozen")
    actual = _source_hashes()
    if protocol.get("source_hashes") != actual:
        changed = [key for key in set(protocol.get("source_hashes", {})) | set(actual) if protocol.get("source_hashes", {}).get(key) != actual.get(key)]
        raise ValueError("residual-anchor source hash changed: %r" % changed)
    if protocol.get("parent_protocol_sha256") != _parent_protocol_hash():
        raise ValueError("residual-anchor parent protocol hash changed")
    return protocol


def _outer_split(target_position: int, fold: int) -> tuple[np.ndarray, np.ndarray]:
    return ladder._outer_split(target_position, fold)


def _fit_seed(target_position: int, candidate_position: int, fold: int) -> int:
    return ladder._fit_seed(target_position, candidate_position, fold)


def _pca_seed(target_position: int, candidate_position: int, fold: int) -> int:
    return ladder._pca_seed(target_position, candidate_position, fold)


def _safe_r2(y_holdout: np.ndarray, prediction: np.ndarray, y_train: np.ndarray) -> float:
    score, _excursion, _scale = ladder._safe_r2(y_holdout, prediction, y_train, "residual-anchor")
    return float(score)


def _load_source(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Bind this diagnostic to the parent source truth and exact matrix bank."""
    parent = ladder.load_protocol()
    source, cohort, datasets = ladder.validate_source(parent)
    # Parent validation supplies the authoritative ordered panels/cohort.  This
    # package records the parent hash and never reparses a mutable source.
    if len(source.get("panels", ())) != len(TARGETS):
        raise ValueError("residual-anchor source target roster is incomplete")
    return source, cohort, datasets


def _source_panel_by_target(source: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    panels = {str(row["target"]): row for row in source.get("panels", ())}
    if tuple(panels) != TARGETS and set(panels) != set(TARGETS):
        raise ValueError("residual-anchor source target roster changed")
    if set(panels) != set(TARGETS):
        raise ValueError("residual-anchor source target set incomplete")
    return panels


def _residual_geometry(
    values_fit: np.ndarray,
    values_holdout: np.ndarray,
    residual_fit: np.ndarray,
    fit_seed: int,
) -> dict[str, Any]:
    """Fit one shared SAVE/RFF/prototype geometry to residual targets."""
    standardized, standardization = _standardize_residual_targets(residual_fit)
    result = ladder._fit_geometry_variant(
        np.asarray(values_fit, dtype=np.float32),
        np.asarray(values_holdout, dtype=np.float32),
        standardized,
        int(fit_seed),
    )
    result["_residual_standardization"] = {
        "center": standardization["center"],
        "scale": standardization["scale"],
        "target_sha256": sha256_array(standardized),
        "formula": "(residual - fit_mean) / max(fit_std, 1e-12)",
    }
    return result


def _standardize_residual_targets(residual_fit: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(residual_fit, dtype=np.float64).reshape(-1)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("residual-anchor residual geometry targets are nonfinite")
    center = float(np.mean(values))
    scale = float(max(np.std(values), 1.0e-12))
    return (values - center) / scale, {"center": center, "scale": scale}


def _feature_for(geometry_result: Mapping[str, Any], method: str, side: str) -> np.ndarray:
    key = GEOMETRY_FEATURES[method]
    values = geometry_result["features"][side][key]
    return np.asarray(values, dtype=np.float64)


def _build_selection_corrections(
    values_train: np.ndarray,
    base_oof: np.ndarray,
    targets: np.ndarray,
    residual_oof: np.ndarray,
    fit_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], tuple[np.ndarray, np.ndarray]]:
    """Fit residual geometry on an inner fit portion and predict its selection portion."""
    fit_rows, select_rows = _deterministic_selection_split(len(targets), int(fit_seed) + 1103)
    inner_geometry = _residual_geometry(values_train[fit_rows], values_train[select_rows], residual_oof[fit_rows], int(fit_seed) + 1301)
    correction_predictions = {}
    readout_diagnostics = {}
    for method in RESIDUAL_METHODS:
        train_features = _feature_for(inner_geometry, method, "train")
        select_features = _feature_for(inner_geometry, method, "holdout")
        prediction, diagnostics = _fit_correction_readout(train_features, select_features, residual_oof[fit_rows])
        full = np.zeros(len(targets), dtype=np.float64)
        full[select_rows] = prediction
        correction_predictions[method] = full
        readout_diagnostics[method] = {
            "feature_dimension": int(train_features.shape[1]),
            "train_feature_sha256": sha256_array(train_features),
            "selection_feature_sha256": sha256_array(select_features),
            "coefficient_l2": float(diagnostics.get("coefficient_l2", 0.0)),
        }
    return correction_predictions, {
        "fit_rows_sha256": sha256_array(fit_rows),
        "selection_rows_sha256": sha256_array(select_rows),
        "readouts": readout_diagnostics,
        "target_access": "fit-portion OOF residuals only",
        "selection_target_access": "selection targets used only for shrinkage scoring",
    }, (fit_rows, select_rows)


def _readout_record(
    method: str,
    base_holdout: np.ndarray,
    correction_holdout: np.ndarray,
    y_holdout: np.ndarray,
    y_train: np.ndarray,
    shrinkage: float,
    selection_r2: float,
    diagnostics: Mapping[str, Any],
    train_features: Optional[np.ndarray] = None,
    holdout_features: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    prediction = np.asarray(base_holdout, dtype=np.float64) + float(shrinkage) * np.asarray(correction_holdout, dtype=np.float64)
    score = _safe_r2(np.asarray(y_holdout, dtype=np.float64), prediction, np.asarray(y_train, dtype=np.float64))
    if train_features is None:
        train_hash = sha256_array(np.asarray(base_holdout, dtype=np.float64))
        holdout_hash = sha256_array(np.asarray(base_holdout, dtype=np.float64))
        feature_dimension = 0
    else:
        train_hash = sha256_array(np.asarray(train_features, dtype=np.float64))
        holdout_hash = sha256_array(np.asarray(holdout_features, dtype=np.float64))
        feature_dimension = int(np.asarray(train_features).shape[1])
    return {
        "selected_shrinkage": float(shrinkage),
        "correction_coefficient_l2": float(diagnostics.get("coefficient_l2", 0.0)),
        "wall_seconds": float(diagnostics.get("wall_seconds", 0.0)),
        "heldout_r2": float(score),
        "prediction": prediction.tolist(),
        "prediction_sha256": sha256_array(prediction),
        "base_prediction_sha256": sha256_array(np.asarray(base_holdout, dtype=np.float64)),
        "feature_dimension": feature_dimension,
        "train_feature_sha256": train_hash,
        "holdout_feature_sha256": holdout_hash,
        "selection_r2": float(selection_r2),
    }


def _deterministic_row_permutation(sample_ids: Sequence[str], target_position: int, candidate_position: int, fold: int, version: str) -> np.ndarray:
    decorated = []
    for index, sample_id in enumerate(sample_ids):
        digest = hashlib.sha256((version + "|" + str(target_position) + "|" + str(candidate_position) + "|" + str(fold) + "|" + str(sample_id)).encode("utf-8")).hexdigest()
        decorated.append((digest, index))
    return np.asarray([index for _digest, index in sorted(decorated)], dtype=np.int64)


def _fold_readouts(
    values_train: np.ndarray,
    values_holdout: np.ndarray,
    y_train: np.ndarray,
    y_holdout: np.ndarray,
    train_rows: np.ndarray,
    holdout_rows: np.ndarray,
    train_sample_ids: Sequence[str],
    holdout_sample_ids: Sequence[str],
    target_position: int,
    candidate_position: int,
    fold: int,
    pca_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    fit_seed = _fit_seed(target_position, candidate_position, fold)
    base_oof, oof_diag = _oof_base_predictions(values_train, y_train, fit_seed + INNER_SEED_OFFSET)
    residual_oof = np.asarray(y_train, dtype=np.float64) - base_oof
    base_holdout, base_diag = _fit_full_base(values_train, values_holdout, y_train, fit_seed)
    selection_corrections, selection_diag, (selection_fit_rows, selection_rows) = _build_selection_corrections(
        values_train, base_oof, y_train, residual_oof, fit_seed
    )
    shrinkages = {}
    for method in RESIDUAL_METHODS:
        selected_shrinkage, choice = _select_shrinkage(
            base_oof, y_train, selection_corrections[method], selection_fit_rows, selection_rows
        )
        shrinkages[method] = (selected_shrinkage, choice)

    geometry_started = time.perf_counter()
    final_geometry = _residual_geometry(values_train, values_holdout, residual_oof, fit_seed + 1301)
    geometry_seconds = float(time.perf_counter() - geometry_started)
    before = final_geometry["before"]
    readouts = {}
    linear_started = time.perf_counter()
    linear_diag = dict(base_diag)
    readouts[OUTER_LINEAR] = _readout_record(
        OUTER_LINEAR, base_holdout, np.zeros_like(base_holdout), y_holdout, y_train, 0.0,
        0.0,
        {"coefficient_l2": linear_diag.get("coefficient_l2", 0.0), "wall_seconds": time.perf_counter() - linear_started},
    )
    for method in RESIDUAL_METHODS:
        train_features = _feature_for(final_geometry, method, "train")
        holdout_features = _feature_for(final_geometry, method, "holdout")
        correction_holdout, correction_diag = _fit_correction_readout(train_features, holdout_features, residual_oof)
        shrinkage, choice = shrinkages[method]
        readouts[method] = _readout_record(
            method, base_holdout, correction_holdout, y_holdout, y_train, shrinkage,
            float(choice["selected_r2"]), correction_diag, train_features, holdout_features,
        )
    fixed_lambda_one_readouts = {}
    for method in RESIDUAL_METHODS:
        train_features = _feature_for(final_geometry, method, "train")
        holdout_features = _feature_for(final_geometry, method, "holdout")
        correction_holdout, correction_diag = _fit_correction_readout(train_features, holdout_features, residual_oof)
        prediction = base_holdout + correction_holdout
        fixed_lambda_one_readouts[method] = {
            "selected_shrinkage": 1.0,
            "heldout_r2": _safe_r2(y_holdout, prediction, y_train),
            "prediction_sha256": sha256_array(prediction),
            "correction_sha256": sha256_array(correction_holdout),
            "feature_dimension": int(train_features.shape[1]),
            "coefficient_l2": float(correction_diag.get("coefficient_l2", 0.0)),
        }
    # Comparator: fit target-aware geometry to the original outer-training y,
    # then read out the raw OOF residuals with lambda fixed at one.  It is an
    # explicitly labelled diagnostic control, never a ranking method.
    original_y_geometry = ladder._fit_geometry_variant(
        values_train, values_holdout, np.asarray(y_train, dtype=np.float64), fit_seed + 1701
    )
    original_y_control = {}
    for method in RESIDUAL_METHODS:
        train_features = _feature_for(original_y_geometry, method, "train")
        holdout_features = _feature_for(original_y_geometry, method, "holdout")
        correction_holdout, _diag = _fit_correction_readout(train_features, holdout_features, residual_oof)
        prediction = base_holdout + correction_holdout
        original_y_control[method] = {
            "fixed_shrinkage": 1.0,
            "heldout_r2": _safe_r2(y_holdout, prediction, y_train),
            "prediction_sha256": sha256_array(prediction),
            "geometry_target": "original_outer_training_y",
            "heldout_target_access": False,
        }
    after = fixed_state = final_geometry["after"]
    if before["sha256"] != fixed_state["sha256"]:
        raise RuntimeError("residual-anchor feature/readout construction mutated fitted geometry")

    permutation = _deterministic_row_permutation(train_sample_ids, target_position, candidate_position, fold, NEGATIVE_PERMUTATION_VERSION)
    permutation_controls = {}
    for method in RESIDUAL_METHODS:
        train_features = _feature_for(final_geometry, method, "train")
        holdout_features = _feature_for(final_geometry, method, "holdout")
        permuted_correction, _diag = _fit_correction_readout(train_features[permutation], holdout_features, residual_oof)
        shrinkage = readouts[method]["selected_shrinkage"]
        prediction = base_holdout + shrinkage * permuted_correction
        permutation_controls[method] = {
            "permutation_version": NEGATIVE_PERMUTATION_VERSION,
            "permutation_sha256": sha256_array(permutation),
            "null_semantics": CONTROL_NULL_SEMANTICS,
            "heldout_target_access": False,
            "heldout_r2": _safe_r2(y_holdout, prediction, y_train),
            "prediction_sha256": sha256_array(prediction),
        }
    permuted_base, _ = _fit_full_base(values_train[permutation], values_holdout, y_train, fit_seed)
    permutation_controls[OUTER_LINEAR] = {
        "permutation_version": NEGATIVE_PERMUTATION_VERSION,
        "permutation_sha256": sha256_array(permutation),
        "null_semantics": CONTROL_NULL_SEMANTICS,
        "heldout_target_access": False,
        "heldout_r2": _safe_r2(y_holdout, permuted_base, y_train),
        "prediction_sha256": sha256_array(permuted_base),
    }

    target_permutation = _deterministic_row_permutation(train_sample_ids, target_position, candidate_position, fold, TARGET_PERMUTATION_VERSION)
    target_geometry = _residual_geometry(values_train, values_holdout, residual_oof[target_permutation], fit_seed + 1601)
    target_readouts = {}
    for method in RESIDUAL_METHODS:
        train_features = _feature_for(target_geometry, method, "train")
        holdout_features = _feature_for(target_geometry, method, "holdout")
        correction, _diag = _fit_correction_readout(train_features, holdout_features, residual_oof)
        shrinkage = readouts[method]["selected_shrinkage"]
        prediction = base_holdout + shrinkage * correction
        target_readouts[method] = {
            "heldout_r2": _safe_r2(y_holdout, prediction, y_train),
            "prediction_sha256": sha256_array(prediction),
        }
    feature_seconds = float(time.perf_counter() - geometry_started - geometry_seconds)
    return {
        "fold": int(fold),
        "fit_seed": int(fit_seed),
        "pca_seed": int(_pca_seed(target_position, candidate_position, fold)),
        "train_row_indices": np.asarray(train_rows, dtype=np.int64).tolist(),
        "holdout_row_indices": np.asarray(holdout_rows, dtype=np.int64).tolist(),
        "train_sample_ids": [str(x) for x in train_sample_ids],
        "holdout_sample_ids": [str(x) for x in holdout_sample_ids],
        "base_oof_prediction_sha256": sha256_array(base_oof),
        "residual_target_sha256": sha256_array(residual_oof),
        "selected_shrinkage": {method: float(readouts[method]["selected_shrinkage"]) for method in RESIDUAL_METHODS},
        "shrinkage_selection": {method: choice for method, (_selected, choice) in shrinkages.items()},
        "readouts": readouts,
        "permutation_controls": permutation_controls,
        "target_permutation_geometry_control": {
            "target_permutation_version": TARGET_PERMUTATION_VERSION,
            "target_permutation_sha256": sha256_array(target_permutation),
            "null_semantics": TARGET_NULL_SEMANTICS,
            "heldout_target_access": False,
            "readouts": target_readouts,
        },
        "fitted_state_before_sha256": before["sha256"],
        "fitted_state_after_sha256": fixed_state["sha256"],
        "fitted_state_unchanged": True,
        "fixed_lambda_one_readouts": fixed_lambda_one_readouts,
        "original_y_geometry_control": original_y_control,
        "geometry_seconds": geometry_seconds,
        "feature_seconds": max(0.0, feature_seconds),
        "readout_seconds": float(time.perf_counter() - started),
        "geometry_diagnostics": {
            "pca": dict(pca_metadata),
            "prototype_space": "equal-weight SAVE+RFF residual geometry",
            "target_access": "outer-training OOF residuals only",
            "heldout_target_access": False,
            "oof_base": oof_diag,
            "selection": selection_diag,
            "residual_standardization": final_geometry["_residual_standardization"],
            "original_y_geometry_state_sha256": original_y_geometry["after"]["sha256"],
            "base_selected_alpha": float(base_diag.get("selected_alpha", FIXED_ALPHA)),
        },
    }


def evaluate_fold(
    pca_train: np.ndarray,
    pca_holdout: np.ndarray,
    y_train: np.ndarray,
    y_holdout: np.ndarray,
    train_rows: np.ndarray,
    holdout_rows: np.ndarray,
    train_sample_ids: Sequence[str],
    holdout_sample_ids: Sequence[str],
    target_position: int,
    candidate_position: int,
    fold: int,
    pca_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if len(train_rows) != OUTER_TRAIN_ROWS or len(holdout_rows) != OUTER_HOLDOUT_ROWS:
        raise ValueError("residual-anchor outer fold row counts changed")
    expected_train, expected_holdout = _outer_split(target_position, fold)
    if not np.array_equal(np.asarray(train_rows), expected_train) or not np.array_equal(np.asarray(holdout_rows), expected_holdout):
        raise ValueError("residual-anchor outer fold identity changed")
    return _fold_readouts(
        np.asarray(pca_train, dtype=np.float32), np.asarray(pca_holdout, dtype=np.float32),
        np.asarray(y_train, dtype=np.float64), np.asarray(y_holdout, dtype=np.float64),
        np.asarray(train_rows, dtype=np.int64), np.asarray(holdout_rows, dtype=np.int64),
        train_sample_ids, holdout_sample_ids, target_position, candidate_position, fold, pca_metadata,
    )


def evaluate_candidate(
    matrix: np.ndarray,
    full_train_indices: np.ndarray,
    y_full: np.ndarray,
    full_sample_ids: Sequence[str],
    source_candidate: Mapping[str, Any],
    target_position: int,
    candidate_position: int,
) -> dict[str, Any]:
    values = np.asarray(matrix)
    full_indices = np.asarray(full_train_indices, dtype=np.int64).reshape(-1)
    targets = np.asarray(y_full, dtype=np.float64).reshape(-1)
    sample_ids = [str(value) for value in full_sample_ids]
    if values.ndim != 2 or full_indices.shape != (N300_TRAIN_ROWS,) or targets.shape != (N300_TRAIN_ROWS,) or len(sample_ids) != N300_TRAIN_ROWS or not np.all(np.isfinite(values)) or not np.all(np.isfinite(targets)):
        raise ValueError("residual-anchor candidate matrix/cohort is malformed")
    folds = []
    for fold in range(OUTER_FOLDS):
        train_rows, holdout_rows = _outer_split(target_position, fold)
        pca_train, pca_holdout, pca_metadata = geometry.pca_geometry(
            values, full_indices[train_rows], full_indices[holdout_rows], _pca_seed(target_position, candidate_position, fold)
        )
        folds.append(evaluate_fold(
            pca_train, pca_holdout, targets[train_rows], targets[holdout_rows],
            train_rows, holdout_rows,
            [sample_ids[index] for index in train_rows], [sample_ids[index] for index in holdout_rows],
            target_position, candidate_position, fold, pca_metadata,
        ))
    scores = {method: float(np.mean([float(row["readouts"][method]["heldout_r2"]) for row in folds])) for method in METHODS}
    runtimes = {method: float(np.sum([float(row["readouts"][method]["wall_seconds"]) for row in folds])) for method in METHODS}
    pipeline = {OUTER_LINEAR: runtimes[OUTER_LINEAR]}
    for method in RESIDUAL_METHODS:
        pipeline[method] = float(np.sum([
            float(row["geometry_seconds"]) + float(row["feature_seconds"]) + float(row["readouts"][method]["wall_seconds"])
            for row in folds
        ]))
    return {
        "candidate_id": str(source_candidate["candidate_id"]),
        "truth": {"heldout_envelope_r2": float(source_candidate["heldout_envelope_r2"])},
        "archived_source_linear_cv_r2": float(source_candidate["heads"]["linear_ridge"]["inner_best_r2"]),
        "archived_source_linear_runtime_seconds": float(source_candidate["heads"]["linear_ridge"]["wall_seconds"]),
        "source_pca": dict(source_candidate["pca"]),
        "scores": scores, "runtime_seconds": runtimes, "pipeline_runtime_seconds": pipeline, "folds": folds,
    }


def run_panel(
    protocol: Mapping[str, Any],
    datasets: Mapping[str, Mapping[str, Any]],
    source_panel: Mapping[str, Any],
    target_position: int,
    cohort: Mapping[str, Any],
) -> dict[str, Any]:
    target = str(source_panel["target"])
    train_indices, _evaluation_indices, y_train, _y_eval, train_ids, eval_ids = ladder._n300_rows(cohort, target)
    dataset_id = str(source_panel["dataset_id"])
    matrices = datasets[dataset_id]["matrices"]
    candidates = []
    for candidate_position, backbone in enumerate(BACKBONES):
        source_candidates = [row for row in source_panel["candidates"] if str(row["candidate_id"]) == backbone]
        if len(source_candidates) != 1:
            raise ValueError("residual-anchor source candidate roster changed")
        candidates.append(evaluate_candidate(
            np.asarray(matrices[backbone]), train_indices, y_train, train_ids,
            source_candidates[0], target_position, candidate_position,
        ))
    return {
        "dataset_id": dataset_id,
        "target_id": str(source_panel["target_id"]),
        "target": target,
        "stratum": STRATA[target],
        "budget_id": N300_BUDGET_ID,
        "outer_seed": int(OUTER_SEED_BASE + target_position),
        "training_rows": N300_TRAIN_ROWS,
        "evaluation_rows": N300_EVALUATION_ROWS,
        "training_sample_ids": [str(x) for x in train_ids],
        "evaluation_sample_ids": [str(x) for x in eval_ids],
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _top_indices(values: Sequence[float], depth: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(-array, kind="stable")
    threshold = array[order[min(depth, len(order)) - 1]]
    return np.flatnonzero(array >= threshold - TIE_ATOL)


def ranking_metrics(truth: Sequence[float], scores: Sequence[float]) -> dict[str, Any]:
    truth_values = np.asarray(truth, dtype=np.float64)
    score_values = np.asarray(scores, dtype=np.float64)
    if truth_values.shape != score_values.shape or truth_values.shape != (len(BACKBONES),) or not np.all(np.isfinite(truth_values)) or not np.all(np.isfinite(score_values)):
        raise ValueError("residual-anchor ranking inputs are malformed")
    selected = np.flatnonzero(score_values >= np.max(score_values) - TIE_ATOL)
    truth_best = np.flatnonzero(truth_values >= np.max(truth_values) - TIE_ATOL)
    selected_top = _top_indices(score_values, 3)
    truth_top = _top_indices(truth_values, 3)
    if np.unique(truth_values).size < 2 or np.unique(score_values).size < 2:
        correlation = None
        status = "undefined_constant_input"
    else:
        correlation = spearmanr(truth_values, score_values).correlation
        status = "defined" if np.isfinite(correlation) else "undefined_nonfinite_result"
    pair_credit = []
    for left in range(len(truth_values)):
        for right in range(left + 1, len(truth_values)):
            delta = float(truth_values[left] - truth_values[right])
            if abs(delta) < MATERIAL_PAIR_DIFFERENCE_R2:
                continue
            score_delta = float(score_values[left] - score_values[right])
            pair_credit.append(0.5 if abs(score_delta) <= TIE_ATOL else float(np.sign(score_delta) == np.sign(delta)))
    return {
        "spearman": None if correlation is None or not np.isfinite(correlation) else float(correlation),
        "spearman_status": status,
        "top1_regret_r2": float(np.max(truth_values) - np.max(truth_values[selected])),
        "exact_best": bool(set(selected).intersection(truth_best)),
        "top3_opportunity_regret_r2": float(np.max(truth_values) - np.max(truth_values[selected_top])),
        "top3_mean_utility_regret_r2": float(np.mean(truth_values[truth_top]) - np.mean(truth_values[selected_top])),
        "top3_set_recall": float(len(set(selected_top).intersection(truth_top)) / len(set(truth_top))),
        "material_pair_accuracy": float(np.mean(pair_credit)) if pair_credit else None,
        "material_pair_count": len(pair_credit),
        "selected_candidate_ids": [BACKBONES[index] for index in selected],
        "selected_top3_candidate_ids": [BACKBONES[index] for index in selected_top],
    }


def ranking_rows(panels: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for panel in panels:
        truth = [float(candidate["truth"]["heldout_envelope_r2"]) for candidate in panel["candidates"]]
        for method in METHODS:
            rows.append({
                "target": str(panel["target"]), "dataset_id": str(panel["dataset_id"]),
                "target_id": str(panel["target_id"]), "stratum": str(panel["stratum"]), "method": method,
                **ranking_metrics(truth, [float(candidate["scores"][method]) for candidate in panel["candidates"]]),
            })
    return rows


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else None


def _ranking_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for group in GROUPS:
        group_rows = [row for row in rows if group == "all" or row["stratum"] == group]
        for method in METHODS:
            selected = [row for row in group_rows if row["method"] == method]
            defined = [row["spearman"] for row in selected if row["spearman"] is not None]
            output.append({
                "group": group, "method": method, "panel_count": len(selected),
                "spearman": _mean(defined),
                "spearman_status": "defined" if len(defined) == len(selected) else "partial_undefined" if defined else "undefined_all_panels",
                "spearman_defined_panel_count": len(defined),
                "top1_regret_r2": _mean([row["top1_regret_r2"] for row in selected]),
                "top3_opportunity_regret_r2": _mean([row["top3_opportunity_regret_r2"] for row in selected]),
                "top3_mean_utility_regret_r2": _mean([row["top3_mean_utility_regret_r2"] for row in selected]),
                "top3_set_recall": _mean([row["top3_set_recall"] for row in selected]),
                "exact_best_rate": _mean([float(row["exact_best"]) for row in selected]),
                "material_pair_accuracy": _mean([row["material_pair_accuracy"] for row in selected if row["material_pair_accuracy"] is not None]),
            })
    return output


def _target_ranking_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for target in TARGETS:
        for row in [row for row in rows if row["target"] == target]:
            output.append({
                "target": target, "stratum": STRATA[target], "method": row["method"],
                "spearman": row["spearman"], "spearman_status": row["spearman_status"],
                "top1_regret_r2": row["top1_regret_r2"], "top3_opportunity_regret_r2": row["top3_opportunity_regret_r2"],
                "top3_mean_utility_regret_r2": row["top3_mean_utility_regret_r2"], "top3_set_recall": row["top3_set_recall"],
                "material_pair_accuracy": row["material_pair_accuracy"], "material_pair_count": row["material_pair_count"],
            })
    return output


def _runtime_summary(panels: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in METHODS:
        values = [float(candidate["runtime_seconds"][method]) for panel in panels for candidate in panel["candidates"]]
        pipelines = [float(candidate["pipeline_runtime_seconds"][method]) for panel in panels for candidate in panel["candidates"]]
        output.append({
            "method": method, "mean_seconds": _mean(values),
            "median_seconds": float(np.median(values)) if values else None,
            "pipeline_mean_seconds": _mean(pipelines),
            "pipeline_median_seconds": float(np.median(pipelines)) if pipelines else None,
            "count": len(values), "timing_label": "same-process four-fold outer CV; diagnostic proxy",
        })
    return output


def _control_summary(panels: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = {method: [] for method in METHODS}
    for panel in panels:
        for candidate in panel["candidates"]:
            for fold in candidate["folds"]:
                records = fold[key]
                if key == "target_permutation_geometry_control":
                    records = records["readouts"]
                for method in METHODS:
                    if method in records:
                        values[method].append(float(records[method]["heldout_r2"]))
    return {
        method: {
            "mean_r2": _mean(rows),
            "min_r2": float(np.min(rows)) if rows else None,
            "max_r2": float(np.max(rows)) if rows else None,
            "count": len(rows),
        }
        for method, rows in values.items()
    }


def _zero_shrinkage_selection_fractions(
    panels: Sequence[Mapping[str, Any]], folds: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Summarize observed zero selections without confusing them with availability."""
    output = {}
    stratum_values: dict[str, dict[str, list[float]]] = {}
    for panel in panels:
        stratum = str(panel["stratum"])
        values = stratum_values.setdefault(stratum, {method: [] for method in RESIDUAL_METHODS})
        panel_folds = [
            fold for candidate in panel["candidates"] for fold in candidate["folds"]
        ]
        for method in RESIDUAL_METHODS:
            values[method].extend(
                float(float(fold["selected_shrinkage"][method]) == 0.0)
                for fold in panel_folds
            )
    for method in RESIDUAL_METHODS:
        output[method] = {
            "overall": _mean([
                float(float(fold["selected_shrinkage"][method]) == 0.0)
                for fold in folds
            ]),
            "by_stratum": {
                stratum: _mean(values[method])
                for stratum, values in sorted(stratum_values.items())
            },
        }
    return output


def _diagnose(panels: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    folds = [fold for panel in panels for candidate in panel["candidates"] for fold in candidate["folds"]]
    means = {
        method: _mean([float(fold["readouts"][method]["heldout_r2"]) for fold in folds])
        for method in METHODS
    }
    state_ok = all(
        fold["fitted_state_unchanged"] is True and fold["fitted_state_before_sha256"] == fold["fitted_state_after_sha256"]
        for fold in folds
    )
    linear_rows = [row for row in rows if row["method"] == OUTER_LINEAR]
    best_method = max(METHODS, key=lambda method: means[method])
    best_rows = [row for row in rows if row["method"] == best_method]
    zero_option_available = all(
        any(
            float(grid_row["shrinkage"]) == 0.0
            for grid_row in fold["shrinkage_selection"][method]["grid"]
        )
        for fold in folds
        for method in RESIDUAL_METHODS
    )
    return {
        "status": "diagnostic_only",
        "mean_heldout_r2": means,
        "exploratory_best_method": best_method,
        "state_nonmutation": state_ok,
        # This gate is about the frozen candidate set, not whether the fitted
        # selector happened to choose zero in each fold.
        "no_harm_option_present": bool(zero_option_available),
        "zero_shrinkage_selection_fraction_by_method": _zero_shrinkage_selection_fractions(panels, folds),
        "ranking_checks": {
            "best_method_top1_regret_r2": _mean([row["top1_regret_r2"] for row in best_rows]),
            "outer_linear_top1_regret_r2": _mean([row["top1_regret_r2"] for row in linear_rows]),
            "best_method_top3_regret_r2": _mean([row["top3_opportunity_regret_r2"] for row in best_rows]),
            "outer_linear_top3_regret_r2": _mean([row["top3_opportunity_regret_r2"] for row in linear_rows]),
        },
        "conditional_association_null_summary": _control_summary(panels, "permutation_controls"),
        "target_permutation_geometry_null_summary": _control_summary(panels, "target_permutation_geometry_control"),
        "interpretation": "The residual channel is diagnostic; zero shrinkage is a valid no-harm outcome and no method is promoted.",
    }


def summarize(panels: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    rows = ranking_rows(panels)
    return {
        "schema_version": 1,
        "status": "completed_continuous_residual_anchor_n300",
        "selected_targets": list(TARGETS),
        "panel_count": len(panels),
        "candidate_count": sum(len(panel["candidates"]) for panel in panels),
        "ranking_rows": rows,
        "ranking_summaries": _ranking_summaries(rows),
        "target_ranking_summaries": _target_ranking_summaries(rows),
        "runtime_summary": _runtime_summary(panels),
        "diagnosis": _diagnose(panels, rows),
        "truth_is_independent": False,
        "reserved_backbones_evaluated": [],
        "method_promotion": "none_diagnostic_only",
    }


def report(summary: Mapping[str, Any]) -> str:
    diagnosis = summary["diagnosis"]
    lines = [
        "# Continuous residual-anchor n=300",
        "",
        "Diagnostic only; no method promotion.",
        "",
        "| method | mean held-out R2 |",
        "|---|---:|",
    ]
    for method in METHODS:
        lines.append("| %s | %.6f |" % (method, float(diagnosis["mean_heldout_r2"][method])))
    lines.extend([
        "", "Selected shrinkage includes a zero no-harm option: %s." % diagnosis["no_harm_option_present"],
        "State nonmutation: %s." % diagnosis["state_nonmutation"],
        "",
        "## Observed zero-shrinkage selection",
        "",
        "| method | overall fraction | by stratum |",
        "|---|---:|---|",
    ])
    fractions = diagnosis["zero_shrinkage_selection_fraction_by_method"]
    for method in RESIDUAL_METHODS:
        by_stratum = fractions[method]["by_stratum"]
        strata_text = ", ".join(
            "%s=%.6f" % (stratum, float(value))
            for stratum, value in by_stratum.items()
        )
        overall = fractions[method]["overall"]
        lines.append("| %s | %s | %s |" % (
            method,
            "undefined" if overall is None else "%.6f" % float(overall),
            strata_text,
        ))
    lines.append("")
    return "\n".join(lines)


def _checkpoint_identity(source_hashes: Mapping[str, str]) -> str:
    return sha256_payload({"protocol_sha256": sha256_path(PROTOCOL_PATH), "source_hashes": dict(source_hashes), "parent_protocol_sha256": _parent_protocol_hash()})


def _validate_readout(record: Mapping[str, Any], label: str) -> None:
    _require_exact_keys(record, READOUT_KEYS, label)
    prediction = np.asarray(record["prediction"], dtype=np.float64)
    if prediction.shape != (OUTER_HOLDOUT_ROWS,) or not np.all(np.isfinite(prediction)):
        raise ValueError("residual-anchor %s prediction malformed" % label)
    if record["prediction_sha256"] != sha256_array(prediction) or record["base_prediction_sha256"] is None:
        raise ValueError("residual-anchor %s prediction digest changed" % label)
    for key in ("selected_shrinkage", "correction_coefficient_l2", "wall_seconds", "heldout_r2", "selection_r2"):
        if isinstance(record[key], bool) or not isinstance(record[key], (int, float)) or not np.isfinite(float(record[key])):
            raise ValueError("residual-anchor %s nonfinite metric" % label)
    if float(record["selected_shrinkage"]) not in SHRINKAGE_GRID:
        raise ValueError("residual-anchor %s selected shrinkage outside frozen grid" % label)
    if not isinstance(record["feature_dimension"], int) or record["feature_dimension"] < 0:
        raise ValueError("residual-anchor %s feature dimension malformed" % label)


def _validate_fold(fold: Mapping[str, Any], target_position: int, candidate_position: int) -> None:
    _require_exact_keys(fold, FOLD_KEYS, "fold")
    fold_id = int(fold["fold"])
    if not 0 <= fold_id < OUTER_FOLDS or int(fold["fit_seed"]) != _fit_seed(target_position, candidate_position, fold_id):
        raise ValueError("residual-anchor fold seed changed")
    train_rows = np.asarray(fold["train_row_indices"], dtype=np.int64)
    holdout_rows = np.asarray(fold["holdout_row_indices"], dtype=np.int64)
    expected_train, expected_holdout = _outer_split(target_position, fold_id)
    if not np.array_equal(train_rows, expected_train) or not np.array_equal(holdout_rows, expected_holdout):
        raise ValueError("residual-anchor fold rows changed")
    if len(fold["train_sample_ids"]) != OUTER_TRAIN_ROWS or len(fold["holdout_sample_ids"]) != OUTER_HOLDOUT_ROWS:
        raise ValueError("residual-anchor fold sample IDs malformed")
    if fold["fitted_state_unchanged"] is not True or fold["fitted_state_before_sha256"] != fold["fitted_state_after_sha256"]:
        raise ValueError("residual-anchor fitted state changed")
    if set(fold["readouts"]) != set(METHODS) or set(fold["permutation_controls"]) != set(METHODS):
        raise ValueError("residual-anchor method/control set changed")
    for method in METHODS:
        _validate_readout(fold["readouts"][method], "%s fold %d" % (method, fold_id))
        control = fold["permutation_controls"][method]
        _require_exact_keys(control, CONTROL_KEYS, "%s control" % method)
        if control["null_semantics"] != CONTROL_NULL_SEMANTICS or control["heldout_target_access"] is not False:
            raise ValueError("residual-anchor conditional control semantics changed")
    if set(fold["fixed_lambda_one_readouts"]) != set(RESIDUAL_METHODS):
        raise ValueError("residual-anchor fixed-lambda control method set changed")
    for method, record in fold["fixed_lambda_one_readouts"].items():
        if record.get("selected_shrinkage") != 1.0 or not np.isfinite(float(record.get("heldout_r2", np.nan))) or not isinstance(record.get("prediction_sha256"), str) or not isinstance(record.get("correction_sha256"), str):
            raise ValueError("residual-anchor fixed-lambda control malformed")
    if set(fold["original_y_geometry_control"]) != set(RESIDUAL_METHODS):
        raise ValueError("residual-anchor original-y geometry control method set changed")
    for method, record in fold["original_y_geometry_control"].items():
        if record.get("fixed_shrinkage") != 1.0 or record.get("geometry_target") != "original_outer_training_y" or record.get("heldout_target_access") is not False or not np.isfinite(float(record.get("heldout_r2", np.nan))):
            raise ValueError("residual-anchor original-y geometry control malformed")
    target_control = fold["target_permutation_geometry_control"]
    if set(target_control["readouts"]) != set(RESIDUAL_METHODS) or target_control["null_semantics"] != TARGET_NULL_SEMANTICS or target_control["heldout_target_access"] is not False:
        raise ValueError("residual-anchor target null semantics changed")
    for method, record in target_control["readouts"].items():
        _require_exact_keys(record, {"heldout_r2", "prediction_sha256"}, "%s target control" % method)
        if not np.isfinite(float(record["heldout_r2"])):
            raise ValueError("residual-anchor target null R2 malformed")


def validate_artifact(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    protocol = load_protocol()
    source, _cohort, _datasets = _load_source(protocol)
    source_by_target = _source_panel_by_target(source)
    root = Path(output_root)
    raw = json.loads((root / "raw_results.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    _require_exact_keys(raw, RAW_KEYS, "raw")
    _require_exact_keys(summary, SUMMARY_KEYS, "summary")
    _require_exact_keys(manifest, MANIFEST_KEYS, "manifest")
    _assert_finite(raw); _assert_finite(summary); _assert_finite(manifest)
    source_hashes = _source_hashes()
    identity = _checkpoint_identity(source_hashes)
    if raw["status"] != "completed" or raw["protocol_sha256"] != sha256_path(PROTOCOL_PATH) or raw["checkpoint_identity"] != identity:
        raise ValueError("residual-anchor terminal identity changed")
    panels = raw["panels"]
    if len(panels) != len(TARGETS):
        raise ValueError("residual-anchor panel count changed")
    for target_position, panel in enumerate(panels):
        _require_exact_keys(panel, PANEL_KEYS, "panel")
        target = TARGETS[target_position]
        source_panel = source_by_target[target]
        if panel["target"] != target or panel["candidate_count"] != len(BACKBONES):
            raise ValueError("residual-anchor panel identity changed")
        if [str(x) for x in panel["training_sample_ids"]] != [str(x) for x in source_panel["training_sample_ids"]] or [str(x) for x in panel["evaluation_sample_ids"]] != [str(x) for x in source_panel["evaluation_sample_ids"]]:
            raise ValueError("residual-anchor panel sample identity changed")
        if tuple(candidate["candidate_id"] for candidate in panel["candidates"]) != BACKBONES:
            raise ValueError("residual-anchor candidate roster changed")
        for candidate_position, candidate in enumerate(panel["candidates"]):
            _require_exact_keys(candidate, CANDIDATE_KEYS, "candidate")
            source_candidate = [row for row in source_panel["candidates"] if str(row["candidate_id"]) == BACKBONES[candidate_position]]
            if len(source_candidate) != 1:
                raise ValueError("residual-anchor source candidate identity changed")
            source_candidate = source_candidate[0]
            if not np.isclose(float(candidate["truth"]["heldout_envelope_r2"]), float(source_candidate["heldout_envelope_r2"]), atol=TIE_ATOL, rtol=0.0) or candidate["source_pca"] != source_candidate["pca"]:
                raise ValueError("residual-anchor candidate source binding changed")
            if len(candidate["folds"]) != OUTER_FOLDS:
                raise ValueError("residual-anchor fold count changed")
            for fold in candidate["folds"]:
                _validate_fold(fold, target_position, candidate_position)
    if summary["panel_count"] != len(TARGETS) or summary["candidate_count"] != len(TARGETS) * len(BACKBONES):
        raise ValueError("residual-anchor summary counts changed")
    if manifest["status"] != "completed" or manifest["decision_status"] != "diagnostic_only":
        raise ValueError("residual-anchor manifest status changed")
    return {"status": "valid", "panel_count": len(panels), "candidate_count": sum(len(panel["candidates"]) for panel in panels), "source_targets": len(source_by_target)}


def _repository() -> dict[str, Any]:
    try:
        import subprocess
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--short"], cwd=str(ROOT), check=True, capture_output=True, text=True).stdout.splitlines()
    except Exception:
        head, status = "unknown", []
    return {"head": head, "dirty": bool(status), "status_lines": status}


def run(output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False) -> dict[str, Any]:
    """Execute outcomes only into a fresh artifact root.

    A terminal artifact is never overwritten.  ``resume`` is deliberately
    conservative: checkpoints must have the exact current protocol/source
    identity and are only reused by the caller after validation.
    """
    validate_runtime_contract()
    protocol = load_protocol()
    source, cohort, datasets = _load_source(protocol)
    source_by_target = _source_panel_by_target(source)
    source_hashes = _source_hashes()
    identity = _checkpoint_identity(source_hashes)
    root = Path(output_root)
    if (root / "terminal.json").exists() or (root / "manifest.json").exists():
        if resume:
            raise RuntimeError("residual-anchor refuses resume into an existing terminal/output root")
        raise FileExistsError("residual-anchor output root exists; choose a fresh root")
    root.mkdir(parents=True, exist_ok=False)
    panels = []
    try:
        for target_position, target in enumerate(TARGETS):
            panel = run_panel(protocol, datasets, source_by_target[target], target_position, cohort)
            panels.append(panel)
            _atomic_json(root / ("checkpoint_%02d.json" % target_position), {
                "schema_version": 1, "status": "completed_panel", "protocol_sha256": sha256_path(PROTOCOL_PATH),
                "checkpoint_identity": identity, "target": target, "panel": panel,
            })
        raw = {
            "schema_version": 1, "status": "completed", "protocol_sha256": sha256_path(PROTOCOL_PATH),
            "source_hashes": source_hashes, "environment": runtime_environment(),
            "checkpoint_identity": identity, "selected_targets": list(TARGETS),
            "reserved_backbones_evaluated": [], "panels": panels,
        }
        summary = summarize(panels, protocol)
        manifest = {
            "schema_version": 1, "status": "completed", "decision_status": "diagnostic_only",
            "protocol_sha256": sha256_path(PROTOCOL_PATH), "source_hashes": source_hashes,
            "checkpoint_identity": identity,
            "artifacts": {"raw_results": "raw_results.json", "summary": "summary.json", "report": "report.md"},
            "counts": {"targets": len(TARGETS), "backbones": len(BACKBONES), "panels": len(panels), "candidates": sum(len(panel["candidates"]) for panel in panels), "outer_folds": len(panels) * len(BACKBONES) * OUTER_FOLDS},
            "environment": runtime_environment(), "method_promotion": "none_diagnostic_only",
        }
        _atomic_json(root / "raw_results.json", raw)
        _atomic_json(root / "summary.json", summary)
        _atomic_bytes(root / "report.md", (report(summary) + "\n").encode("utf-8"))
        _atomic_json(root / "manifest.json", manifest)
        _atomic_json(root / "terminal.json", {"schema_version": 1, "status": "completed", "protocol_sha256": sha256_path(PROTOCOL_PATH), "checkpoint_identity": identity, "artifact_hashes": {name: sha256_path(root / name) for name in ("raw_results.json", "summary.json", "report.md", "manifest.json")}})
        validate_artifact(root)
        return summary
    except BaseException:
        # A partially written root is retained as evidence; no terminal marker
        # is produced and a subsequent run must use a fresh path.
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        print(canonical_json(validate_artifact(args.output_root)))
    else:
        print(canonical_json(run(args.output_root, resume=args.resume)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





def _oof_base_predictions(values: np.ndarray, targets: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Produce train-only OOF Ridge predictions for residual targets."""
    values = np.asarray(values, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or len(values) != len(targets) or len(values) < INNER_FOLDS:
        raise ValueError("residual-anchor OOF base inputs are malformed")
    splitter = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=int(seed))
    predictions = np.empty(len(targets), dtype=np.float64)
    fold_digests = []
    for inner_fold, (fit_rows, holdout_rows) in enumerate(splitter.split(values)):
        prediction, diagnostics = stable._fit_unscaled_ridge(
            values[fit_rows], values[holdout_rows], targets[fit_rows], int(seed) + inner_fold
        )
        predictions[holdout_rows] = np.asarray(prediction, dtype=np.float64)
        fold_digests.append({
            "fold": int(inner_fold),
            "fit_rows_sha256": sha256_array(np.asarray(fit_rows, dtype=np.int64)),
            "holdout_rows_sha256": sha256_array(np.asarray(holdout_rows, dtype=np.int64)),
            "selected_alpha": float(diagnostics["selected_alpha"]),
        })
    if not np.all(np.isfinite(predictions)):
        raise ValueError("residual-anchor OOF base predictions are nonfinite")
    return predictions, {
        "splitter": "KFold",
        "n_splits": INNER_FOLDS,
        "seed": int(seed),
        "prediction_sha256": sha256_array(predictions),
        "folds": fold_digests,
    }


def _fit_full_base(values_train: np.ndarray, values_holdout: np.ndarray, targets: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    prediction, diagnostics = stable._fit_unscaled_ridge(values_train, values_holdout, targets, int(seed) + INNER_SEED_OFFSET)
    return np.asarray(prediction, dtype=np.float64), dict(diagnostics)


def _fit_correction_readout(train_features: np.ndarray, holdout_features: np.ndarray, residual_train: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    prediction, diagnostics = stable._fit_fixed_alpha_ridge(
        np.asarray(train_features, dtype=np.float64), np.asarray(holdout_features, dtype=np.float64),
        np.asarray(residual_train, dtype=np.float64), alpha=FIXED_ALPHA,
    )
    return np.asarray(prediction, dtype=np.float64), dict(diagnostics)


def _deterministic_selection_split(length: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if length < 3:
        raise ValueError("residual-anchor shrinkage selection needs at least three rows")
    # One deterministic fifth is reserved for shrinkage selection.  The
    # remaining 80% is the only data used to fit calibration geometry and
    # correction readouts.
    splitter = KFold(n_splits=5, shuffle=True, random_state=int(seed))
    fit_rows, select_rows = next(iter(splitter.split(np.arange(length))))
    return np.asarray(fit_rows, dtype=np.int64), np.asarray(select_rows, dtype=np.int64)


def _select_shrinkage(
    base_oof: np.ndarray,
    targets: np.ndarray,
    correction_predictions: np.ndarray,
    fit_rows: np.ndarray,
    select_rows: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Select lambda with a deterministic fit-only inner selection split.

    ``correction_predictions`` must be generated without using the selection
    rows' residual targets.  The helper is intentionally pure and is tested
    separately so a future implementation cannot silently drop lambda=0.
    """
    base = np.asarray(base_oof, dtype=np.float64).reshape(-1)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    correction = np.asarray(correction_predictions, dtype=np.float64).reshape(-1)
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    select_rows = np.asarray(select_rows, dtype=np.int64)
    if base.shape != y.shape or correction.shape != y.shape or len(np.intersect1d(fit_rows, select_rows)) or len(np.unique(np.concatenate([fit_rows, select_rows]))) != len(y):
        raise ValueError("residual-anchor shrinkage selection rows are malformed")
    candidates = []
    for shrinkage in SHRINKAGE_GRID:
        prediction = base[select_rows] + float(shrinkage) * correction[select_rows]
        score = _safe_r2(y[select_rows], prediction, y[fit_rows])
        candidates.append({"shrinkage": float(shrinkage), "selection_r2": float(score)})
    selected = max(candidates, key=lambda row: (row["selection_r2"], -row["shrinkage"]))
    return float(selected["shrinkage"]), {
        "fit_rows": fit_rows.tolist(),
        "selection_rows": select_rows.tolist(),
        "grid": candidates,
        "selected_shrinkage": float(selected["shrinkage"]),
        "selected_r2": float(selected["selection_r2"]),
        "tie_break": "smallest shrinkage",
    }
