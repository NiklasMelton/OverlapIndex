"""Run the frozen real-image structured-target head-diversity screen."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from experiments.continuous_bounded_shared_evaluator import shared
from experiments.continuous_real_bbox_panels import prepare as bbox_prepare
from experiments.continuous_real_pet_area_panel import extract as pet_extract
from experiments.continuous_real_pet_area_panel import prepare as pet_prepare
from experiments.continuous_residual_ranking import run as residual


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_structured_head_discovery"
SCOUT_BACKBONES = (
    "dinov2-small",
    "deit-tiny",
    "mobilenetv3-large",
    "resnet50",
    "convnext-tiny",
)
RESERVED_BACKBONES = (
    "openclip-vit-b-32",
    "efficientnet-b0",
    "swin-tiny",
    "vit-small-16",
    "densenet121",
)
REPLICATE_SEEDS = (4103, 4201, 4303)
HEADS = ("linear_ridge", "quadratic_ridge", "knn", "rbf_svr")
NONLINEAR_HEADS = HEADS[1:]
METHODS = (
    "continuous_broad_rff",
    "continuous_save_rff",
    "linear_cv",
    "raw_cv_max",
)
PCA_RANK = 32
TIE_ATOL = 1.0e-12


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_path(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (canonical_json(value) + "\n").encode("utf-8"))


def load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("structured-discovery protocol SHA-256 sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_development_structured_head_discovery_before_target_outcomes":
        raise ValueError("structured-discovery protocol is not frozen")
    if tuple(protocol["backbones"]["scout"]) != SCOUT_BACKBONES:
        raise ValueError("structured-discovery scout roster differs from code")
    if tuple(protocol["backbones"]["reserved"]) != RESERVED_BACKBONES:
        raise ValueError("structured-discovery reserved roster differs from code")
    if set(SCOUT_BACKBONES).intersection(RESERVED_BACKBONES):
        raise ValueError("structured-discovery scout and reserved rosters overlap")
    if tuple(protocol["replicate_seeds"]) != REPLICATE_SEEDS:
        raise ValueError("structured-discovery replicate seeds differ from code")
    if tuple(protocol["heads"]["families"]) != HEADS:
        raise ValueError("structured-discovery head set differs from code")
    expected_head_grid = {
        "linear_ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "quadratic_ridge": {
            "degree": 2,
            "include_bias": False,
            "alpha": [0.01, 0.1, 1.0, 10.0],
        },
        "knn": {"n_neighbors": [3, 5, 9, 15, 31], "weights": ["distance"]},
        "rbf_svr": {
            "C": [1.0, 10.0, 100.0],
            "gamma": ["scale", 0.03, 0.1, 0.3],
            "epsilon": [0.05, 0.1],
        },
    }
    if any(protocol["heads"].get(head) != grid for head, grid in expected_head_grid.items()):
        raise ValueError("structured-discovery head grid differs from implementation")
    if tuple(protocol["descriptive_rankers"]) != METHODS:
        raise ValueError("structured-discovery ranker set differs from code")
    if int(protocol["preprocessing"]["pca_rank"]) != PCA_RANK:
        raise ValueError("structured-discovery PCA rank differs from code")
    expected_targets = {
        "torchvision_gtsrb": (
            "box_area_fraction", "box_center_x", "box_center_y", "box_log_aspect_ratio"
        ),
        "torchvision_fgvc_aircraft": (
            "box_area_fraction", "box_center_x", "box_center_y", "box_log_aspect_ratio"
        ),
        "oxford_iiit_pet": (
            "mask_area_fraction", "mask_centroid_x", "mask_centroid_y",
            "mask_log_bbox_aspect_ratio",
        ),
    }
    observed = {
        dataset_id: tuple(row["targets"])
        for dataset_id, row in protocol["datasets"].items()
    }
    if observed != expected_targets:
        raise ValueError("structured-discovery target set differs from code")
    grid = protocol["expected_grid"]
    if (
        int(grid["dataset_count"]) != len(expected_targets)
        or int(grid["task_count"]) != sum(len(values) for values in expected_targets.values())
        or int(grid["panel_count"]) != 36
        or int(grid["candidate_rows"]) != 180
    ):
        raise ValueError("structured-discovery expected grid differs from code")
    return protocol


def box_geometry(
    width: int, height: int, x1: int, y1: int, x2: int, y2: int
) -> dict[str, float]:
    values = (width, height, x1, y1, x2, y2)
    if any(type(value) is not int for value in values):
        raise TypeError("box geometry requires strict integer dimensions and coordinates")
    if width <= 1 or height <= 1 or not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
        raise ValueError("box geometry is outside the image")
    box_width = x2 - x1 + 1
    box_height = y2 - y1 + 1
    result = {
        "box_area_fraction": float((box_width * box_height) / (width * height)),
        "box_center_x": float((x1 + x2) / (2.0 * (width - 1))),
        "box_center_y": float((y1 + y2) / (2.0 * (height - 1))),
        "box_log_aspect_ratio": float(
            np.log((box_width / float(width)) / (box_height / float(height)))
        ),
    }
    if not all(np.isfinite(tuple(result.values()))):
        raise ValueError("box geometry produced a nonfinite target")
    return result


def mask_geometry(mask: np.ndarray) -> dict[str, float]:
    values = np.asarray(mask)
    if values.ndim != 2 or min(values.shape) <= 1:
        raise ValueError("mask geometry requires a nontrivial two-dimensional mask")
    if not set(np.unique(values)).issubset({1, 2, 3}):
        raise ValueError("mask geometry contains an unknown trimap label")
    foreground = values != 2
    yy, xx = np.nonzero(foreground)
    if len(xx) == 0:
        raise ValueError("mask geometry has no foreground-or-boundary pixels")
    height, width = values.shape
    box_width = int(np.max(xx) - np.min(xx) + 1)
    box_height = int(np.max(yy) - np.min(yy) + 1)
    result = {
        "mask_area_fraction": float(len(xx) / values.size),
        "mask_centroid_x": float(np.mean(xx) / (width - 1)),
        "mask_centroid_y": float(np.mean(yy) / (height - 1)),
        "mask_log_bbox_aspect_ratio": float(
            np.log((box_width / float(width)) / (box_height / float(height)))
        ),
    }
    if not all(np.isfinite(tuple(result.values()))):
        raise ValueError("mask geometry produced a nonfinite target")
    return result


def _registry(protocol: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = protocol["external_embedding_registry"]
    path = Path(str(descriptor["path"]))
    if sha256_path(path) != descriptor["sha256"]:
        raise ValueError("structured-discovery embedding registry hash changed")
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("status") != "audited_complete" or registry.get("outcomes_inspected") is not False:
        raise ValueError("structured-discovery embedding registry is not an outcome-free input")
    return registry


def _registry_record(registry: Mapping[str, Any], dataset_id: str) -> Mapping[str, Any]:
    selected = [
        row for row in registry["datasets"]
        if isinstance(row, Mapping) and row.get("dataset_id") == dataset_id
    ]
    if len(selected) != 1:
        raise ValueError(f"structured-discovery registry row is not unique: {dataset_id}")
    return selected[0]


def _load_registry_matrix(
    record: Mapping[str, Any], backbone: str, row_ids: Sequence[str]
) -> np.ndarray:
    descriptor = record["embeddings"][backbone]
    matrix_path = Path(str(descriptor["matrix_path"]))
    manifest_path = Path(str(descriptor["manifest_path"]))
    if sha256_path(matrix_path) != descriptor["matrix_sha256"]:
        raise ValueError(f"structured-discovery matrix hash changed: {backbone}")
    if sha256_path(manifest_path) != descriptor["manifest_sha256"]:
        raise ValueError(f"structured-discovery matrix manifest hash changed: {backbone}")
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    union = tuple(str(value) for value in record["cohort_manifest"]["union"]["sample_ids"])
    if len(union) != len(set(union)) or len(matrix) != len(union):
        raise ValueError("structured-discovery registry matrix/row identity is malformed")
    if matrix.dtype != np.dtype("float32") or matrix.ndim != 2 or tuple(matrix.shape) != tuple(descriptor["shape"]):
        raise ValueError("structured-discovery matrix dtype or shape changed")
    positions = {sample_id: position for position, sample_id in enumerate(union)}
    if not set(row_ids).issubset(positions):
        raise ValueError("structured-discovery cohort is not contained in the embedding union")
    selected = np.asarray(matrix[[positions[value] for value in row_ids]], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ValueError("structured-discovery selected embeddings are nonfinite")
    return np.ascontiguousarray(selected)


def _bbox_targets(
    dataset_id: str,
    cohort_rows: Sequence[Mapping[str, Any]],
    record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_root = Path(str(record["source"]["raw_root"]))
    train, test = bbox_prepare._provider_datasets(dataset_id, raw_root)
    output: list[dict[str, Any]] = []
    if dataset_id == "torchvision_gtsrb":
        train_annotations, test_annotations, _sources = bbox_prepare._read_gtsrb_annotations(raw_root)
        for row in cohort_rows:
            sample_id = str(row["sample_id"])
            _prefix, split, position_text = sample_id.rsplit("/", 2)
            provider = train if split == "train" else test
            path = Path(provider._samples[int(position_text)][0])
            annotation = (
                train_annotations[(path.parent.name, path.name)]
                if split == "train" else test_annotations[path.name]
            )
            width, height, x1, y1, x2, y2 = (
                int(annotation[name])
                for name in ("Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2")
            )
            with Image.open(path) as image:
                if image.size != (width, height):
                    raise ValueError("GTSRB image dimensions differ from its annotation")
            targets = box_geometry(width, height, x1, y1, x2, y2)
            if abs(targets["box_area_fraction"] - float(row["target"])) > np.finfo(float).eps:
                raise ValueError("GTSRB stored area target does not match recomputation")
            output.append({**dict(row), "targets": targets})
        return output

    if dataset_id != "torchvision_fgvc_aircraft":
        raise ValueError(f"unknown structured bbox dataset {dataset_id!r}")
    box_path = raw_root / "fgvc-aircraft-2013b" / "data" / "images_box.txt"
    boxes: dict[str, tuple[int, int, int, int]] = {}
    for line in box_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5 or fields[0] in boxes:
            raise ValueError("Aircraft box annotations are malformed or duplicated")
        boxes[fields[0]] = tuple(int(value) for value in fields[1:])
    if len(boxes) != 10_000:
        raise ValueError("Aircraft box annotation population changed")
    for row in cohort_rows:
        sample_id = str(row["sample_id"])
        _prefix, split, position_text = sample_id.rsplit("/", 2)
        provider = train if split == "trainval" else test
        path = Path(provider._image_files[int(position_text)])
        with Image.open(path) as image:
            width, height = image.size
        targets = box_geometry(width, height, *boxes[path.stem])
        if abs(targets["box_area_fraction"] - float(row["target"])) > np.finfo(float).eps:
            raise ValueError("Aircraft stored area target does not match recomputation")
        output.append({**dict(row), "targets": targets})
    return output


def _load_bbox_datasets(
    protocol: Mapping[str, Any], registry: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    cohort_path = ROOT / str(protocol["datasets"]["torchvision_gtsrb"]["source_cohort"])
    if sha256_path(cohort_path) != protocol["datasets"]["torchvision_gtsrb"]["source_cohort_sha256"]:
        raise ValueError("structured-discovery bbox cohort hash changed")
    cohort = bbox_prepare.load_cohort(cohort_path.parent)
    by_dataset = {row["dataset_id"]: row for row in cohort["datasets"]}
    result: dict[str, dict[str, Any]] = {}
    for dataset_id in ("torchvision_gtsrb", "torchvision_fgvc_aircraft"):
        source = by_dataset[dataset_id]
        record = _registry_record(registry, dataset_id)
        rows = _bbox_targets(dataset_id, source["rows"], record)
        row_ids = [str(row["sample_id"]) for row in rows]
        result[dataset_id] = {
            "rows": rows,
            "matrices": {
                backbone: _load_registry_matrix(record, backbone, row_ids)
                for backbone in SCOUT_BACKBONES
            },
        }
    return result


def _load_pet_dataset(protocol: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = protocol["datasets"]["oxford_iiit_pet"]
    cohort_path = ROOT / str(descriptor["source_cohort"])
    embedding_manifest = ROOT / str(descriptor["source_embedding_manifest"])
    if sha256_path(cohort_path) != descriptor["source_cohort_sha256"]:
        raise ValueError("structured-discovery Pet cohort hash changed")
    if sha256_path(embedding_manifest) != descriptor["source_embedding_manifest_sha256"]:
        raise ValueError("structured-discovery Pet embedding manifest hash changed")
    cohort = pet_prepare.load_cohort(cohort_path.parent)
    rows = []
    for row in cohort["rows"]:
        sample_id = str(row["sample_id"])
        with Image.open(cohort_path.parent / "trimaps" / f"{sample_id}.png") as source:
            targets = mask_geometry(np.asarray(source))
        if abs(targets["mask_area_fraction"] - float(row["foreground_fraction"])) > np.finfo(float).eps:
            raise ValueError("Pet stored area target does not match recomputation")
        rows.append({**dict(row), "targets": targets})
    spec_by_id = {str(spec["backbone_id"]): spec for spec in pet_extract.BACKBONE_SPECS}
    matrices = {}
    for backbone in SCOUT_BACKBONES:
        spec = spec_by_id[backbone]
        pet_extract._validate_cache(embedding_manifest.parent, spec, cohort)
        matrix = np.load(embedding_manifest.parent / f"{backbone}.npy", mmap_mode="r", allow_pickle=False)
        selected = np.asarray(matrix, dtype=np.float32)
        if selected.shape[0] != len(rows) or not np.isfinite(selected).all():
            raise ValueError("structured-discovery Pet matrix is incomplete or nonfinite")
        matrices[backbone] = np.ascontiguousarray(selected)
    return {"rows": rows, "matrices": matrices}


def load_datasets(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    registry = _registry(protocol)
    result = _load_bbox_datasets(protocol, registry)
    result["oxford_iiit_pet"] = _load_pet_dataset(protocol)
    if set(result) != set(protocol["datasets"]):
        raise ValueError("structured-discovery dataset grid is incomplete")
    for dataset_id, data in result.items():
        rows = data["rows"]
        if len(rows) != 1_200 or len({row["sample_id"] for row in rows}) != len(rows):
            raise ValueError(f"structured-discovery {dataset_id} row grid is incomplete")
        for seed in REPLICATE_SEEDS:
            if sum(row["split"] == "training" and row["replicate_seed"] == seed for row in rows) != 200:
                raise ValueError(f"structured-discovery {dataset_id} training grid is incomplete")
        if sum(row["split"] == "evaluation" for row in rows) != 200:
            raise ValueError(f"structured-discovery {dataset_id} evaluation grid is incomplete")
        for target_id in protocol["datasets"][dataset_id]["targets"]:
            for split in ("training", "evaluation"):
                values = np.asarray(
                    [row["targets"][target_id] for row in rows if row["split"] == split],
                    dtype=np.float64,
                )
                if len(values) == 0 or not np.isfinite(values).all() or float(np.ptp(values)) <= 0.0:
                    raise ValueError(f"structured-discovery {dataset_id}/{target_id}/{split} target is degenerate")
    return result


def _indices(rows: Sequence[Mapping[str, Any]], seed: int) -> tuple[np.ndarray, np.ndarray]:
    train = [
        index for index, row in enumerate(rows)
        if row["split"] == "training" and int(row["replicate_seed"]) == seed
    ]
    test = [index for index, row in enumerate(rows) if row["split"] == "evaluation"]
    if len(train) != 200 or len(test) != 200 or set(train).intersection(test):
        raise ValueError("structured-discovery panel split is incomplete or overlapping")
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def pca_geometry(
    matrix: np.ndarray, train: np.ndarray, test: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_values = normalize(np.asarray(matrix[train], dtype=np.float32), norm="l2")
    test_values = normalize(np.asarray(matrix[test], dtype=np.float32), norm="l2")
    transform = PCA(n_components=PCA_RANK, svd_solver="randomized", random_state=int(seed))
    reduced_train = np.asarray(transform.fit_transform(train_values), dtype=np.float32)
    reduced_test = np.asarray(transform.transform(test_values), dtype=np.float32)
    if not np.isfinite(reduced_train).all() or not np.isfinite(reduced_test).all():
        raise ValueError("structured-discovery PCA produced nonfinite values")
    return reduced_train, reduced_test, {
        "rank": PCA_RANK,
        "input_dimension": int(matrix.shape[1]),
        "fit_rows": int(len(train)),
        "explained_variance_ratio_sum": float(np.sum(transform.explained_variance_ratio_)),
    }


def _fit_seed(seed: int, dataset_position: int, target_position: int, backbone_position: int) -> int:
    result = int(seed) * 100_000 + dataset_position * 10_000 + target_position * 100 + backbone_position
    if not 0 <= result < 2**32:
        raise ValueError("structured-discovery derived seed is outside uint32")
    return result


def _geometry_seed(seed: int, dataset_position: int, backbone_position: int) -> int:
    result = int(seed) * 10_000 + dataset_position * 100 + backbone_position
    if not 0 <= result < 2**32:
        raise ValueError("structured-discovery geometry seed is outside uint32")
    return result


def _selector_seed(seed: int, dataset_position: int, target_position: int) -> int:
    result = int(seed) + dataset_position * 100 + target_position * 10
    if not 0 <= result < 2**32:
        raise ValueError("structured-discovery selector seed is outside uint32")
    return result


def evaluate_candidate(
    backbone: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    head_seed: int,
    selector_seed: int,
    position: int,
    pca: Mapping[str, Any],
) -> dict[str, Any]:
    heads = residual.evaluate_heads(X_train, y_train, X_test, y_test, head_seed)
    selector = shared.evaluate_shared_pair(
        X_train,
        np.zeros((len(X_train), 2), dtype=np.float32),
        y_train,
        selector_seed,
        position,
    )
    heldout = {head: float(heads[head]["test_r2"]) for head in HEADS}
    inner = {head: float(heads[head]["inner_best_r2"]) for head in HEADS}
    envelope = max(heldout.values())
    best_head_ids = [head for head in HEADS if abs(heldout[head] - envelope) <= TIE_ATOL]
    inner_maximum = max(inner.values())
    inner_best_ids = [head for head in HEADS if abs(inner[head] - inner_maximum) <= TIE_ATOL]
    return {
        "candidate_id": backbone,
        "pca": dict(pca),
        "heads": heads,
        "best_head_envelope_r2": float(envelope),
        "best_head_ids": best_head_ids,
        "inner_best_head_ids": inner_best_ids,
        "nonlinear_minus_linear_r2": float(max(heldout[head] for head in NONLINEAR_HEADS) - heldout["linear_ridge"]),
        "selectors": {
            "continuous_broad_rff": float(selector["broad_rff"]["score"]),
            "continuous_save_rff": float(selector["save_rff_current"]["score"]),
            "linear_cv": float(inner["linear_ridge"]),
            "raw_cv_max": float(max(inner.values())),
        },
        "selector_wall_seconds": float(selector["wall_seconds"]),
    }


def run_panel(
    protocol: Mapping[str, Any],
    datasets: Mapping[str, Mapping[str, Any]],
    dataset_id: str,
    target_id: str,
    seed: int,
    geometry_cache: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    dataset_position = tuple(protocol["datasets"]).index(dataset_id)
    target_position = tuple(protocol["datasets"][dataset_id]["targets"]).index(target_id)
    data = datasets[dataset_id]
    rows = data["rows"]
    train, test = _indices(rows, seed)
    targets = np.asarray([row["targets"][target_id] for row in rows], dtype=np.float64)
    candidates = []
    cache = {} if geometry_cache is None else geometry_cache
    for position, backbone in enumerate(SCOUT_BACKBONES):
        fit_seed = _fit_seed(seed, dataset_position, target_position, position)
        geometry_key = (dataset_id, int(seed), backbone)
        if geometry_key not in cache:
            cache[geometry_key] = pca_geometry(
                data["matrices"][backbone],
                train,
                test,
                _geometry_seed(seed, dataset_position, position),
            )
        X_train, X_test, pca = cache[geometry_key]
        candidates.append(
            evaluate_candidate(
                backbone,
                X_train,
                X_test,
                targets[train],
                targets[test],
                fit_seed,
                _selector_seed(seed, dataset_position, target_position),
                position,
                pca,
            )
        )
    return {
        "dataset_id": dataset_id,
        "target_id": target_id,
        "seed": int(seed),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    output = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and abs(array[order[stop]] - array[order[start]]) <= TIE_ATOL:
            stop += 1
        output[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return output


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    x, y = _ranks(left), _ranks(right)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("structured-discovery Spearman inputs are not aligned")
    x -= np.mean(x)
    y -= np.mean(y)
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return None if denominator <= np.finfo(float).eps else float(np.dot(x, y) / denominator)


def ranking_metrics(panel: Mapping[str, Any], method: str) -> dict[str, Any]:
    truth = np.asarray([row["best_head_envelope_r2"] for row in panel["candidates"]], dtype=float)
    scores = np.asarray([row["selectors"][method] for row in panel["candidates"]], dtype=float)
    if not np.isfinite(truth).all() or not np.isfinite(scores).all():
        raise ValueError("structured-discovery ranking values are nonfinite")
    selected = np.flatnonzero(np.abs(scores - np.max(scores)) <= TIE_ATOL)
    optimum = float(np.max(truth))
    selected_utility = float(np.mean(truth[selected]))
    return {
        "method": method,
        "spearman": spearman(truth, scores),
        "top1_regret_r2": float(optimum - selected_utility),
        "exact_best": bool(np.any(np.abs(truth[selected] - optimum) <= TIE_ATOL)),
        "selected_count": int(len(selected)),
    }


def _task_summary(
    task_panels: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    rows = [candidate for panel in task_panels for candidate in panel["candidates"]]
    admission = protocol["admission"]
    threshold = float(admission["material_nonlinear_row_threshold_r2"])
    stable = []
    for backbone in SCOUT_BACKBONES:
        candidate_rows = [
            candidate for panel in task_panels for candidate in panel["candidates"]
            if candidate["candidate_id"] == backbone
        ]
        unique_nonlinear = [
            candidate["best_head_ids"][0]
            for candidate in candidate_rows
            if len(candidate["best_head_ids"]) == 1 and candidate["best_head_ids"][0] in NONLINEAR_HEADS
        ]
        counts = Counter(unique_nonlinear)
        if counts:
            family, count = counts.most_common(1)[0]
            if count >= 2:
                stable.append({"backbone": backbone, "head": family, "replicate_wins": int(count)})
    spans = [
        float(
            max(candidate["best_head_envelope_r2"] for candidate in panel["candidates"])
            - min(candidate["best_head_envelope_r2"] for candidate in panel["candidates"])
        )
        for panel in task_panels
    ]
    inner_correct = [
        bool(set(candidate["inner_best_head_ids"]).intersection(candidate["best_head_ids"]))
        for candidate in rows
    ]
    unique_winner_counts = Counter(
        candidate["best_head_ids"][0]
        for candidate in rows
        if len(candidate["best_head_ids"]) == 1
    )
    nonlinear_counts = Counter(
        {head: unique_winner_counts.get(head, 0) for head in NONLINEAR_HEADS}
    )
    modal_nonlinear = (
        nonlinear_counts.most_common(1)[0][0]
        if sum(nonlinear_counts.values()) else None
    )
    observed = {
        "median_best_head_r2": float(np.median([row["best_head_envelope_r2"] for row in rows])),
        "median_nonlinear_minus_linear_r2": float(np.median([row["nonlinear_minus_linear_r2"] for row in rows])),
        "material_nonlinear_row_fraction": float(np.mean([row["nonlinear_minus_linear_r2"] >= threshold for row in rows])),
        "stable_nonlinear_backbone_count": int(len(stable)),
        "median_backbone_envelope_span": float(np.median(spans)),
        "inner_cv_family_accuracy": float(np.mean(inner_correct)),
    }
    checks = [
        {
            "criterion": "recoverable_best_head",
            "observed": observed["median_best_head_r2"],
            "minimum": float(admission["median_best_head_r2_minimum"]),
            "pass": observed["median_best_head_r2"] >= float(admission["median_best_head_r2_minimum"]),
        },
        {
            "criterion": "material_median_nonlinear_advantage",
            "observed": observed["median_nonlinear_minus_linear_r2"],
            "minimum": float(admission["median_nonlinear_minus_linear_r2_minimum"]),
            "pass": observed["median_nonlinear_minus_linear_r2"] >= float(admission["median_nonlinear_minus_linear_r2_minimum"]),
        },
        {
            "criterion": "material_nonlinear_row_fraction",
            "observed": observed["material_nonlinear_row_fraction"],
            "minimum": float(admission["material_nonlinear_row_fraction_minimum"]),
            "pass": observed["material_nonlinear_row_fraction"] >= float(admission["material_nonlinear_row_fraction_minimum"]),
        },
        {
            "criterion": "stable_nonlinear_backbones",
            "observed": observed["stable_nonlinear_backbone_count"],
            "minimum": int(admission["stable_nonlinear_backbone_count_minimum"]),
            "pass": observed["stable_nonlinear_backbone_count"] >= int(admission["stable_nonlinear_backbone_count_minimum"]),
        },
        {
            "criterion": "backbone_envelope_spread",
            "observed": observed["median_backbone_envelope_span"],
            "minimum": float(admission["median_backbone_envelope_span_minimum"]),
            "pass": observed["median_backbone_envelope_span"] >= float(admission["median_backbone_envelope_span_minimum"]),
        },
        {
            "criterion": "inner_cv_family_reliability",
            "observed": observed["inner_cv_family_accuracy"],
            "minimum": float(admission["inner_cv_family_accuracy_minimum"]),
            "pass": observed["inner_cv_family_accuracy"] >= float(admission["inner_cv_family_accuracy_minimum"]),
        },
    ]
    rank_rows = [ranking_metrics(panel, method) for panel in task_panels for method in METHODS]
    rank_summary = []
    for method in METHODS:
        selected = [row for row in rank_rows if row["method"] == method]
        defined = [row["spearman"] for row in selected if row["spearman"] is not None]
        rank_summary.append(
            {
                "method": method,
                "mean_spearman": float(np.mean(defined)) if defined else None,
                "defined_spearman_panels": len(defined),
                "mean_top1_regret_r2": float(np.mean([row["top1_regret_r2"] for row in selected])),
                "exact_best_rate": float(np.mean([row["exact_best"] for row in selected])),
            }
        )
    return {
        "dataset_id": task_panels[0]["dataset_id"],
        "target_id": task_panels[0]["target_id"],
        "status": "admit" if all(row["pass"] for row in checks) else "reject",
        "observed": observed,
        "checks": checks,
        "unique_winner_counts": {head: int(unique_winner_counts.get(head, 0)) for head in HEADS},
        "stable_nonlinear_backbones": stable,
        "modal_nonlinear_head": modal_nonlinear,
        "ranker_summaries": rank_summary,
    }


def summarize(panels: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    expected = protocol["expected_grid"]
    if len(panels) != int(expected["panel_count"]):
        raise ValueError("structured-discovery completed panel count is wrong")
    identities = {(row["dataset_id"], row["target_id"], int(row["seed"])) for row in panels}
    expected_identities = {
        (dataset_id, target_id, seed)
        for dataset_id, descriptor in protocol["datasets"].items()
        for target_id in descriptor["targets"]
        for seed in REPLICATE_SEEDS
    }
    if identities != expected_identities or len(identities) != len(panels):
        raise ValueError("structured-discovery completed panel identities are incomplete or duplicated")
    for panel in panels:
        if tuple(row["candidate_id"] for row in panel["candidates"]) != SCOUT_BACKBONES:
            raise ValueError("structured-discovery panel scout roster changed")
    task_summaries = []
    for dataset_id, descriptor in protocol["datasets"].items():
        for target_id in descriptor["targets"]:
            selected = sorted(
                [row for row in panels if row["dataset_id"] == dataset_id and row["target_id"] == target_id],
                key=lambda row: int(row["seed"]),
            )
            task_summaries.append(_task_summary(selected, protocol))
    admitted = [
        {"dataset_id": row["dataset_id"], "target_id": row["target_id"], "modal_nonlinear_head": row["modal_nonlinear_head"]}
        for row in task_summaries if row["status"] == "admit"
    ]
    family_coverage = {
        head: sum(row["modal_nonlinear_head"] == head for row in admitted)
        for head in NONLINEAR_HEADS
    }
    return {
        "schema_version": 1,
        "status": "completed_development_structured_head_discovery",
        "scope": protocol["scope"],
        "panel_count": len(panels),
        "candidate_rows": sum(len(row["candidates"]) for row in panels),
        "task_summaries": task_summaries,
        "admitted_tasks": admitted,
        "admitted_task_count": len(admitted),
        "nonlinear_family_coverage": family_coverage,
        "reserved_backbones_evaluated": [],
        "next_stage": (
            "freeze an untouched reserved-backbone/structured-dataset validation panel"
            if admitted else "expand discovery to fresh structured datasets; no task met the frozen admission rule"
        ),
    }


def report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Structured-target nonlinear-head discovery", "",
        "This is a development-only dataset screen. `admit` means nonlinear-but-recoverable under the frozen rule; it is not a method promotion or confirmation result.", "",
        f"Admitted tasks: **{summary['admitted_task_count']} / {len(summary['task_summaries'])}**.", "",
        "| Dataset | Target | Decision | Best-head R2 | Nonlinear - linear R2 | Material rows | Stable backbones | Envelope span | Inner-CV family accuracy | Modal nonlinear head |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["task_summaries"]:
        observed = row["observed"]
        lines.append(
            f"| {row['dataset_id']} | {row['target_id']} | {row['status']} | "
            f"{observed['median_best_head_r2']:.4f} | {observed['median_nonlinear_minus_linear_r2']:+.4f} | "
            f"{observed['material_nonlinear_row_fraction']:.3f} | {observed['stable_nonlinear_backbone_count']} | "
            f"{observed['median_backbone_envelope_span']:.4f} | {observed['inner_cv_family_accuracy']:.3f} | "
            f"{row['modal_nonlinear_head'] or 'none'} |"
        )
    lines.extend(["", "## Nonlinear-family coverage", ""])
    lines.extend(f"- `{head}`: {count} admitted tasks" for head, count in summary["nonlinear_family_coverage"].items())
    lines.extend(["", "## Descriptive ranking", ""])
    for row in summary["task_summaries"]:
        lines.append(f"### {row['dataset_id']} / {row['target_id']}")
        lines.append("")
        lines.append("| Ranker | Mean Spearman | Mean top-1 regret R2 | Exact-best rate |")
        lines.append("|---|---:|---:|---:|")
        for method in row["ranker_summaries"]:
            correlation = "undefined" if method["mean_spearman"] is None else f"{method['mean_spearman']:.4f}"
            lines.append(
                f"| {method['method']} | {correlation} | {method['mean_top1_regret_r2']:.5f} | {method['exact_best_rate']:.3f} |"
            )
        lines.append("")
    lines.extend([
        "## Interpretation guardrails", "",
        "- Difficulty by itself was not used as a proxy for nonlinearity.",
        "- The five reserved backbones were not evaluated.",
        "- Earlier source-embedding outcomes make this development evidence only.",
        "- Missing nonlinear families remain missing; the screen does not relax its thresholds after seeing results.",
        "",
        f"Next stage: {summary['next_stage']}.",
    ])
    return "\n".join(lines) + "\n"


def _source_hashes(protocol: Mapping[str, Any]) -> dict[str, str]:
    paths = {
        "protocol": PROTOCOL_PATH,
        "runner": Path(__file__).resolve(),
        "bbox_prepare": Path(bbox_prepare.__file__).resolve(),
        "pet_prepare": Path(pet_prepare.__file__).resolve(),
        "pet_extract": Path(pet_extract.__file__).resolve(),
        "head_evaluator": Path(residual.__file__).resolve(),
        "selector_evaluator": Path(shared.__file__).resolve(),
    }
    hashes = {name: sha256_path(path) for name, path in paths.items()}
    hashes["external_registry"] = str(protocol["external_embedding_registry"]["sha256"])
    return hashes


def run(
    output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False
) -> dict[str, Any]:
    if type(resume) is not bool:
        raise TypeError("resume must be a strict bool")
    protocol = load_protocol()
    datasets = load_datasets(protocol)
    source_hashes = _source_hashes(protocol)
    output_root = output_root.resolve()
    terminal = output_root / "manifest.json"
    if terminal.exists():
        raise FileExistsError("structured-discovery terminal manifest already exists")
    checkpoints = output_root / "checkpoints"
    if checkpoints.exists() and not resume:
        raise FileExistsError("structured-discovery checkpoints exist; pass --resume")
    checkpoints.mkdir(parents=True, exist_ok=True)
    panels = []
    geometry_cache: dict[
        tuple[str, int, str], tuple[np.ndarray, np.ndarray, dict[str, Any]]
    ] = {}
    total = int(protocol["expected_grid"]["panel_count"])
    completed = 0
    for dataset_id, descriptor in protocol["datasets"].items():
        for seed in REPLICATE_SEEDS:
            for target_id in descriptor["targets"]:
                checkpoint = checkpoints / f"{dataset_id}__{target_id}__seed_{seed}.json"
                if checkpoint.exists():
                    if not resume:
                        raise FileExistsError(f"structured-discovery checkpoint exists: {checkpoint.name}")
                    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                    if payload.get("source_hashes") != source_hashes:
                        raise ValueError(f"structured-discovery stale checkpoint: {checkpoint.name}")
                    panel = payload["panel"]
                else:
                    started = time.perf_counter()
                    panel = run_panel(
                        protocol,
                        datasets,
                        dataset_id,
                        target_id,
                        seed,
                        geometry_cache,
                    )
                    _atomic_json(
                        checkpoint,
                        {
                            "schema_version": 1,
                            "source_hashes": source_hashes,
                            "panel_wall_seconds": float(time.perf_counter() - started),
                            "panel": panel,
                        },
                    )
                panels.append(panel)
                completed += 1
                print(f"completed {completed}/{total}: {dataset_id}/{target_id}/seed={seed}", flush=True)
    summary = summarize(panels, protocol)
    raw = {
        "schema_version": 1,
        "status": "completed",
        "protocol": protocol,
        "source_hashes": source_hashes,
        "panels": panels,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_root / "raw_results.json", raw)
    _atomic_json(output_root / "summary.json", summary)
    _atomic_bytes(output_root / "report.md", report(summary).encode("utf-8"))
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "status": "completed_development_structured_head_discovery",
        "implementation_commit": git_commit,
        "working_tree_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
            ).stdout.strip()
        ),
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "source_hashes": source_hashes,
        "files": {
            name: {"sha256": sha256_path(output_root / name), "bytes": (output_root / name).stat().st_size}
            for name in ("raw_results.json", "summary.json", "report.md")
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    _atomic_json(terminal, manifest)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    summary = run(args.output_root, resume=args.resume)
    print(canonical_json({"admitted_tasks": summary["admitted_tasks"], "next_stage": summary["next_stage"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
