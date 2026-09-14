"""Prepare the frozen four-source continuous-target cohort bank."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np
from PIL import Image
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_INPUT_ROOT = ROOT / "artifacts" / "continuous_real_source_qualification_v1" / "inputs"
DEFAULT_RAW_ROOT = DEFAULT_INPUT_ROOT / "raw"
DEFAULT_PREPARED_ROOT = DEFAULT_INPUT_ROOT / "prepared"
TARGET_BINS = 10


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("real-source qualification protocol SHA-256 mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_continuous_real_source_qualification_v1_before_outcomes":
        raise ValueError("real-source qualification protocol is not frozen")
    return protocol


def _safe_zip_members(source: zipfile.ZipFile) -> set[str]:
    names: set[str] = set()
    for info in source.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
            raise ValueError(f"unsafe archive member: {info.filename}")
        if not info.is_dir():
            names.add(info.filename)
    return names


def _rank_bins(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: (float(row["target"]), str(row["sample_id"])))
    if len(ordered) < TARGET_BINS or not all(math.isfinite(float(row["target"])) for row in ordered):
        raise ValueError("target binning requires finite values and at least ten rows")
    return {
        str(row["sample_id"]): min(TARGET_BINS - 1, position * TARGET_BINS // len(ordered))
        for position, row in enumerate(ordered)
    }


def _hash_order(rows: Iterable[Mapping[str, Any]], seed: int, role: str) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{seed}|{role}|{row['sample_id']}".encode("utf-8")).hexdigest(),
            str(row["sample_id"]),
        ),
    )


def _pick_balanced(
    rows: Sequence[Mapping[str, Any]], count: int, *, seed: int, role: str
) -> list[dict[str, Any]]:
    if count % TARGET_BINS:
        raise ValueError("balanced cohort count must be divisible by ten")
    bins = _rank_bins(rows)
    per_bin = count // TARGET_BINS
    selected: list[dict[str, Any]] = []
    for target_bin in range(TARGET_BINS):
        candidates = [row for row in rows if bins[str(row["sample_id"])] == target_bin]
        chosen = _hash_order(candidates, seed, f"{role}:{target_bin}")[:per_bin]
        if len(chosen) != per_bin:
            raise ValueError(f"target bin {target_bin} cannot supply {role}")
        selected.extend({**row, "target_bin": target_bin} for row in chosen)
    return selected


def _partition_native(
    training: Sequence[Mapping[str, Any]], evaluation: Sequence[Mapping[str, Any]],
    *, training_rows: int, evaluation_rows: int, seed: int,
) -> list[dict[str, Any]]:
    train = _pick_balanced(training, training_rows, seed=seed, role="training")
    test = _pick_balanced(evaluation, evaluation_rows, seed=seed, role="evaluation")
    return _finalize_rows(train, test)


def _partition_internal(
    rows: Sequence[Mapping[str, Any]], *, training_rows: int, evaluation_rows: int, seed: int,
) -> list[dict[str, Any]]:
    bins = _rank_bins(rows)
    evaluation: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    per_eval, per_train = evaluation_rows // TARGET_BINS, training_rows // TARGET_BINS
    if evaluation_rows % TARGET_BINS or training_rows % TARGET_BINS:
        raise ValueError("internal cohort sizes must be divisible by ten")
    for target_bin in range(TARGET_BINS):
        candidates = [row for row in rows if bins[str(row["sample_id"])] == target_bin]
        ordered = _hash_order(candidates, seed, f"partition:{target_bin}")
        if len(ordered) < per_eval + per_train:
            raise ValueError(f"target bin {target_bin} cannot supply disjoint cohorts")
        evaluation.extend({**row, "target_bin": target_bin} for row in ordered[:per_eval])
        training.extend(
            {**row, "target_bin": target_bin}
            for row in ordered[per_eval : per_eval + per_train]
        )
    return _finalize_rows(training, evaluation)


def _finalize_rows(
    training: Sequence[Mapping[str, Any]], evaluation: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    train = sorted(training, key=lambda row: (int(row["target_bin"]), str(row["sample_id"])))
    test = sorted(evaluation, key=lambda row: (int(row["target_bin"]), str(row["sample_id"])))
    output = [
        {**row, "split": "training", "budget_rank": position}
        for position, row in enumerate(train)
    ] + [
        {**row, "split": "evaluation", "budget_rank": None}
        for row in test
    ]
    ids = [str(row["sample_id"]) for row in output]
    if len(ids) != len(set(ids)):
        raise ValueError("selected cohort contains duplicate sample identities")
    return output


def _koniq_rows(raw_root: Path) -> list[dict[str, Any]]:
    scores_path = raw_root / "koniq" / "koniq10k_scores_and_distributions.zip"
    images_path = raw_root / "koniq" / "koniq10k_512x384.zip"
    with zipfile.ZipFile(scores_path) as scores, zipfile.ZipFile(images_path) as images:
        _safe_zip_members(scores)
        image_names = _safe_zip_members(images)
        with scores.open("koniq10k_scores_and_distributions.csv") as handle:
            reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8-sig", newline=""))
            rows = []
            for record in reader:
                name = str(record["image_name"])
                member = f"512x384/{name}"
                if member not in image_names:
                    raise ValueError(f"KonIQ image is missing: {name}")
                rows.append({
                    "sample_id": name, "target": float(record["MOS"]),
                    "image_archive": "koniq_images", "image_member": member,
                    "source_split": "published_full",
                })
    return rows


def _jhu_rows(raw_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    archive = raw_root / "jhu_crowd" / "jhu_crowd_v2.0.zip"
    output: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    with zipfile.ZipFile(archive) as source:
        names = _safe_zip_members(source)
        for split in output:
            label_member = f"jhu_crowd_v2.0/{split}/image_labels.txt"
            text = source.read(label_member).decode("utf-8-sig")
            for line in text.splitlines():
                fields = [value.strip() for value in line.split(",")]
                if len(fields) < 2:
                    raise ValueError(f"invalid JHU label row: {line!r}")
                identifier, count = fields[0], int(fields[1])
                member = f"jhu_crowd_v2.0/{split}/images/{identifier}.jpg"
                if member not in names:
                    raise ValueError(f"JHU image is missing: {member}")
                output[split].append({
                    "sample_id": f"{split}/{identifier}.jpg", "target": float(np.log1p(count)),
                    "raw_count": count, "image_archive": "jhu", "image_member": member,
                    "source_split": split,
                })
    return output["train"], output["test"]


def _mat_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, bytes):
        return [value.decode("utf-8")]
    if isinstance(value, np.ndarray):
        output: list[str] = []
        for item in value.reshape(-1):
            output.extend(_mat_strings(item))
        return output
    if isinstance(value, (list, tuple)):
        output = []
        for item in value:
            output.extend(_mat_strings(item))
        return output
    return []


def _normalize_sun_directory(value: str) -> str:
    normalized = value.replace("\\", "/").rstrip("/")
    marker = "SUNRGBD/"
    position = normalized.find(marker)
    if position < 0:
        raise ValueError(f"SUN split path has no SUNRGBD root: {value!r}")
    return normalized[position:]


def _sun_split_directories(raw_root: Path) -> tuple[set[str], set[str]]:
    toolbox = raw_root / "sunrgbd" / "SUNRGBDtoolbox.zip"
    with zipfile.ZipFile(toolbox) as source:
        names = _safe_zip_members(source)
        candidates = [
            name for name in names
            if name.endswith("/allsplit.mat") and not name.startswith("__MACOSX/")
        ]
        if len(candidates) != 1:
            raise ValueError("SUN toolbox must contain exactly one allsplit.mat")
        payload = loadmat(io.BytesIO(source.read(candidates[0])), simplify_cells=True)
    keys = {str(key).lower(): key for key in payload if not str(key).startswith("__")}
    train_key = next((keys[key] for key in keys if "train" in key), None)
    test_key = next((keys[key] for key in keys if "test" in key), None)
    if train_key is None or test_key is None:
        raise ValueError("SUN official split keys are missing")
    train = {_normalize_sun_directory(value) for value in _mat_strings(payload[train_key])}
    test = {_normalize_sun_directory(value) for value in _mat_strings(payload[test_key])}
    if not train or not test or train.intersection(test):
        raise ValueError("SUN official split is empty or overlapping")
    return train, test


def _decode_sun_depth(data: bytes) -> tuple[float, float]:
    with Image.open(io.BytesIO(data)) as image:
        encoded = np.asarray(image, dtype=np.uint16)
    decoded = np.bitwise_or(np.right_shift(encoded, 3), np.left_shift(encoded, 13)).astype(np.uint16)
    depth_m = decoded.astype(np.float64) / 1000.0
    valid = depth_m[(depth_m > 0.0) & (depth_m <= 10.0)]
    coverage = float(len(valid) / depth_m.size)
    if coverage < 0.25:
        raise ValueError("SUN depth map has insufficient valid coverage")
    median = float(np.median(valid))
    return float(np.log(median)), coverage


def _sun_rows(raw_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    archive = raw_root / "sunrgbd" / "SUNRGBD.zip"
    train_directories, test_directories = _sun_split_directories(raw_root)
    output: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    with zipfile.ZipFile(archive) as source:
        names = _safe_zip_members(source)
        images_by_directory: dict[str, list[str]] = {}
        for name in names:
            if "/image/" in name and name.lower().endswith((".jpg", ".jpeg", ".png")):
                images_by_directory.setdefault(name.split("/image/", 1)[0], []).append(name)
        depth_members = sorted(name for name in names if "/depth_bfx/" in name and name.lower().endswith(".png"))
        for depth_member in depth_members:
            directory = depth_member.split("/depth_bfx/", 1)[0]
            split = "train" if directory in train_directories else "test" if directory in test_directories else None
            if split is None:
                continue
            images = sorted(images_by_directory.get(directory, ()))
            if len(images) != 1:
                raise ValueError(f"SUN sample has {len(images)} RGB images: {directory}")
            target, coverage = _decode_sun_depth(source.read(depth_member))
            sample_id = directory.removeprefix("SUNRGBD/")
            output[split].append({
                "sample_id": sample_id, "target": target, "valid_depth_fraction": coverage,
                "image_archive": "sunrgbd", "image_member": images[0],
                "depth_member": depth_member, "source_split": split,
            })
    return output["train"], output["test"]


def _rsna_rows(raw_root: Path) -> list[dict[str, Any]]:
    annotations = raw_root / "rsna_bone_age" / "Bone_Age_Training_Set_Annotations.zip"
    images = raw_root / "rsna_bone_age" / "Bone_Age_Training_Set.zip"
    with zipfile.ZipFile(annotations) as labels, zipfile.ZipFile(images) as source:
        label_names = _safe_zip_members(labels)
        image_names = _safe_zip_members(source)
        image_by_stem: dict[str, list[str]] = {}
        for name in image_names:
            image_by_stem.setdefault(PurePosixPath(name).stem, []).append(name)
        csv_names = [name for name in label_names if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError("RSNA annotation archive must contain exactly one CSV")
        reader = csv.DictReader(io.TextIOWrapper(labels.open(csv_names[0]), encoding="utf-8-sig", newline=""))
        rows = []
        for record in reader:
            normalized = {str(key).strip().lower(): value for key, value in record.items()}
            identifier = str(normalized["id"]).strip()
            matches = sorted(image_by_stem.get(identifier, ()))
            if len(matches) != 1:
                raise ValueError(f"RSNA image identity is ambiguous: {identifier}")
            male_text = str(normalized["male"]).strip().lower()
            if male_text not in {"true", "false", "1", "0"}:
                raise ValueError(f"invalid RSNA sex value: {male_text!r}")
            rows.append({
                "sample_id": identifier, "target": float(normalized["boneage"]),
                "male": male_text in {"true", "1"}, "image_archive": "rsna_images",
                "image_member": matches[0], "source_split": "publisher_training",
            })
    return rows


def build_cohort(raw_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    counts = protocol["cohort"]
    train_rows, eval_rows = int(counts["training_pool_rows"]), int(counts["evaluation_rows"])
    seed = int(counts["selection_seed"])
    datasets: dict[str, Any] = {}
    koniq = _partition_internal(_koniq_rows(raw_root), training_rows=train_rows, evaluation_rows=eval_rows, seed=seed + 1)
    jhu_train, jhu_test = _jhu_rows(raw_root)
    jhu = _partition_native(jhu_train, jhu_test, training_rows=train_rows, evaluation_rows=eval_rows, seed=seed + 2)
    sun_train, sun_test = _sun_rows(raw_root)
    sun = _partition_native(sun_train, sun_test, training_rows=train_rows, evaluation_rows=eval_rows, seed=seed + 3)
    rsna = _partition_internal(_rsna_rows(raw_root), training_rows=train_rows, evaluation_rows=eval_rows, seed=seed + 4)
    for dataset_id, target_id, rows in (
        ("koniq10k_iqa", "mos", koniq),
        ("jhu_crowd_v2", "log1p_head_count", jhu),
        ("sunrgbd_scene_depth", "log_median_depth_m", sun),
        ("rsna_bone_age", "bone_age_months", rsna),
    ):
        for matrix_index, row in enumerate(rows):
            row["matrix_index"] = matrix_index
        datasets[dataset_id] = {"target_ids": [target_id], "rows": rows, "target_cohorts": {target_id: rows}}
    return {"schema_version": 1, "datasets": datasets}


def _archive_paths(raw_root: Path) -> dict[str, Path]:
    return {
        "koniq_images": raw_root / "koniq" / "koniq10k_512x384.zip",
        "koniq_scores": raw_root / "koniq" / "koniq10k_scores_and_distributions.zip",
        "jhu": raw_root / "jhu_crowd" / "jhu_crowd_v2.0.zip",
        "sunrgbd": raw_root / "sunrgbd" / "SUNRGBD.zip",
        "sunrgbd_toolbox": raw_root / "sunrgbd" / "SUNRGBDtoolbox.zip",
        "rsna_images": raw_root / "rsna_bone_age" / "Bone_Age_Training_Set.zip",
        "rsna_annotations": raw_root / "rsna_bone_age" / "Bone_Age_Training_Set_Annotations.zip",
    }


def validate_archives(raw_root: Path, protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    paths = _archive_paths(raw_root)
    expected = protocol["sources"]["archives"]
    if set(paths) != set(expected):
        raise ValueError("protocol source archive inventory changed")
    output = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"source archive is missing: {path}")
        observed = {"size_bytes": path.stat().st_size, "sha256": sha256_path(path)}
        if observed != {"size_bytes": int(expected[name]["size_bytes"]), "sha256": expected[name]["sha256"]}:
            raise ValueError(f"source archive identity changed: {name}")
        output[name] = {"path": str(path.relative_to(raw_root)), **observed}
    return output


def _copy_selected_images(raw_root: Path, prepared_root: Path, cohort: Mapping[str, Any]) -> None:
    archives = _archive_paths(raw_root)
    handles: dict[str, zipfile.ZipFile] = {}
    try:
        for dataset_id, dataset in cohort["datasets"].items():
            destination_root = prepared_root / "images" / dataset_id
            destination_root.mkdir(parents=True, exist_ok=False)
            for row in dataset["rows"]:
                archive_id = str(row["image_archive"])
                if archive_id not in handles:
                    handles[archive_id] = zipfile.ZipFile(archives[archive_id])
                    _safe_zip_members(handles[archive_id])
                data = handles[archive_id].read(str(row["image_member"]))
                suffix = PurePosixPath(str(row["image_member"])).suffix.lower() or ".img"
                filename = hashlib.sha256(str(row["sample_id"]).encode("utf-8")).hexdigest() + suffix
                destination = destination_root / filename
                temporary = destination.with_name(f".{destination.name}.tmp")
                temporary.write_bytes(data)
                temporary.replace(destination)
                with Image.open(destination) as image:
                    image.verify()
                row["image_path"] = str(destination.relative_to(prepared_root))
                row["image_sha256"] = sha256_path(destination)
    finally:
        for handle in handles.values():
            handle.close()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_inputs(raw_root: Path = DEFAULT_RAW_ROOT, prepared_root: Path = DEFAULT_PREPARED_ROOT) -> dict[str, Any]:
    raw_root, prepared_root = raw_root.resolve(), prepared_root.resolve()
    protocol = load_protocol()
    if prepared_root.exists():
        raise FileExistsError("prepared real-source inputs already exist")
    temporary_root = prepared_root.with_name(f".{prepared_root.name}.tmp")
    if temporary_root.exists():
        raise FileExistsError("partial temporary prepared-input stage exists")
    archive_records = validate_archives(raw_root, protocol)
    cohort = build_cohort(raw_root, protocol)
    temporary_root.mkdir(parents=True, exist_ok=False)
    _copy_selected_images(raw_root, temporary_root, cohort)
    cohort_path = temporary_root / "cohort.json"
    _atomic_json(cohort_path, cohort)
    target_count = sum(len(dataset["target_ids"]) for dataset in cohort["datasets"].values())
    row_count = sum(len(dataset["rows"]) for dataset in cohort["datasets"].values())
    manifest = {
        "schema_version": 1, "status": "prepared_reference_only_real_source_inputs",
        "protocol_sha256": sha256_path(PROTOCOL_PATH), "archives": archive_records,
        "cohort": {"path": "cohort.json", "sha256": sha256_path(cohort_path), "identity_sha256": sha256_payload(cohort)},
        "counts": {"datasets": len(cohort["datasets"]), "targets": target_count, "rows": row_count},
        "outcomes_inspected": False, "overlapindex_fit_count": 0, "reserved_backbones_evaluated": [],
    }
    _atomic_json(temporary_root / "manifest.json", manifest)
    temporary_root.replace(prepared_root)
    return manifest


def load_cohort(prepared_root: Path = DEFAULT_PREPARED_ROOT) -> dict[str, Any]:
    prepared_root = prepared_root.resolve()
    protocol = load_protocol()
    manifest = json.loads((prepared_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "prepared_reference_only_real_source_inputs":
        raise ValueError("prepared input manifest status changed")
    if manifest.get("protocol_sha256") != sha256_path(PROTOCOL_PATH):
        raise ValueError("prepared input protocol identity changed")
    cohort_path = prepared_root / "cohort.json"
    if manifest["cohort"]["sha256"] != sha256_path(cohort_path):
        raise ValueError("prepared cohort file identity changed")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if manifest["cohort"]["identity_sha256"] != sha256_payload(cohort):
        raise ValueError("prepared cohort payload identity changed")
    expected_rows = int(protocol["expected_grid"]["selected_image_count"])
    if sum(len(dataset["rows"]) for dataset in cohort["datasets"].values()) != expected_rows:
        raise ValueError("prepared cohort row count changed")
    for dataset_id, dataset in cohort["datasets"].items():
        for row in dataset["rows"]:
            image = prepared_root / row["image_path"]
            if not image.is_file() or sha256_path(image) != row["image_sha256"]:
                raise ValueError(f"prepared image identity changed: {dataset_id}/{row['sample_id']}")
    return cohort


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    args = parser.parse_args(argv)
    print(canonical_json(prepare_inputs(args.raw_root, args.prepared_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
