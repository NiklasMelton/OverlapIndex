"""Extract the frozen five-backbone bank for the real-source qualification."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from experiments.continuous_real_age_panel import extract as model_source
from experiments.continuous_real_source_qualification_v1 import prepare


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_real_source_qualification_v1" / "embeddings"
BATCH_SIZE = 16
DEVICE = "cpu"
SPECS_BY_ID = {str(spec["backbone_id"]): spec for spec in model_source.BACKBONE_SPECS}


def _protocol_specs(protocol: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    ids = tuple(str(value) for value in protocol["backbones"])
    missing = [value for value in ids if value not in SPECS_BY_ID]
    if missing:
        raise ValueError(f"frozen backbone specifications are missing: {missing}")
    return tuple(SPECS_BY_ID[value] for value in ids)


def _values_sha(matrix: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(matrix).tobytes(order="C")).hexdigest()


def validate_model_inputs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for spec in _protocol_specs(protocol):
        checkpoint = model_source._checkpoint_path(spec)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"frozen model checkpoint is missing: {checkpoint}")
        observed = prepare.sha256_path(checkpoint)
        if observed != spec["checkpoint_sha256"]:
            raise ValueError(f"frozen checkpoint SHA-256 mismatch for {spec['backbone_id']}")
        records.append({"backbone_id": spec["backbone_id"], "checkpoint_sha256": observed})
    return records


def _extract_dataset(
    spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], prepared_root: Path,
) -> np.ndarray:
    extractor = model_source._build_extractor(spec)
    chunks: list[np.ndarray] = []
    try:
        for start in range(0, len(rows), BATCH_SIZE):
            images = []
            for row in rows[start : start + BATCH_SIZE]:
                with Image.open(prepared_root / str(row["image_path"])) as source:
                    images.append(source.convert("RGB").copy())
            outputs = extractor.transform_many(images)
            selected = [row for row in outputs if row.name == spec["output"]]
            if len(selected) != 1:
                raise ValueError("extractor did not emit exactly one frozen output")
            chunk = np.asarray(selected[0].embeddings, dtype=np.float32)
            if chunk.shape != (len(images), int(spec["dimension"])) or not np.isfinite(chunk).all():
                raise ValueError(f"invalid embedding chunk for {spec['backbone_id']}: {chunk.shape}")
            chunks.append(np.ascontiguousarray(chunk))
    finally:
        for name in ("_model", "_processor", "_torch", "_preprocess", "_tokenizer"):
            if hasattr(extractor, name):
                setattr(extractor, name, None)
        gc.collect()
    matrix = np.concatenate(chunks, axis=0)
    if matrix.shape != (len(rows), int(spec["dimension"])):
        raise ValueError(f"invalid complete embedding matrix for {spec['backbone_id']}")
    return np.ascontiguousarray(matrix)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(prepare.canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_matrix(path: Path, matrix: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
    temporary.replace(path)


def _validate_cache(
    dataset_id: str, spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], output_root: Path,
) -> dict[str, Any]:
    root = output_root / dataset_id
    matrix_path, record_path = root / f"{spec['backbone_id']}.npy", root / f"{spec['backbone_id']}.json"
    if not matrix_path.is_file() or not record_path.is_file():
        raise FileNotFoundError(f"partial embedding cache for {dataset_id}/{spec['backbone_id']}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    checks = (
        record.get("dataset_id") == dataset_id,
        record.get("backbone_id") == spec["backbone_id"],
        record.get("spec_sha256") == prepare.sha256_payload(spec),
        record.get("sample_ids_sha256") == prepare.sha256_payload([row["sample_id"] for row in rows]),
        record.get("targets_sha256") == prepare.sha256_payload([row["target"] for row in rows]),
        record.get("matrix_sha256") == prepare.sha256_path(matrix_path),
        record.get("values_sha256") == _values_sha(np.asarray(matrix)),
        tuple(matrix.shape) == (len(rows), int(spec["dimension"])),
        matrix.dtype == np.dtype("float32"),
        bool(np.isfinite(np.asarray(matrix)).all()),
    )
    if not all(checks):
        raise ValueError(f"embedding cache validation failed for {dataset_id}/{spec['backbone_id']}")
    return record


def _git_identity() -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return {"commit": commit, "dirty": dirty}


def run_extraction(
    prepared_root: Path = prepare.DEFAULT_PREPARED_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *, resume: bool = False,
) -> dict[str, Any]:
    if type(resume) is not bool:
        raise TypeError("resume must be a strict bool")
    prepared_root, output_root = prepared_root.resolve(), output_root.resolve()
    protocol = prepare.load_protocol()
    git = _git_identity()
    if git["dirty"]:
        raise ValueError("embedding extraction requires a clean committed source tree")
    cohort = prepare.load_cohort(prepared_root)
    model_records = validate_model_inputs(protocol)
    terminal = output_root / "manifest.json"
    if terminal.exists():
        if not resume:
            raise FileExistsError("terminal embedding manifest exists")
        return validate_bank(prepared_root, output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset_id, dataset in cohort["datasets"].items():
        dataset_root = output_root / dataset_id
        dataset_root.mkdir(parents=True, exist_ok=True)
        rows = dataset["rows"]
        for spec in _protocol_specs(protocol):
            matrix_path = dataset_root / f"{spec['backbone_id']}.npy"
            record_path = dataset_root / f"{spec['backbone_id']}.json"
            if matrix_path.exists() or record_path.exists():
                if not resume:
                    raise FileExistsError(f"partial embedding cache exists for {dataset_id}/{spec['backbone_id']}")
                records.append(_validate_cache(dataset_id, spec, rows, output_root))
                continue
            matrix = _extract_dataset(spec, rows, prepared_root)
            _write_matrix(matrix_path, matrix)
            record = {
                "dataset_id": dataset_id, "backbone_id": spec["backbone_id"],
                "rows": len(rows), "dimension": int(matrix.shape[1]), "dtype": "float32",
                "sample_ids_sha256": prepare.sha256_payload([row["sample_id"] for row in rows]),
                "targets_sha256": prepare.sha256_payload([row["target"] for row in rows]),
                "spec": dict(spec), "spec_sha256": prepare.sha256_payload(spec),
                "matrix_sha256": prepare.sha256_path(matrix_path), "values_sha256": _values_sha(matrix),
            }
            _atomic_json(record_path, record)
            records.append(record)
            print(f"embedding {len(records)}/{protocol['expected_grid']['embedding_matrix_count']}: {dataset_id}/{spec['backbone_id']}", flush=True)
    environment = {
        "python": platform.python_version(), "numpy": np.__version__,
        "Pillow": importlib.metadata.version("Pillow"), "torch": importlib.metadata.version("torch"),
        "torchvision": importlib.metadata.version("torchvision"), "timm": importlib.metadata.version("timm"),
        "transformers": importlib.metadata.version("transformers"), "vertebrae": importlib.metadata.version("vertebrae"),
        "device": DEVICE, "batch_size": BATCH_SIZE,
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"), "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
    }
    manifest = {
        "schema_version": 1, "status": "completed_real_source_embedding_extraction",
        "protocol_sha256": prepare.sha256_path(prepare.PROTOCOL_PATH),
        "input_manifest_sha256": prepare.sha256_path(prepared_root / "manifest.json"),
        "cohort_sha256": prepare.sha256_path(prepared_root / "cohort.json"),
        "backbone_ids": list(protocol["backbones"]), "records": records,
        "counts": {"datasets": len(cohort["datasets"]), "matrices": len(records), "image_passes": sum(int(row["rows"]) for row in records)},
        "model_inputs": model_records, "environment": environment, "git": git,
        "outcomes_inspected": False, "overlapindex_fit_count": 0, "reserved_backbones_evaluated": [],
    }
    _atomic_json(terminal, manifest)
    return validate_bank(prepared_root, output_root)


def validate_bank(
    prepared_root: Path = prepare.DEFAULT_PREPARED_ROOT, output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    protocol = prepare.load_protocol()
    cohort = prepare.load_cohort(prepared_root)
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_real_source_embedding_extraction":
        raise ValueError("embedding manifest status changed")
    if manifest.get("protocol_sha256") != prepare.sha256_path(prepare.PROTOCOL_PATH):
        raise ValueError("embedding protocol identity changed")
    if manifest.get("input_manifest_sha256") != prepare.sha256_path(prepared_root / "manifest.json"):
        raise ValueError("embedding input identity changed")
    current_git = _git_identity()
    if current_git["dirty"] or manifest.get("git") != current_git:
        raise ValueError("embedding source commit identity changed")
    observed = []
    for dataset_id, dataset in cohort["datasets"].items():
        for spec in _protocol_specs(protocol):
            observed.append(_validate_cache(dataset_id, spec, dataset["rows"], output_root))
    if len(observed) != int(protocol["expected_grid"]["embedding_matrix_count"]):
        raise ValueError("embedding matrix count changed")
    if observed != manifest["records"]:
        raise ValueError("embedding record order or identity changed")
    return manifest


def load_bank(
    prepared_root: Path = prepare.DEFAULT_PREPARED_ROOT, output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    validate_bank(prepared_root, output_root)
    cohort = prepare.load_cohort(prepared_root)
    protocol = prepare.load_protocol()
    matrices = {
        dataset_id: {
            backbone: np.load(output_root / dataset_id / f"{backbone}.npy", mmap_mode="r", allow_pickle=False)
            for backbone in protocol["backbones"]
        }
        for dataset_id in cohort["datasets"]
    }
    return cohort, matrices


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, default=prepare.DEFAULT_PREPARED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    result = validate_bank(args.prepared_root, args.output_root) if args.validate_only else run_extraction(args.prepared_root, args.output_root, resume=args.resume)
    print(prepare.canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
