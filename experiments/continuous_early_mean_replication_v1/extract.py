"""Extract the frozen five-backbone early-mean representation bank."""

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

from experiments.continuous_cached_embedding_information_audit_v1 import run as information
from experiments.continuous_intermediate_pooling_audit_v1 import extract as parent_extract
from experiments.continuous_ranking_benchmark_v2 import run as base
from experiments.continuous_real_age_panel import extract as model_source
from experiments.continuous_real_source_qualification_v1 import prepare


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_early_mean_replication_v1" / "embeddings"
PARENT_EMBEDDING_ROOT = ROOT / "artifacts" / "continuous_intermediate_pooling_audit_v1" / "embeddings"
PARENT_AUDIT_ROOT = ROOT / "artifacts" / "continuous_intermediate_pooling_audit_v1" / "audit"
SPECS = {str(row["backbone_id"]): row for row in model_source.BACKBONE_SPECS}


def load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or base.sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("early-mean replication protocol SHA-256 mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_continuous_early_mean_replication_v1_before_outcomes":
        raise ValueError("early-mean replication protocol is not frozen")
    return protocol


def _source_paths() -> dict[str, Path]:
    return {
        "prepared_cohort": prepare.DEFAULT_PREPARED_ROOT / "cohort.json",
        "prepared_manifest": prepare.DEFAULT_PREPARED_ROOT / "manifest.json",
        "intermediate_terminal": PARENT_AUDIT_ROOT / "terminal.json",
        "intermediate_summary": PARENT_AUDIT_ROOT / "summary.json",
        "intermediate_embedding_manifest": PARENT_EMBEDDING_ROOT / "manifest.json",
        "model_spec_source": Path(model_source.__file__).resolve(),
    }


def _source_hashes(protocol: Mapping[str, Any]) -> dict[str, str]:
    observed = {name: base.sha256_path(path) for name, path in _source_paths().items()}
    for name, value in observed.items():
        if value != protocol["ancestry"][f"{name}_sha256"]:
            raise ValueError(f"early-mean replication ancestry changed: {name}")
    return observed


def _rows(cohort: Mapping[str, Any], target: Mapping[str, Any]) -> tuple[str, list[Mapping[str, Any]]]:
    dataset_id, target_id = str(target["target_key"]).split("::", 1)
    rows = list(cohort["datasets"][dataset_id]["target_cohorts"][target_id])
    training = [row for row in rows if row["split"] == "training"]
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    expected_train = int(target["training_pool_rows"])
    expected_eval = int(target["evaluation_rows"])
    if (
        len(training) != expected_train
        or len(evaluation) != expected_eval
        or rows != training + evaluation
        or len({str(row["sample_id"]) for row in rows}) != len(rows)
        or [int(row["matrix_index"]) for row in rows] != list(range(len(rows)))
    ):
        raise ValueError(f"early-mean prepared row order changed: {target['target_key']}")
    return dataset_id, rows


def _images(rows: Sequence[Mapping[str, Any]]) -> list[Image.Image]:
    images = []
    for row in rows:
        with Image.open(prepare.DEFAULT_PREPARED_ROOT / str(row["image_path"])) as source:
            images.append(source.convert("RGB").copy())
    return images


def _extract_huggingface(
    config: Mapping[str, Any], spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], batch_size: int,
) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        str(spec["model_id"]), revision=str(spec["revision"]), local_files_only=True, use_fast=False
    )
    model_kwargs = {"add_pooling_layer": False} if spec["backbone_id"] == "deit-tiny" else {}
    model = AutoModel.from_pretrained(
        str(spec["model_id"]), revision=str(spec["revision"]), local_files_only=True, **model_kwargs
    ).to("cpu").eval()
    chunks = []
    try:
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                encoded = processor(images=_images(rows[start : start + batch_size]), return_tensors="pt")
                output = model(**encoded, output_hidden_states=True)
                tokens = output.hidden_states[int(config["hidden_layer"])][:, int(config["special_tokens"]) :, :]
                chunk = tokens.mean(dim=1).detach().cpu().numpy().astype(np.float32, copy=False)
                chunks.append(np.ascontiguousarray(chunk))
    finally:
        del model, processor
        gc.collect()
    return np.ascontiguousarray(np.vstack(chunks))


def _extract_timm(
    config: Mapping[str, Any], spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], batch_size: int,
) -> np.ndarray:
    import timm
    import torch

    model = timm.create_model(str(spec["model_id"]), pretrained=True, num_classes=0).to("cpu").eval()
    transform = timm.data.create_transform(**timm.data.resolve_model_data_config(model))
    index = int(config["feature_index"])
    chunks = []
    try:
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                batch = torch.stack([transform(image) for image in _images(rows[start : start + batch_size])])
                _final, stages = model.forward_intermediates(batch, indices=[index], norm=False, output_fmt="NCHW")
                if len(stages) != 1:
                    raise ValueError("early-mean timm extraction did not return exactly one stage")
                chunk = stages[0].flatten(start_dim=2).mean(dim=2).detach().cpu().numpy().astype(np.float32, copy=False)
                chunks.append(np.ascontiguousarray(chunk))
    finally:
        del model
        gc.collect()
    return np.ascontiguousarray(np.vstack(chunks))


def _write_matrix(path: Path, matrix: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
    temporary.replace(path)


def _values_sha(matrix: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(matrix).tobytes(order="C")).hexdigest()


def _record_path(output_root: Path, dataset_id: str, backbone: str) -> Path:
    return output_root / dataset_id / f"{backbone}.json"


def _matrix_path(output_root: Path, dataset_id: str, backbone: str) -> Path:
    return output_root / dataset_id / f"{backbone}.npy"


def _parent_parity(
    protocol: Mapping[str, Any], cohort: Mapping[str, Any], target: Mapping[str, Any], backbone: str, matrix: np.ndarray,
) -> dict[str, Any] | None:
    if backbone not in {"dinov2-small", "resnet50", "convnext-tiny"}:
        return None
    dataset_id, draw, _pool, evaluation = information._target_rows(
        cohort,
        {
            **dict(target),
            "draw_seed": int(target["draw_seeds"][0]),
            "archived_training_rows": int(target["training_rows"]),
        },
    )
    parent = np.load(
        PARENT_EMBEDDING_ROOT / dataset_id / backbone / "early_mean.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    selected = list(draw) + list(evaluation)
    indices = np.asarray([int(row["matrix_index"]) for row in selected], dtype=np.int64)
    if tuple(parent.shape) != (len(selected), matrix.shape[1]):
        raise ValueError(f"parent early-mean matrix structure changed: {dataset_id}/{backbone}")
    delta = float(np.max(np.abs(np.asarray(parent, dtype=np.float32) - matrix[indices])))
    if delta > float(protocol["extraction"]["existing_three_backbone_overlap_parity_maximum_absolute_delta"]):
        raise ValueError(f"early-mean overlap parity failed: {dataset_id}/{backbone}: {delta}")
    return {
        "parent_rows": len(selected),
        "parent_sample_ids_sha256": base.sha256_payload([str(row["sample_id"]) for row in selected]),
        "maximum_absolute_delta": delta,
    }


def _validate_record(output_root: Path, record: Mapping[str, Any]) -> None:
    path = _matrix_path(output_root, str(record["dataset_id"]), str(record["backbone_id"]))
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        base.sha256_path(path) != record["matrix_sha256"]
        or tuple(matrix.shape) != (int(record["rows"]), int(record["dimension"]))
        or matrix.dtype != np.dtype("float32")
        or not np.isfinite(np.asarray(matrix)).all()
    ):
        raise ValueError(f"early-mean matrix identity changed: {record['dataset_id']}/{record['backbone_id']}")


def run_extraction(output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False) -> dict[str, Any]:
    if type(resume) is not bool:
        raise TypeError("resume must be a strict bool")
    output_root = output_root.resolve()
    protocol = load_protocol()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("early-mean extraction requires a clean committed source tree")
    source_hashes = _source_hashes(protocol)
    terminal_path = output_root / "manifest.json"
    if terminal_path.exists():
        if not resume:
            raise FileExistsError("early-mean embedding manifest exists; use --resume or --validate-only")
        return validate_bank(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(int(protocol["extraction"]["torch_intraop_threads"]))
    torch.set_num_interop_threads(int(protocol["extraction"]["torch_interop_threads"]))
    cohort = prepare.load_cohort()
    records = []
    for config in protocol["backbones"]:
        backbone = str(config["backbone_id"])
        spec = SPECS[backbone]
        checkpoint = model_source._checkpoint_path(spec)
        if not checkpoint.is_file() or base.sha256_path(checkpoint) != spec["checkpoint_sha256"]:
            raise ValueError(f"frozen checkpoint identity changed: {backbone}")
    expected_records = int(protocol["expected_grid"]["embedding_matrix_count"])
    for target in protocol["targets"]:
        dataset_id, rows = _rows(cohort, target)
        for config in protocol["backbones"]:
            backbone = str(config["backbone_id"])
            record_path = _record_path(output_root, dataset_id, backbone)
            matrix_path = _matrix_path(output_root, dataset_id, backbone)
            if record_path.exists() or matrix_path.exists():
                if not (resume and record_path.is_file() and matrix_path.is_file()):
                    raise FileExistsError(f"partial early-mean cache exists: {dataset_id}/{backbone}")
                record = json.loads(record_path.read_text(encoding="utf-8"))
                _validate_record(output_root, record)
                records.append(record)
                continue
            spec = SPECS[backbone]
            batch_size = int(protocol["extraction"]["batch_size"])
            matrix = (
                _extract_huggingface(config, spec, rows, batch_size)
                if config["provider"] == "huggingface"
                else _extract_timm(config, spec, rows, batch_size)
            )
            expected_shape = (len(rows), int(config["channels"]))
            if matrix.shape != expected_shape or matrix.dtype != np.dtype("float32") or not np.isfinite(matrix).all():
                raise ValueError(f"invalid early-mean matrix: {dataset_id}/{backbone}: {matrix.shape}")
            parity = _parent_parity(protocol, cohort, target, backbone, matrix)
            matrix_path.parent.mkdir(parents=True, exist_ok=True)
            _write_matrix(matrix_path, matrix)
            record = {
                "dataset_id": dataset_id,
                "target_key": str(target["target_key"]),
                "backbone_id": backbone,
                "rows": len(rows),
                "dimension": int(matrix.shape[1]),
                "dtype": "float32",
                "sample_ids_sha256": base.sha256_payload([str(row["sample_id"]) for row in rows]),
                "targets_sha256": base.sha256_payload([float(row["target"]) for row in rows]),
                "model_spec_sha256": base.sha256_payload(spec),
                "matrix_sha256": base.sha256_path(matrix_path),
                "values_sha256": _values_sha(matrix),
                "parent_overlap_parity": parity,
            }
            base._atomic_json(record_path, record)
            records.append(record)
            print(f"embedding {len(records)}/{expected_records}: {dataset_id}/{backbone}", flush=True)
    if len(records) != expected_records:
        raise ValueError("early-mean embedding record count changed")
    manifest = {
        "schema_version": 1,
        "status": "completed_continuous_early_mean_replication_extraction_v1",
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH),
        "source_hashes": source_hashes,
        "records": records,
        "counts": {
            "targets": len(protocol["targets"]),
            "backbones": len(protocol["backbones"]),
            "matrices": len(records),
            "image_passes": sum(int(row["rows"]) for row in records),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "Pillow": importlib.metadata.version("Pillow"),
            "torch": importlib.metadata.version("torch"),
            "timm": importlib.metadata.version("timm"),
            "transformers": importlib.metadata.version("transformers"),
            "device": "cpu",
            "torch_intraop_threads": int(torch.get_num_threads()),
            "torch_interop_threads": int(torch.get_num_interop_threads()),
            "thread_environment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
            },
        },
        "git": {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "dirty": False,
        },
        "outcomes_inspected": False,
        "overlapindex_fit_count": 0,
        "reserved_backbones_evaluated": [],
    }
    base._atomic_json(terminal_path, manifest)
    return validate_bank(output_root)


def validate_bank(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    protocol = load_protocol()
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed_continuous_early_mean_replication_extraction_v1"
        or manifest.get("protocol_sha256") != base.sha256_path(PROTOCOL_PATH)
        or manifest.get("source_hashes") != _source_hashes(protocol)
        or int(manifest.get("counts", {}).get("matrices", -1)) != int(protocol["expected_grid"]["embedding_matrix_count"])
        or int(manifest.get("counts", {}).get("image_passes", -1)) != 30000
    ):
        raise ValueError("early-mean embedding manifest changed")
    records = manifest.get("records", [])
    if len(records) != int(protocol["expected_grid"]["embedding_matrix_count"]):
        raise ValueError("early-mean embedding record grid changed")
    for record in records:
        _validate_record(output_root, record)
    return manifest


def load_bank(output_root: Path = DEFAULT_OUTPUT_ROOT) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    manifest = validate_bank(output_root)
    matrices: dict[str, dict[str, np.ndarray]] = {}
    for record in manifest["records"]:
        dataset_id, backbone = str(record["dataset_id"]), str(record["backbone_id"])
        matrices.setdefault(dataset_id, {})[backbone] = np.load(
            _matrix_path(output_root, dataset_id, backbone), mmap_mode="r", allow_pickle=False
        )
    return manifest, matrices


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    result = validate_bank(args.output_root) if args.validate_only else run_extraction(args.output_root, resume=args.resume)
    print(base.canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
