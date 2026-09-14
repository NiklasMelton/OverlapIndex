"""Extract frozen intermediate-layer pooling views for the continuous audit."""

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
from experiments.continuous_cached_embedding_smooth_boundary_v1 import run as boundary
from experiments.continuous_ranking_benchmark_v2 import run as base
from experiments.continuous_real_age_panel import extract as model_source
from experiments.continuous_real_source_qualification_v1 import extract as terminal_source
from experiments.continuous_real_source_qualification_v1 import prepare


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_intermediate_pooling_audit_v1" / "embeddings"
BOUNDARY_ROOT = ROOT / "artifacts" / "continuous_cached_embedding_smooth_boundary_v1" / "audit"
MODEL_SPEC_PATH = Path(model_source.__file__).resolve()
SPECS = {str(row["backbone_id"]): row for row in model_source.BACKBONE_SPECS}


def load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or base.sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("intermediate-pooling protocol SHA-256 mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_continuous_intermediate_pooling_audit_v1_before_outcomes":
        raise ValueError("intermediate-pooling protocol is not frozen")
    return protocol


def _source_hashes(protocol: Mapping[str, Any]) -> dict[str, str]:
    paths = {
        "boundary_terminal": BOUNDARY_ROOT / "terminal.json",
        "boundary_manifest": BOUNDARY_ROOT / "manifest.json",
        "boundary_summary": BOUNDARY_ROOT / "summary.json",
        "model_spec_source": MODEL_SPEC_PATH,
        "prepared_cohort": prepare.DEFAULT_PREPARED_ROOT / "cohort.json",
        "prepared_manifest": prepare.DEFAULT_PREPARED_ROOT / "manifest.json",
    }
    observed = {name: base.sha256_path(path) for name, path in paths.items()}
    if any(observed[name] != protocol["ancestry"][f"{name}_sha256"] for name in observed):
        raise ValueError("intermediate-pooling ancestry changed")
    terminal = boundary.validate_artifacts(BOUNDARY_ROOT)
    if terminal["decision"]["material_rank_conflict"]:
        raise ValueError("intermediate-pooling audit requires a resolved terminal-view nonconflict")
    return observed


def _selected_rows(protocol: Mapping[str, Any], cohort: Mapping[str, Any], target: Mapping[str, Any]) -> tuple[str, list[Mapping[str, Any]]]:
    dataset_id, draw, _, evaluation = information._target_rows(cohort, {
        **target, "archived_training_rows": int(target["training_rows"])
    })
    rows = list(draw) + list(evaluation)
    expected = int(protocol["extraction"]["selected_rows_per_target"])
    if len(rows) != expected or len({str(row["sample_id"]) for row in rows}) != expected:
        raise ValueError(f"intermediate-pooling selected rows changed: {target['target_key']}")
    return dataset_id, rows


def _pool_tokens(values: Any, torch: Any) -> dict[str, np.ndarray]:
    mean = values.mean(dim=1)
    standard_deviation = values.std(dim=1, unbiased=False)
    return {
        "mean": mean.detach().cpu().numpy().astype(np.float32, copy=False),
        "mean_std": torch.cat((mean, standard_deviation), dim=1).detach().cpu().numpy().astype(np.float32, copy=False),
    }


def _pool_spatial(values: Any, torch: Any) -> dict[str, np.ndarray]:
    flattened = values.flatten(start_dim=2)
    mean = flattened.mean(dim=2)
    standard_deviation = flattened.std(dim=2, unbiased=False)
    return {
        "mean": mean.detach().cpu().numpy().astype(np.float32, copy=False),
        "mean_std": torch.cat((mean, standard_deviation), dim=1).detach().cpu().numpy().astype(np.float32, copy=False),
    }


def _images(rows: Sequence[Mapping[str, Any]]) -> list[Image.Image]:
    images = []
    for row in rows:
        with Image.open(prepare.DEFAULT_PREPARED_ROOT / str(row["image_path"])) as source:
            images.append(source.convert("RGB").copy())
    return images


def _extract_dino(config: Mapping[str, Any], spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        str(spec["model_id"]), revision=str(spec["revision"]), local_files_only=True, use_fast=False
    )
    model = AutoModel.from_pretrained(
        str(spec["model_id"]), revision=str(spec["revision"]), local_files_only=True
    ).to("cpu").eval()
    output = {f"{depth['depth_id']}_{pool['pooling_id']}": [] for depth in config["relative_depths"] for pool in load_protocol()["pooling"]}
    terminal = []
    batch_size = int(load_protocol()["extraction"]["batch_size"])
    try:
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                images = _images(rows[start : start + batch_size])
                encoded = processor(images=images, return_tensors="pt")
                model_output = model(**encoded, output_hidden_states=True)
                for depth in config["relative_depths"]:
                    tokens = model_output.hidden_states[int(depth["hidden_layer"])][:, 1:, :]
                    for pooling_id, values in _pool_tokens(tokens, torch).items():
                        output[f"{depth['depth_id']}_{pooling_id}"].append(values)
                terminal.append(model_output.hidden_states[-1][:, 0, :].detach().cpu().numpy().astype(np.float32, copy=False))
    finally:
        del model, processor
        gc.collect()
    return {name: np.ascontiguousarray(np.vstack(values)) for name, values in output.items()}, np.ascontiguousarray(np.vstack(terminal))


def _extract_timm(config: Mapping[str, Any], spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    import timm
    import torch

    model = timm.create_model(str(spec["model_id"]), pretrained=True, num_classes=0).to("cpu").eval()
    transform = timm.data.create_transform(**timm.data.resolve_model_data_config(model))
    indices = [int(depth["feature_index"]) for depth in config["relative_depths"]]
    output = {f"{depth['depth_id']}_{pool['pooling_id']}": [] for depth in config["relative_depths"] for pool in load_protocol()["pooling"]}
    terminal = []
    batch_size = int(load_protocol()["extraction"]["batch_size"])
    try:
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                images = _images(rows[start : start + batch_size])
                batch = torch.stack([transform(image) for image in images])
                final, stages = model.forward_intermediates(batch, indices=indices, norm=False, output_fmt="NCHW")
                for depth, stage in zip(config["relative_depths"], stages):
                    for pooling_id, values in _pool_spatial(stage, torch).items():
                        output[f"{depth['depth_id']}_{pooling_id}"].append(values)
                terminal.append(model.forward_head(final, pre_logits=True).detach().cpu().numpy().astype(np.float32, copy=False))
    finally:
        del model
        gc.collect()
    return {name: np.ascontiguousarray(np.vstack(values)) for name, values in output.items()}, np.ascontiguousarray(np.vstack(terminal))


def _values_sha(matrix: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(matrix).tobytes(order="C")).hexdigest()


def _write_matrix(path: Path, matrix: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
    temporary.replace(path)


def _record_path(output_root: Path, dataset_id: str, backbone: str) -> Path:
    return output_root / dataset_id / f"{backbone}.json"


def _validate_record(output_root: Path, record: Mapping[str, Any]) -> None:
    dataset_id, backbone = str(record["dataset_id"]), str(record["backbone_id"])
    for output in record["outputs"]:
        path = output_root / dataset_id / backbone / f"{output['view_id']}.npy"
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        if base.sha256_path(path) != output["matrix_sha256"] or tuple(matrix.shape) != (int(record["rows"]), int(output["dimension"])) or matrix.dtype != np.dtype("float32"):
            raise ValueError(f"intermediate matrix identity changed: {dataset_id}/{backbone}/{output['view_id']}")


def run_extraction(output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False) -> dict[str, Any]:
    output_root = output_root.resolve(); protocol = load_protocol()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("intermediate-pooling extraction requires a clean committed source tree")
    source_hashes = _source_hashes(protocol); terminal_path = output_root / "manifest.json"
    if terminal_path.exists():
        if not resume: raise FileExistsError("intermediate-pooling embedding manifest exists; use --resume or --validate-only")
        return validate_bank(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(int(protocol["extraction"]["torch_intraop_threads"]))
    torch.set_num_interop_threads(int(protocol["extraction"]["torch_interop_threads"]))
    cohort = prepare.load_cohort(); records = []
    for config in protocol["backbones"]:
        backbone = str(config["backbone_id"]); spec = SPECS[backbone]
        checkpoint = model_source._checkpoint_path(spec)
        if not checkpoint.is_file() or base.sha256_path(checkpoint) != spec["checkpoint_sha256"]:
            raise ValueError(f"frozen model checkpoint identity changed: {backbone}")
    for target in protocol["targets"]:
        dataset_id, rows = _selected_rows(protocol, cohort, target)
        for config in protocol["backbones"]:
            backbone = str(config["backbone_id"]); record_path = _record_path(output_root, dataset_id, backbone)
            root = output_root / dataset_id / backbone
            if record_path.exists():
                if not resume: raise FileExistsError(f"partial intermediate cache exists: {dataset_id}/{backbone}")
                record = json.loads(record_path.read_text(encoding="utf-8")); _validate_record(output_root, record); records.append(record); continue
            if root.exists() and any(root.iterdir()):
                raise FileExistsError(f"uncommitted partial intermediate matrices exist: {dataset_id}/{backbone}")
            spec = SPECS[backbone]
            matrices, terminal = (_extract_dino(config, spec, rows) if config["provider"] == "huggingface" else _extract_timm(config, spec, rows))
            expected_dimensions = {
                f"{depth['depth_id']}_{pool['pooling_id']}": int(depth["channels"]) * int(pool["dimension_multiplier"])
                for depth in config["relative_depths"] for pool in protocol["pooling"]
            }
            if set(matrices) != set(expected_dimensions):
                raise ValueError(f"intermediate output roster changed: {dataset_id}/{backbone}")
            for view_id, matrix in matrices.items():
                if matrix.shape != (len(rows), expected_dimensions[view_id]) or matrix.dtype != np.dtype("float32") or not np.isfinite(matrix).all():
                    raise ValueError(f"intermediate output structure changed: {dataset_id}/{backbone}/{view_id}")
            expected_terminal = np.load(terminal_source.DEFAULT_OUTPUT_ROOT / dataset_id / f"{backbone}.npy", mmap_mode="r", allow_pickle=False)[np.asarray([int(row["matrix_index"]) for row in rows], dtype=np.int64)]
            parity_delta = float(np.max(np.abs(terminal - expected_terminal)))
            if parity_delta > float(protocol["cached_final_control"]["parity_maximum_absolute_delta"]):
                raise ValueError(f"intermediate extraction terminal parity failed: {dataset_id}/{backbone}: {parity_delta}")
            root.mkdir(parents=True, exist_ok=True)
            outputs = []
            for view_id, matrix in matrices.items():
                path = root / f"{view_id}.npy"; _write_matrix(path, matrix)
                outputs.append({"view_id": view_id, "dimension": int(matrix.shape[1]), "matrix_sha256": base.sha256_path(path), "values_sha256": _values_sha(matrix)})
            record = {
                "dataset_id": dataset_id, "target_key": str(target["target_key"]), "backbone_id": backbone,
                "rows": len(rows), "sample_ids_sha256": base.sha256_payload([str(row["sample_id"]) for row in rows]),
                "model_spec_sha256": base.sha256_payload(spec), "terminal_parity_maximum_absolute_delta": parity_delta,
                "outputs": outputs,
            }
            base._atomic_json(record_path, record); records.append(record)
            print(f"extraction {len(records)}/{len(protocol['targets']) * len(protocol['backbones'])}: {dataset_id}/{backbone}", flush=True)
    if len(records) != len(protocol["targets"]) * len(protocol["backbones"]):
        raise ValueError("intermediate extraction record count changed")
    manifest = {
        "schema_version": 1, "status": "completed_continuous_intermediate_pooling_extraction_v1",
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH), "source_hashes": source_hashes,
        "records": records,
        "counts": {"targets": len(protocol["targets"]), "backbones": len(protocol["backbones"]), "matrices": sum(len(row["outputs"]) for row in records)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "Pillow": importlib.metadata.version("Pillow"), "torch": importlib.metadata.version("torch"), "timm": importlib.metadata.version("timm"), "transformers": importlib.metadata.version("transformers"), "device": "cpu", "torch_intraop_threads": int(torch.get_num_threads()), "torch_interop_threads": int(torch.get_num_interop_threads())},
        "git": {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "dirty": False},
        "outcomes_inspected": False,
        "overlapindex_fit_count": 0, "reserved_backbones_evaluated": [],
    }
    base._atomic_json(terminal_path, manifest); return validate_bank(output_root)


def validate_bank(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve(); protocol = load_protocol()
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_continuous_intermediate_pooling_extraction_v1" or manifest.get("protocol_sha256") != base.sha256_path(PROTOCOL_PATH):
        raise ValueError("intermediate-pooling extraction manifest changed")
    if manifest.get("source_hashes") != _source_hashes(protocol): raise ValueError("intermediate-pooling extraction ancestry changed")
    for record in manifest["records"]: _validate_record(output_root, record)
    if manifest["counts"]["matrices"] != int(protocol["expected_grid"]["extracted_matrix_count"]): raise ValueError("intermediate-pooling matrix count changed")
    return manifest


def load_bank(output_root: Path = DEFAULT_OUTPUT_ROOT) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, np.ndarray]]]]:
    manifest = validate_bank(output_root); matrices: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for record in manifest["records"]:
        dataset_id, backbone = str(record["dataset_id"]), str(record["backbone_id"])
        matrices.setdefault(dataset_id, {})[backbone] = {
            str(row["view_id"]): np.load(output_root / dataset_id / backbone / f"{row['view_id']}.npy", mmap_mode="r", allow_pickle=False)
            for row in record["outputs"]
        }
    return manifest, matrices


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv); result = validate_bank(args.output_root) if args.validate_only else run_extraction(args.output_root, resume=args.resume)
    print(base.canonical_json(result)); return 0


if __name__ == "__main__": raise SystemExit(main())
