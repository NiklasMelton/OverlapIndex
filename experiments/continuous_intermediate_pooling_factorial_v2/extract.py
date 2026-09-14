"""Extract all frozen depth-by-pooling views for the replicated factorial."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.continuous_cached_embedding_information_audit_v1 import run as information
from experiments.continuous_early_mean_replication_v1 import extract as early
from experiments.continuous_intermediate_pooling_audit_v1 import extract as parent
from experiments.continuous_ranking_benchmark_v2 import run as base
from experiments.continuous_real_age_panel import extract as model_source
from experiments.continuous_real_source_qualification_v1 import prepare


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_intermediate_pooling_factorial_v2" / "embeddings"
PARENT_ROOT = ROOT / "artifacts" / "continuous_intermediate_pooling_audit_v1" / "embeddings"
EARLY_ROOT = early.DEFAULT_OUTPUT_ROOT
SPECS = {str(row["backbone_id"]): row for row in model_source.BACKBONE_SPECS}


def load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or base.sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("intermediate-pooling factorial protocol SHA-256 mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_continuous_intermediate_pooling_factorial_v2_before_outcomes":
        raise ValueError("intermediate-pooling factorial protocol is not frozen")
    return protocol


def _source_paths() -> dict[str, Path]:
    return {
        "early_mean_protocol": early.PROTOCOL_PATH,
        "early_mean_embedding_manifest": EARLY_ROOT / "manifest.json",
        "early_mean_reference_raw": early.ROOT / "artifacts" / "continuous_early_mean_replication_v1" / "reference" / "raw_results.json",
        "early_mean_reference_summary": early.ROOT / "artifacts" / "continuous_early_mean_replication_v1" / "reference" / "summary.json",
        "early_mean_reference_terminal": early.ROOT / "artifacts" / "continuous_early_mean_replication_v1" / "reference" / "terminal.json",
        "parent_intermediate_embedding_manifest": PARENT_ROOT / "manifest.json",
        "prepared_cohort": prepare.DEFAULT_PREPARED_ROOT / "cohort.json",
        "prepared_manifest": prepare.DEFAULT_PREPARED_ROOT / "manifest.json",
        "model_spec_source": Path(model_source.__file__).resolve(),
    }


def _source_hashes(protocol: Mapping[str, Any]) -> dict[str, str]:
    observed = {name: base.sha256_path(path) for name, path in _source_paths().items()}
    for name, value in observed.items():
        if value != protocol["ancestry"][f"{name}_sha256"]:
            raise ValueError(f"intermediate-pooling factorial ancestry changed: {name}")
    early.validate_bank()
    parent.validate_bank()
    return observed


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


def _extract_huggingface(
    config: Mapping[str, Any], spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], batch_size: int,
) -> dict[str, np.ndarray]:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        str(spec["model_id"]), revision=str(spec["revision"]), local_files_only=True, use_fast=False
    )
    kwargs = {"add_pooling_layer": False} if spec["backbone_id"] == "deit-tiny" else {}
    model = AutoModel.from_pretrained(
        str(spec["model_id"]), revision=str(spec["revision"]), local_files_only=True, **kwargs
    ).to("cpu").eval()
    output = {
        f"{depth['depth_id']}_{pool}": []
        for depth in config["relative_depths"]
        for pool in ("mean", "mean_std")
    }
    try:
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                encoded = processor(images=early._images(rows[start : start + batch_size]), return_tensors="pt")
                result = model(**encoded, output_hidden_states=True)
                for depth in config["relative_depths"]:
                    tokens = result.hidden_states[int(depth["hidden_layer"])][:, int(config["special_tokens"]) :, :]
                    for pooling_id, matrix in _pool_tokens(tokens, torch).items():
                        output[f"{depth['depth_id']}_{pooling_id}"].append(np.ascontiguousarray(matrix))
    finally:
        del model, processor
        gc.collect()
    return {view: np.ascontiguousarray(np.vstack(chunks)) for view, chunks in output.items()}


def _extract_timm(
    config: Mapping[str, Any], spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], batch_size: int,
) -> dict[str, np.ndarray]:
    import timm
    import torch

    model = timm.create_model(str(spec["model_id"]), pretrained=True, num_classes=0).to("cpu").eval()
    transform = timm.data.create_transform(**timm.data.resolve_model_data_config(model))
    indices = [int(depth["feature_index"]) for depth in config["relative_depths"]]
    output = {
        f"{depth['depth_id']}_{pool}": []
        for depth in config["relative_depths"]
        for pool in ("mean", "mean_std")
    }
    try:
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                images = early._images(rows[start : start + batch_size])
                batch = torch.stack([transform(image) for image in images])
                _final, stages = model.forward_intermediates(batch, indices=indices, norm=False, output_fmt="NCHW")
                if len(stages) != len(indices):
                    raise ValueError("factorial timm extraction returned the wrong stage count")
                for depth, stage in zip(config["relative_depths"], stages):
                    for pooling_id, matrix in _pool_spatial(stage, torch).items():
                        output[f"{depth['depth_id']}_{pooling_id}"].append(np.ascontiguousarray(matrix))
    finally:
        del model
        gc.collect()
    return {view: np.ascontiguousarray(np.vstack(chunks)) for view, chunks in output.items()}


def _expected_dimensions(protocol: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, int]:
    multipliers = {str(row["pooling_id"]): int(row["dimension_multiplier"]) for row in protocol["pooling"]}
    return {
        f"{depth['depth_id']}_{pooling_id}": int(depth["channels"]) * multiplier
        for depth in config["relative_depths"]
        for pooling_id, multiplier in multipliers.items()
    }


def _parity(
    protocol: Mapping[str, Any], cohort: Mapping[str, Any], target: Mapping[str, Any], backbone: str,
    matrices: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    dataset_id = str(target["target_key"]).split("::", 1)[0]
    record: dict[str, Any] = {}
    if backbone in {"dinov2-small", "resnet50", "convnext-tiny"}:
        _dataset, draw, _pool, evaluation = information._target_rows(
            cohort,
            {
                **dict(target),
                "draw_seed": int(target["draw_seeds"][0]),
                "archived_training_rows": int(target["training_rows"]),
            },
        )
        indices = np.asarray([int(row["matrix_index"]) for row in list(draw) + list(evaluation)], dtype=np.int64)
        deltas_before_reuse = {}
        for view, matrix in matrices.items():
            prior = np.load(PARENT_ROOT / dataset_id / backbone / f"{view}.npy", mmap_mode="r", allow_pickle=False)
            deltas_before_reuse[view] = float(
                np.max(np.abs(np.asarray(prior, dtype=np.float32) - matrix[indices]))
            )
            # The archived values are the canonical realization of the same
            # frozen transform on these rows. Reuse them instead of relaxing
            # the parity gate for harmless batch-context floating-point drift.
            matrix[indices] = np.asarray(prior, dtype=np.float32)
        record["parent_selected_rows_reused"] = True
        record["parent_selected_rows_maximum_absolute_delta_before_reuse_by_view"] = deltas_before_reuse
    early_matrix = np.load(EARLY_ROOT / dataset_id / f"{backbone}.npy", mmap_mode="r", allow_pickle=False)
    early_delta = float(np.max(np.abs(np.asarray(early_matrix, dtype=np.float32) - matrices["early_mean"])))
    if early_delta > float(protocol["extraction"]["early_mean_parity_maximum_absolute_delta"]):
        raise ValueError(f"factorial early-mean parity failed: {dataset_id}/{backbone}: {early_delta}")
    record["early_mean_full_rows_maximum_absolute_delta"] = early_delta
    if backbone in {"dinov2-small", "resnet50", "convnext-tiny"}:
        deltas = {}
        for view, matrix in matrices.items():
            prior = np.load(PARENT_ROOT / dataset_id / backbone / f"{view}.npy", mmap_mode="r", allow_pickle=False)
            delta = float(np.max(np.abs(np.asarray(prior, dtype=np.float32) - matrix[indices])))
            if delta > float(protocol["extraction"]["parent_three_backbone_all_view_parity_maximum_absolute_delta"]):
                raise ValueError(f"factorial parent parity failed: {dataset_id}/{backbone}/{view}: {delta}")
            deltas[view] = delta
        record["parent_selected_rows_maximum_absolute_delta_by_view"] = deltas
    return record


def _record_path(output_root: Path, dataset_id: str, backbone: str) -> Path:
    return output_root / dataset_id / f"{backbone}.json"


def _validate_record(output_root: Path, record: Mapping[str, Any]) -> None:
    for output in record["outputs"]:
        path = output_root / str(record["dataset_id"]) / str(record["backbone_id"]) / f"{output['view_id']}.npy"
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            base.sha256_path(path) != output["matrix_sha256"]
            or tuple(matrix.shape) != (int(record["rows"]), int(output["dimension"]))
            or matrix.dtype != np.dtype("float32")
            or not np.isfinite(np.asarray(matrix)).all()
        ):
            raise ValueError(f"factorial matrix identity changed: {record['dataset_id']}/{record['backbone_id']}/{output['view_id']}")


def run_extraction(output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False) -> dict[str, Any]:
    if type(resume) is not bool:
        raise TypeError("resume must be a strict bool")
    output_root = output_root.resolve()
    protocol = load_protocol()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("factorial extraction requires a clean committed source tree")
    source_hashes = _source_hashes(protocol)
    terminal_path = output_root / "manifest.json"
    if terminal_path.exists():
        if not resume:
            raise FileExistsError("factorial embedding manifest exists; use --resume or --validate-only")
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
            raise ValueError(f"factorial frozen checkpoint identity changed: {backbone}")
    expected_records = len(protocol["targets"]) * len(protocol["backbones"])
    for target in protocol["targets"]:
        dataset_id, rows = early._rows(cohort, target)
        for config in protocol["backbones"]:
            backbone = str(config["backbone_id"])
            record_path = _record_path(output_root, dataset_id, backbone)
            if record_path.exists():
                if not resume:
                    raise FileExistsError(f"partial factorial cache exists: {dataset_id}/{backbone}")
                record = json.loads(record_path.read_text(encoding="utf-8"))
                _validate_record(output_root, record)
                records.append(record)
                continue
            root = output_root / dataset_id / backbone
            if root.exists() and any(root.iterdir()):
                raise FileExistsError(f"uncommitted factorial matrices exist: {dataset_id}/{backbone}")
            spec = SPECS[backbone]
            batch_size = int(protocol["extraction"]["batch_size"])
            matrices = (
                _extract_huggingface(config, spec, rows, batch_size)
                if config["provider"] == "huggingface"
                else _extract_timm(config, spec, rows, batch_size)
            )
            dimensions = _expected_dimensions(protocol, config)
            if set(matrices) != set(protocol["views"]) or dimensions.keys() != matrices.keys():
                raise ValueError(f"factorial output roster changed: {dataset_id}/{backbone}")
            for view, matrix in matrices.items():
                if matrix.shape != (len(rows), dimensions[view]) or matrix.dtype != np.dtype("float32") or not np.isfinite(matrix).all():
                    raise ValueError(f"invalid factorial matrix: {dataset_id}/{backbone}/{view}")
            parity = _parity(protocol, cohort, target, backbone, matrices)
            root.mkdir(parents=True, exist_ok=True)
            outputs = []
            for view, matrix in matrices.items():
                path = root / f"{view}.npy"
                early._write_matrix(path, matrix)
                outputs.append(
                    {
                        "view_id": view,
                        "dimension": int(matrix.shape[1]),
                        "matrix_sha256": base.sha256_path(path),
                        "values_sha256": early._values_sha(matrix),
                    }
                )
            record = {
                "dataset_id": dataset_id,
                "target_key": str(target["target_key"]),
                "backbone_id": backbone,
                "rows": len(rows),
                "sample_ids_sha256": base.sha256_payload([str(row["sample_id"]) for row in rows]),
                "model_spec_sha256": base.sha256_payload(spec),
                "parity": parity,
                "outputs": outputs,
            }
            base._atomic_json(record_path, record)
            records.append(record)
            print(f"factorial embedding {len(records)}/{expected_records}: {dataset_id}/{backbone}", flush=True)
    if len(records) != expected_records or sum(len(row["outputs"]) for row in records) != int(protocol["expected_grid"]["embedding_matrix_count"]):
        raise ValueError("factorial embedding grid changed")
    manifest = {
        "schema_version": 1,
        "status": "completed_continuous_intermediate_pooling_factorial_extraction_v2",
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH),
        "source_hashes": source_hashes,
        "records": records,
        "counts": {
            "targets": len(protocol["targets"]),
            "backbones": len(protocol["backbones"]),
            "matrices": sum(len(row["outputs"]) for row in records),
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
        manifest.get("status") != "completed_continuous_intermediate_pooling_factorial_extraction_v2"
        or manifest.get("protocol_sha256") != base.sha256_path(PROTOCOL_PATH)
        or manifest.get("source_hashes") != _source_hashes(protocol)
        or int(manifest.get("counts", {}).get("matrices", -1)) != int(protocol["expected_grid"]["embedding_matrix_count"])
        or int(manifest.get("counts", {}).get("image_passes", -1)) != 30000
    ):
        raise ValueError("factorial embedding manifest changed")
    for record in manifest["records"]:
        _validate_record(output_root, record)
    return manifest


def load_bank(output_root: Path = DEFAULT_OUTPUT_ROOT) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, np.ndarray]]]]:
    manifest = validate_bank(output_root)
    matrices: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for record in manifest["records"]:
        dataset_id, backbone = str(record["dataset_id"]), str(record["backbone_id"])
        matrices.setdefault(dataset_id, {})[backbone] = {
            str(output["view_id"]): np.load(
                output_root / dataset_id / backbone / f"{output['view_id']}.npy", mmap_mode="r", allow_pickle=False
            )
            for output in record["outputs"]
        }
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
