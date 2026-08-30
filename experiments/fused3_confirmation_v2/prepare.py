"""Prepare the immutable five-dataset input registry for confirmation V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import datasets, extractors, lineage, runner


def raw_roots(data_root: Path) -> dict[str, Path]:
    return {
        dataset_id: data_root / dataset_id for dataset_id in datasets.DATASET_IDS
    }


def write_lineage_lock(output: Path) -> dict[str, Any]:
    """Build and atomically seal the immutable design/code/authority lock."""

    runner.validate_protocol(require_frozen=True)
    sources = runner.source_identity()
    lock = lineage.build_lock_from_files(
        protocol_path=runner.PROTOCOL_PATH,
        protocol_sidecar_path=runner.PROTOCOL_SIDECAR,
        code_identity_sha256=str(sources["code_identity_sha256"]),
        candidate_authority_path=runner.CANDIDATE_AUTHORITY_PATH,
    )
    lineage.write_lock(output, lock)
    return lock


def validate_lineage_lock(path: Path) -> dict[str, Any]:
    """Require one lock to match the exact current frozen source identity."""

    protocol_sha, _ = runner.validate_protocol(require_frozen=True)
    sources = runner.source_identity()
    expected = lineage.build_lineage_from_files(
        protocol_path=runner.PROTOCOL_PATH,
        protocol_sidecar_path=runner.PROTOCOL_SIDECAR,
        code_identity_sha256=str(sources["code_identity_sha256"]),
    )
    value = lineage.read_json_object(path)
    validated = lineage.validate_lock_envelope(
        value,
        expected_lineage=expected,
        expected_candidate_authority_sha256=lineage.authority_sha256(
            runner.CANDIDATE_AUTHORITY_PATH
        ),
    )
    if validated["lineage"]["protocol_sha256"] != protocol_sha:
        raise ValueError("lineage lock protocol differs from current frozen protocol")
    return validated


def prepare_registry(
    *,
    data_root: Path,
    input_root: Path,
    lineage_lock_path: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Hash raw inputs, extract the frozen union, and write one registry."""

    lock = validate_lineage_lock(lineage_lock_path)
    roots = raw_roots(data_root)
    source_metadata = extractors.source_metadata_from_roots(
        roots, manifest_dir=input_root / "source_manifests"
    )
    specs = extractors.resolved_local_backbone_specs()
    model_lock = extractors.model_weight_lock_from_specs(specs)
    environment = extractors.observed_extraction_environment(
        device="cpu", batch_size=16
    )
    return extractors.prepare_inputs(
        raw_roots=roots,
        output_dir=input_root / "embeddings",
        registry_path=input_root / "audited_registry.json",
        protocol_sha256=str(lock["lineage"]["protocol_sha256"]),
        lineage_lock_sha256=lineage.sha256_path(lineage_lock_path),
        parent_registry_sha256=str(
            lock["lineage"]["v1_confirmation_registry_sha256"]
        ),
        extraction_environment=environment,
        model_weight_lock=model_lock,
        backbone_specs=specs,
        source_metadata=source_metadata,
        download=False,
        resume=resume,
        batch_size=16,
        device="cpu",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    lock = subparsers.add_parser("lock", help="write the immutable lineage lock")
    lock.add_argument("--output", type=Path, required=True)

    download = subparsers.add_parser(
        "download", help="explicitly download and validate the five raw datasets"
    )
    download.add_argument("--data-root", type=Path, required=True)
    download.add_argument("--allow-download", action="store_true")

    prepare = subparsers.add_parser(
        "extract", help="hash raw bytes, extract embeddings, and write the audited registry"
    )
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--input-root", type=Path, required=True)
    prepare.add_argument("--lineage-lock", type=Path, required=True)
    prepare.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "lock":
        lock = write_lineage_lock(args.output)
        print(
            json.dumps(
                {
                    "status": lock["status"],
                    "lineage_lock_sha256": lineage.sha256_path(args.output),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "download":
        if not args.allow_download:
            raise PermissionError("dataset download requires explicit --allow-download")
        extractors.download_frozen_datasets(raw_roots(args.data_root))
        return 0
    if args.command == "extract":
        registry = prepare_registry(
            data_root=args.data_root,
            input_root=args.input_root,
            lineage_lock_path=args.lineage_lock,
            resume=bool(args.resume),
        )
        print(
            json.dumps(
                {
                    "status": registry["status"],
                    "registry_sha256": lineage.sha256_path(
                        args.input_root / "audited_registry.json"
                    ),
                },
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "main",
    "prepare_registry",
    "raw_roots",
    "validate_lineage_lock",
    "write_lineage_lock",
]
