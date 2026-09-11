"""Read-only compatibility loader for the archived fresh embedding bank.

The original experiment-local source was not committed and was removed when
macOS cleaned its temporary worktree.  The immutable downstream extraction
manifest still binds the exact backbone specifications and reused LSP matrix
hashes.  This module restores only the read-only interface required to validate
and load those archived matrices; it cannot extract or overwrite embeddings.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    ROOT / "artifacts" / "continuous_fresh_structured_opportunity_discovery" / "embeddings"
)
_DOWNSTREAM_MANIFEST = (
    ROOT
    / "artifacts"
    / "continuous_large_structured_opportunity_benchmark"
    / "embeddings"
    / "manifest.json"
)
BACKBONES = (
    "dinov2-small",
    "deit-tiny",
    "mobilenetv3-large",
    "resnet50",
    "convnext-tiny",
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest() -> dict[str, Any]:
    if not _DOWNSTREAM_MANIFEST.is_file():
        raise FileNotFoundError("archived downstream embedding manifest is missing")
    value = json.loads(_DOWNSTREAM_MANIFEST.read_text(encoding="utf-8"))
    if (
        value.get("status")
        != "completed_large_structured_opportunity_embedding_extraction"
        or tuple(value.get("backbone_ids", ())) != BACKBONES
    ):
        raise ValueError("archived downstream embedding manifest identity changed")
    return value


def _specs(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for record in manifest.get("new_records", ()):
        backbone = str(record.get("backbone_id"))
        spec = record.get("spec")
        if backbone not in BACKBONES or not isinstance(spec, dict):
            raise ValueError("archived backbone specification is malformed")
        if backbone in output and output[backbone] != spec:
            raise ValueError("archived backbone specifications disagree")
        output[backbone] = dict(spec)
    if tuple(output) != BACKBONES:
        output = {backbone: output[backbone] for backbone in BACKBONES}
    return output


SPEC_BY_ID = _specs(_manifest())


def validate_model_inputs() -> list[dict[str, Any]]:
    """Return the terminal-manifest-bound model identities."""
    manifest = _manifest()
    values = manifest.get("model_inputs")
    if (
        not isinstance(values, list)
        or tuple(str(row.get("backbone_id")) for row in values) != BACKBONES
        or any(not isinstance(row.get("checkpoint_sha256"), str) for row in values)
        or any(not isinstance(row.get("spec_sha256"), str) for row in values)
    ):
        raise ValueError("archived model-input identities changed")
    return [dict(row) for row in values]


def load_bank(
    input_root: Optional[Path] = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Validate and load the archived CUB/LSP matrices without mutation."""
    del input_root
    root = Path(output_root).resolve()
    manifest = _manifest()
    lsp_records = {
        str(row["backbone_id"]): row
        for row in manifest.get("reused_matrices", ())
        if row.get("dataset_id") == "lsp_2000_pose_geometry"
    }
    if set(lsp_records) != set(BACKBONES):
        raise ValueError("archived LSP matrix identities are incomplete")
    matrices: dict[str, dict[str, np.ndarray]] = {}
    for dataset_id in ("cub_200_2011_beak_tail", "lsp_2000_pose"):
        matrices[dataset_id] = {}
        for backbone in BACKBONES:
            path = root / dataset_id / f"{backbone}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"archived embedding matrix is missing: {dataset_id}/{backbone}")
            matrix = np.load(path, mmap_mode="r", allow_pickle=False)
            expected_dimension = int(SPEC_BY_ID[backbone]["dimension"])
            if (
                matrix.ndim != 2
                or matrix.shape[1] != expected_dimension
                or matrix.dtype != np.dtype("float32")
                or not np.isfinite(np.asarray(matrix)).all()
            ):
                raise ValueError(f"archived embedding matrix is invalid: {dataset_id}/{backbone}")
            if dataset_id == "lsp_2000_pose":
                record = lsp_records[backbone]
                if (
                    int(record.get("rows", -1)) != int(matrix.shape[0])
                    or int(record.get("dimension", -1)) != int(matrix.shape[1])
                    or record.get("matrix_sha256") != _sha256_path(path)
                ):
                    raise ValueError(f"archived LSP embedding identity changed: {backbone}")
            matrices[dataset_id][backbone] = matrix
    cohort = {
        "status": "read_only_compatibility_loader",
        "source_manifest": str(_DOWNSTREAM_MANIFEST.relative_to(ROOT)),
    }
    return cohort, matrices


__all__ = [
    "BACKBONES",
    "DEFAULT_OUTPUT_ROOT",
    "SPEC_BY_ID",
    "load_bank",
    "validate_model_inputs",
]
