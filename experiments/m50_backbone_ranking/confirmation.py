"""Fail-closed untouched confirmation runner for M50.

Confirmation is intentionally unavailable in this v1 experiment package.
The v1 terminal artifact is a retrospective development/resource lock only;
it cannot consume embeddings, evaluate outcomes, or make a confirmed product
claim.  A placeholder registry raises ``blocked_missing_untouched_registry``;
even a valid audited registry raises ``blocked_unavailable_embeddings`` until
a separately versioned v2 runner with lineage-aware authorization exists.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import manifest


REGISTRY_PATH = Path(__file__).with_name("confirmation_registry.json")
CONFIRMATION_BUDGETS = (32, 64)
CONFIRMATION_REPLICATES = (0, 1, 2, 3, 4)
CONFIRMATION_HEADS = ("linear", "quadratic", "knn", "rbf")
KNOWN_FORBIDDEN_DATASET_IDS = frozenset(
    {"food101", "cifar100_coarse", "oxford_iiit_pet", "prior_synthetic_panels"}
)


def canonical_json(value: Any) -> str:
    return manifest.canonical_json(value)


def validate_registry(
    registry: Mapping[str, Any] | os.PathLike[str] | str = REGISTRY_PATH,
    *,
    minimum_datasets: int = 5,
) -> tuple[Mapping[str, Any], ...]:
    """Validate through the one canonical manifest registry validator."""

    return manifest.validate_untouched_registry(
        registry,
        minimum_panels=int(minimum_datasets),
    )


def load_registry(
    path: os.PathLike[str] | str = REGISTRY_PATH,
    *,
    minimum_datasets: int = 5,
) -> tuple[Mapping[str, Any], ...]:
    try:
        return validate_registry(path, minimum_datasets=minimum_datasets)
    except (PermissionError, RuntimeError, OSError, ValueError, TypeError) as exc:
        raise RuntimeError("blocked_missing_untouched_registry") from exc


def confirmation_panel_id(dataset_id: str, replicate: int, budget: int) -> str:
    if not str(dataset_id).strip():
        raise ValueError("dataset_id is required")
    if int(replicate) not in CONFIRMATION_REPLICATES:
        raise ValueError("replicate is outside the frozen confirmation grid")
    if int(budget) not in CONFIRMATION_BUDGETS:
        raise ValueError("budget is outside the frozen confirmation grid")
    return f"{dataset_id}__replicate-{int(replicate)}__budget-{int(budget)}"


def confirmation_identities(
    registry: Sequence[Mapping[str, Any]],
    *,
    selected_chain: str,
    chain_kind: str,
    development_decision_sha256: str,
    protocol_sha256: str,
    code_identity_sha256: str,
) -> tuple[dict[str, Any], ...]:
    """Build the complete immutable identity table without opening outcomes."""

    if selected_chain not in {"M0-SW", "M1-SW", "F", "G"}:
        raise PermissionError("confirmation lock has an invalid candidate")
    expected_kind = {
        "M0-SW": "pure_oi",
        "M1-SW": "pure_oi",
        "F": "oi_fusion",
        "G": "product_guardrail",
    }[selected_chain]
    if chain_kind != expected_kind:
        raise PermissionError("confirmation lock chain_kind does not match selected_chain")
    rows: list[dict[str, Any]] = []
    for dataset in registry:
        dataset_id = str(dataset.get("id", ""))
        embedding_manifest = dataset.get("embedding_manifest")
        if isinstance(embedding_manifest, Mapping):
            manifest_hash = str(embedding_manifest.get("sha256"))
            matrix_hash = str(embedding_manifest.get("matrix_sha256"))
        else:
            raise PermissionError("confirmation registry embedding manifest must be an object")
        for replicate in CONFIRMATION_REPLICATES:
            for budget in CONFIRMATION_BUDGETS:
                rows.append(
                    {
                        "stage": "confirmation",
                        "dataset_id": dataset_id,
                        "panel_id": confirmation_panel_id(dataset_id, replicate, budget),
                        "replicate": int(replicate),
                        "budget": int(budget),
                        "selected_chain": selected_chain,
                        "chain_kind": chain_kind,
                        "development_decision_sha256": str(development_decision_sha256),
                        "protocol_sha256": str(protocol_sha256),
                        "code_identity_sha256": str(code_identity_sha256),
                        "embedding_manifest_sha256": manifest_hash,
                        "matrix_sha256": matrix_hash,
                        "heads": list(CONFIRMATION_HEADS),
                        "outcome_reselection": False,
                    }
                )
    return tuple(rows)


def run_confirmation(
    *,
    output: Path,
    development_decision: Mapping[str, Any] | os.PathLike[str] | str,
    registry: Mapping[str, Any] | os.PathLike[str] | str = REGISTRY_PATH,
    protocol_sha256: str | None = None,
    code_identity_sha256: str | None = None,
    outcome_factory: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Keep v1 terminally blocked until a separately versioned v2 runner exists.

    Registry validation intentionally occurs before any outcome access.  This
    module only validates the planned registry and final development lock; it
    does not load new embeddings or produce the canonical ``contrast_rows``
    table.  In particular, a caller cannot turn a missing embedding panel
    into a misleading completed artifact by supplying a generic outcome
    callback.  A future v2 implementation must use a new lineage-bound
    validator rather than resuming this v1 stage.
    """

    if Path(output).resolve().joinpath("manifest.json").exists():
        raise RuntimeError("terminal confirmation artifact exists; refuse overwrite/resume")
    current_protocol = manifest.verify_protocol_hash()
    current_code = manifest.code_identity_sha256(require_existing=True)
    if protocol_sha256 is not None and str(protocol_sha256) != current_protocol:
        raise PermissionError("confirmation protocol hash does not match current verified protocol")
    if code_identity_sha256 is not None and str(code_identity_sha256) != current_code:
        raise PermissionError("confirmation code identity does not match current verified sources")
    lock = manifest.validate_development_lock(
        development_decision,
        protocol_hash=current_protocol,
        code_identity_hash=current_code,
    )
    registry_rows = load_registry(registry)
    # The current experiment has no confirmation embedding loader or outcome
    # writer.  Do not create an empty/partial terminal artifact that analysis
    # could mistake for an audited confirmation run.
    del lock, registry_rows, outcome_factory
    raise RuntimeError("blocked_unavailable_embeddings")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-decision", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    return parser


def main() -> int:
    args = _parser().parse_args()
    run_confirmation(
        output=args.output,
        development_decision=args.development_decision,
        registry=args.registry,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CONFIRMATION_BUDGETS", "CONFIRMATION_HEADS", "CONFIRMATION_REPLICATES", "KNOWN_FORBIDDEN_DATASET_IDS", "REGISTRY_PATH", "confirmation_identities", "confirmation_panel_id", "load_registry", "run_confirmation", "validate_registry",
]
