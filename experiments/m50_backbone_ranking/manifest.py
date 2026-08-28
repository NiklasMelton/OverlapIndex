"""Frozen planning and provenance primitives for the M50 bridge experiment.

The M50 study is deliberately a research-local package.  This module owns
identity, scheduling, and authorization checks; it does not fit an estimator
or inspect evaluation outcomes.  In particular, the code identity is the
hash of an explicit source list, never a recursive hash of the output tree.
That distinction is important for resumable runs: writing a checkpoint must
not change the identity of the code that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_BRANCH = "codex/experiment-m50-backbone-ranking"
EXPECTED_STARTING_COMMIT = "1faf2ea06878c47afdbb85d51b5cfe7cf3fd3742"
EXPECTED_ORIGINAL_DEVELOP_BASE = "165fa344653a72dd2abc04a20387d90b891b9acc"
# Common spellings are retained as constants, not API aliases, because they
# make provenance records self-describing when read outside this package.
STARTING_COMMIT = EXPECTED_STARTING_COMMIT
ORIGINAL_DEVELOP_BASE = EXPECTED_ORIGINAL_DEVELOP_BASE

PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
PROTOCOL_SHA256_PATH = Path(__file__).with_name("protocol.sha256")

# These roots may contain checkpoints, raw JSONL, plots, and temporary files.
# None of them is allowed into code identity.
OUTPUT_ROOTS = (
    "artifacts/m50_backbone_ranking",
    "artifacts/m50_backbone_ranking/",
)

DECLARED_SOURCE_PATHS = (
    "pyproject.toml",
    "overlapindex/BallCover.py",
    "overlapindex/ContinuousOverlapIndex.py",
    "overlapindex/OverlapIndex.py",
    "overlapindex/__init__.py",
    "overlapindex/_prototype_refinement.py",
    "overlapindex/_universal_scorer.py",
    "overlapindex/clustering.py",
    "overlapindex/utils.py",
    "experiments/m50_backbone_ranking/__init__.py",
    "experiments/m50_backbone_ranking/candidates.py",
    "experiments/m50_backbone_ranking/conditioning_adapter.py",
    "experiments/m50_backbone_ranking/manifest.py",
    "experiments/m50_backbone_ranking/food101.py",
    "experiments/m50_backbone_ranking/runtime_benchmark.py",
    "experiments/m50_backbone_ranking/confirmation.py",
    "experiments/m50_backbone_ranking/fusion.py",
    "experiments/m50_backbone_ranking/analysis.py",
    "experiments/m50_backbone_ranking/statistics.py",
    "experiments/m50_backbone_ranking/reporting.py",
    "experiments/m50_backbone_ranking/README.md",
    "experiments/m50_backbone_ranking/confirmation_registry.json",
    "experiments/m50_backbone_ranking/protocol.json",
    "experiments/m50_backbone_ranking/protocol.sha256",
)

SCHEMA_VERSION = 1
SCHEDULE_SEED = 20260827

MODELS = (
    "dinov2-small",
    "deit-tiny",
    "convnext-tiny",
    "mobilenetv3-large",
    "openclip-vit-b-32",
    "resnet50",
    "efficientnet-b0",
    "swin-tiny",
    "vit-small-16",
    "densenet121",
)
ARMS = ("baseline", "nonlinearity_full", "nuisance_full")
BUDGETS = (64, 68, 72, 80)
REPLICATES = (0, 1, 2, 3, 4)
FOLDS = 5
K = 10
SELECTOR_METHODS = (
    "A",
    "B",
    "M0-SW",
    "M1-SW",
    "M0-CB",
    "M1-CB",
    "LP-FULL",
)
OI_METHODS = SELECTOR_METHODS[:-1]
RUNTIME_BUDGETS = (64, 128, 256, 512, 640)
RUNTIME_REPEATS = (0, 1, 2, 3, 4)
RUNTIME_ARMS = ARMS
# Runtime executes one fixed primitive schedule whenever a SW tier is
# eligible.  F/G are policy chains derived from these measurements; they are
# never estimator IDs and therefore never appear in this primitive schedule.
RUNTIME_PRIMITIVE_METHODS = (
    "A",
    "B",
    "M0-SW",
    "M1-SW",
    "LP-FULL",
    "LP-CAPPED-2048",
)
DEFAULT_RUNTIME_METHODS = RUNTIME_PRIMITIVE_METHODS

# The v1 confirmation panel is immutable planning evidence chosen before
# embeddings/outcomes exist.  These exact IDs and task definitions are frozen
# in code/protocol.  A future v2 must copy and hash-link this plan while using
# its own new code identity, audited registry, and evaluator; v1 never fills
# in these rows or resumes confirmation under its own identity.
CONFIRMATION_DATASET_IDS = (
    "torchvision_cifar10",
    "torchvision_stl10",
    "torchvision_gtsrb",
    "torchvision_fgvc_aircraft",
    "torchvision_dtd",
)
CONFIRMATION_DATASET_SPECS = {
    "torchvision_cifar10": {
        "version": "torchvision_dataset_definition_pinned_at_extraction",
        "training_split": "train",
        "evaluation_split": "test",
    },
    "torchvision_stl10": {
        "version": "torchvision_dataset_definition_pinned_at_extraction",
        "training_split": "train",
        "evaluation_split": "test",
    },
    "torchvision_gtsrb": {
        "version": "torchvision_dataset_definition_pinned_at_extraction",
        "training_split": "train",
        "evaluation_split": "test",
    },
    "torchvision_fgvc_aircraft": {
        "version": "torchvision_dataset_definition_pinned_at_extraction",
        "training_split": "trainval",
        "evaluation_split": "test",
    },
    "torchvision_dtd": {
        "version": "torchvision_dataset_definition_pinned_at_extraction",
        "training_split": "train_plus_val_partition_1",
        "evaluation_split": "test_partition_1",
    },
}
CONFIRMATION_MIN_TRAINING_EXAMPLES_PER_CLASS = 64
CONFIRMATION_MIN_EVALUATION_EXAMPLES_PER_CLASS = 20


def _json_safe(value: Any) -> Any:
    """Convert ordinary/numpy-like metadata to strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not value == value:
        return None
    if isinstance(value, float) and value in {float("inf"), float("-inf")}:
        return None
    return value


def canonical_json(value: Any) -> str:
    """Return the one byte-stable JSON representation used for hashes."""

    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_sha256(path: os.PathLike[str] | str = PROTOCOL_PATH) -> str:
    return sha256_path(path)


def recorded_protocol_sha256(
    path: os.PathLike[str] | str | None = None,
) -> str:
    """Read the one-line, lower-case protocol digest sidecar.

    A protocol sidecar is part of the frozen input contract.  Treating a
    missing or malformed sidecar as ``None`` would let a run proceed with an
    unrecorded protocol identity, so all such states fail closed.
    """

    candidate = Path(PROTOCOL_SHA256_PATH if path is None else path)
    if not candidate.is_file():
        raise RuntimeError(f"protocol SHA-256 sidecar is missing: {candidate}")
    try:
        raw = candidate.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"protocol SHA-256 sidecar is unreadable: {candidate}") from exc
    lines = raw.splitlines()
    if len(lines) != 1:
        raise RuntimeError("protocol SHA-256 sidecar must contain exactly one digest line")
    value = lines[0]
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RuntimeError(
            "protocol SHA-256 sidecar must contain exactly one lower-case 64-hex digest"
        )
    return value


def verify_protocol_hash(path: os.PathLike[str] | str = PROTOCOL_PATH) -> str:
    digest = protocol_sha256(path)
    recorded = recorded_protocol_sha256()
    if recorded != digest:
        raise RuntimeError(
            f"protocol hash mismatch: recorded {recorded}, observed {digest}"
        )
    return digest


def verify_protocol_source_evidence(
    protocol_path: os.PathLike[str] | str = PROTOCOL_PATH,
    *,
    root: os.PathLike[str] | str = ROOT,
) -> dict[str, dict[str, str]]:
    """Verify every readable ``source_evidence`` path before a run.

    The protocol records both archived external files and prior local
    development artifacts.  This check is intentionally independent from the
    source-code identity: prior outcome artifacts are provenance evidence,
    not executable inputs, but a changed readable artifact must still be
    surfaced before an outcome run starts.
    """

    payload = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    evidence = payload.get("source_evidence") if isinstance(payload, Mapping) else None
    if not isinstance(evidence, Mapping):
        raise RuntimeError("protocol source_evidence is missing or malformed")
    root_path = Path(root).resolve()
    observed: dict[str, dict[str, str]] = {}
    for name, raw in evidence.items():
        if not isinstance(raw, Mapping):
            # ``cache_identity`` is the one intentional prose declaration in
            # source_evidence.  Any other non-object entry is malformed.
            if str(name) == "cache_identity" and isinstance(raw, str) and raw:
                continue
            raise RuntimeError(f"protocol source evidence entry {name!r} is malformed")
        path_value = raw.get("path")
        expected_hash = raw.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected_hash, str):
            raise RuntimeError(f"protocol source evidence entry {name!r} lacks path/sha256")
        if len(expected_hash) != 64 or any(
            char not in "0123456789abcdef" for char in expected_hash
        ):
            raise RuntimeError(f"protocol source evidence hash for {name!r} is malformed")
        path = Path(path_value)
        if not path.is_absolute():
            path = root_path / path
        if not path.is_file():
            raise FileNotFoundError(f"protocol source evidence is unreadable: {path}")
        digest = sha256_path(path)
        if digest != expected_hash:
            raise RuntimeError(
                f"protocol source evidence hash mismatch for {name}: "
                f"expected {expected_hash}, got {digest}"
            )
        observed[str(name)] = {"path": str(path), "sha256": digest}
    return observed


def _relative_source_path(root: Path, relative: str) -> Path:
    if str(relative).startswith("/"):
        raise ValueError("source paths must be repository-relative")
    if any(
        str(relative) == output or str(relative).startswith(output)
        for output in OUTPUT_ROOTS
    ):
        raise ValueError(f"generated output root cannot be hashed as source: {relative!r}")
    candidate = (root / str(relative)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"source path escapes repository root: {relative!r}") from exc
    return candidate


def source_hashes(
    root: os.PathLike[str] | str = ROOT,
    source_paths: Optional[Sequence[str]] = None,
    *,
    require_existing: bool = False,
) -> dict[str, str | None]:
    """Hash only the explicit recipe files; never walk output/checkpoint roots."""

    root_path = Path(root).resolve()
    relatives = tuple(source_paths or DECLARED_SOURCE_PATHS)
    result: dict[str, str | None] = {}
    for relative in relatives:
        path = _relative_source_path(root_path, str(relative))
        if not path.is_file():
            if require_existing:
                raise RuntimeError(f"missing declared experiment source: {relative}")
            result[str(relative)] = None
        else:
            result[str(relative)] = sha256_path(path)
    return result


def code_identity_sha256(
    root: os.PathLike[str] | str = ROOT,
    source_hashes_map: Optional[Mapping[str, str | None]] = None,
    *,
    require_existing: bool = False,
) -> str:
    """Hash the declared source recipe and frozen base, independent of output."""

    hashes = dict(
        source_hashes_map
        if source_hashes_map is not None
        else source_hashes(root, require_existing=require_existing)
    )
    payload = {
        "schema": "m50_declared_sources_v1",
        "starting_commit": EXPECTED_STARTING_COMMIT,
        "original_develop_base": EXPECTED_ORIGINAL_DEVELOP_BASE,
        "protocol_sha256": (
            protocol_sha256(PROTOCOL_PATH) if PROTOCOL_PATH.is_file() else None
        ),
        "source_hashes": dict(sorted(hashes.items())),
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def working_tree_hash(
    root: os.PathLike[str] | str = ROOT,
    source_hashes_map: Optional[Mapping[str, str | None]] = None,
) -> str:
    """Output-independent identity used by resume/comparison checks."""

    return code_identity_sha256(root, source_hashes_map)


def _git_output(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=str(root), text=True, stderr=subprocess.PIPE
    ).strip()


def repository_provenance(
    root: os.PathLike[str] | str = ROOT,
    *,
    require_sources: bool = False,
) -> dict[str, Any]:
    """Verify branch/base and return reproducibility metadata."""

    root_path = Path(root).resolve()
    try:
        branch = _git_output(root_path, "branch", "--show-current")
        head = _git_output(root_path, "rev-parse", "HEAD")
        develop = _git_output(root_path, "rev-parse", "develop")
        merge_base = _git_output(root_path, "merge-base", "HEAD", "develop")
        start_merge = _git_output(
            root_path, "merge-base", "HEAD", EXPECTED_STARTING_COMMIT
        )
        status = _git_output(
            root_path, "status", "--porcelain=v1", "--untracked-files=all"
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot collect git provenance") from exc
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(
            f"experiment must run on {EXPECTED_BRANCH!r}; found {branch!r}"
        )
    if develop != EXPECTED_ORIGINAL_DEVELOP_BASE or merge_base != EXPECTED_ORIGINAL_DEVELOP_BASE:
        raise RuntimeError(
            "recorded original develop base mismatch: expected "
            f"{EXPECTED_ORIGINAL_DEVELOP_BASE}, develop={develop}, merge-base={merge_base}"
        )
    if start_merge != EXPECTED_STARTING_COMMIT:
        raise RuntimeError(
            "recorded experiment starting point is not the verified branch base: "
            f"expected {EXPECTED_STARTING_COMMIT}, merge-base={start_merge}"
        )
    hashes = source_hashes(root_path, require_existing=require_sources)
    return {
        "branch": branch,
        "starting_commit": EXPECTED_STARTING_COMMIT,
        "original_develop_base": EXPECTED_ORIGINAL_DEVELOP_BASE,
        "develop_commit_at_execution": develop,
        "merge_base_with_develop": merge_base,
        "merge_base_with_starting_commit": start_merge,
        "experiment_commit": head,
        "git_dirty": bool(status),
        "git_status_porcelain": status.splitlines() if status else [],
        "git_status_sha256": sha256_bytes(status.encode("utf-8")),
        "source_hashes": hashes,
        "code_identity_sha256": code_identity_sha256(root_path, hashes),
        "output_roots_excluded": list(OUTPUT_ROOTS),
    }


@dataclass(frozen=True)
class FoodPanel:
    """A stable identity for one Food-101 selector panel."""

    model: str
    replicate: int
    arm: str
    budget: int

    @property
    def panel_id(self) -> str:
        return f"{self.model}__replicate-{self.replicate}__{self.arm}__budget-{self.budget}"

    def identity(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "replicate": int(self.replicate),
            "arm": self.arm,
            "budget": int(self.budget),
        }


def development_panels() -> tuple[FoodPanel, ...]:
    return tuple(
        FoodPanel(model, replicate, arm, budget)
        for model in MODELS
        for replicate in REPLICATES
        for arm in ARMS
        for budget in BUDGETS
    )


EXPECTED_DEVELOPMENT_PANEL_COUNT = (
    len(MODELS) * len(REPLICATES) * len(ARMS) * len(BUDGETS)
)
EXPECTED_DEVELOPMENT_ROW_COUNT = EXPECTED_DEVELOPMENT_PANEL_COUNT * len(SELECTOR_METHODS)


def validate_development_panels(panels: Sequence[FoodPanel]) -> bool:
    expected = development_panels()
    normalized = tuple(panels)
    if normalized != expected:
        raise ValueError("Food development panels do not match the exact frozen grid")
    return True


def panel_grid_sha256(panels: Sequence[FoodPanel] | None = None) -> str:
    values = tuple(panels or development_panels())
    return sha256_bytes((canonical_json([p.identity() for p in values]) + "\n").encode())


def _hash_rank(value: str, seed: int) -> bytes:
    return hashlib.sha256(f"{int(seed)}\0{value}".encode("utf-8")).digest()


def planned_execution_order(
    panels: Sequence[FoodPanel | Mapping[str, Any] | str],
    methods: Sequence[str] = SELECTOR_METHODS,
    *,
    schedule_seed: int = SCHEDULE_SEED,
) -> tuple[dict[str, Any], ...]:
    """Return one cyclic, near-counterbalanced method schedule per panel.

    The seven-method development grid has 600 panels, so a cyclic schedule
    necessarily gives each method/position 85 or 86 assignments.  The
    six-method runtime grid has 750 panels and is exactly counterbalanced.
    """

    method_tuple = tuple(str(method) for method in methods)
    if not method_tuple or len(set(method_tuple)) != len(method_tuple):
        raise ValueError("methods must be a non-empty unique sequence")
    panel_ids: list[str] = []
    for value in panels:
        if isinstance(value, str):
            panel_ids.append(value)
        elif isinstance(value, FoodPanel):
            panel_ids.append(value.panel_id)
        elif isinstance(value, Mapping):
            if "panel_id" in value:
                panel_ids.append(str(value["panel_id"]))
            else:
                panel_ids.append(
                    FoodPanel(
                        str(value["model"]), int(value["replicate"]),
                        str(value["arm"]), int(value["budget"]),
                    ).panel_id
                )
        else:
            raise TypeError("panels must contain FoodPanel, mapping, or panel id values")
    if len(set(panel_ids)) != len(panel_ids):
        raise ValueError("panel ids must be unique")
    ranked = sorted(panel_ids, key=lambda item: (_hash_rank(item, schedule_seed), item))
    offsets = {panel_id: rank % len(method_tuple) for rank, panel_id in enumerate(ranked)}
    rows: list[dict[str, Any]] = []
    for panel_id in sorted(panel_ids):
        offset = offsets[panel_id]
        for position in range(len(method_tuple)):
            rows.append(
                {
                    "panel_id": panel_id,
                    "method_id": method_tuple[(offset + position) % len(method_tuple)],
                    "execution_position": position,
                    "schedule_seed": int(schedule_seed),
                }
            )
    return tuple(rows)


def validate_execution_order(
    schedule: Sequence[Mapping[str, Any]],
    panels: Sequence[FoodPanel | Mapping[str, Any] | str],
    methods: Sequence[str] = SELECTOR_METHODS,
    *,
    schedule_seed: int = SCHEDULE_SEED,
) -> bool:
    expected = planned_execution_order(panels, methods, schedule_seed=schedule_seed)
    normalized = tuple(
        {
            "panel_id": str(row.get("panel_id")),
            "method_id": str(row.get("method_id")),
            "execution_position": int(row.get("execution_position", -1)),
            "schedule_seed": int(row.get("schedule_seed", schedule_seed)),
        }
        for row in schedule
    )
    if normalized != expected:
        raise ValueError("execution schedule does not match the frozen counterbalance")
    n_methods = len(tuple(methods))
    for panel_id in {row["panel_id"] for row in normalized}:
        positions = [row["execution_position"] for row in normalized if row["panel_id"] == panel_id]
        if set(positions) != set(range(n_methods)):
            raise ValueError(f"panel {panel_id!r} has an incomplete execution cycle")
    return True


def runtime_panels() -> tuple[tuple[str, str, int, int], ...]:
    return tuple(
        (model, arm, int(budget), int(repeat))
        for model in MODELS
        for arm in RUNTIME_ARMS
        for budget in RUNTIME_BUDGETS
        for repeat in RUNTIME_REPEATS
    )


EXPECTED_RUNTIME_PANEL_COUNT = len(runtime_panels())
EXPECTED_RUNTIME_ROW_COUNT = EXPECTED_RUNTIME_PANEL_COUNT * len(RUNTIME_PRIMITIVE_METHODS)


def runtime_execution_order(
    methods: Sequence[str],
    *,
    schedule_seed: int = SCHEDULE_SEED,
) -> tuple[dict[str, Any], ...]:
    """Counterbalance runtime methods over the exact model/arm/budget/repeat grid."""

    method_tuple = tuple(str(value) for value in methods)
    if not method_tuple or len(set(method_tuple)) != len(method_tuple):
        raise ValueError("runtime methods must be non-empty and unique")
    blocks = tuple(
        f"{model}\0{arm}\0{budget}\0{repeat}"
        for model, arm, budget, repeat in runtime_panels()
    )
    ranked = sorted(blocks, key=lambda value: (_hash_rank(value, schedule_seed), value))
    offset = {block: rank % len(method_tuple) for rank, block in enumerate(ranked)}
    rows: list[dict[str, Any]] = []
    for model, arm, budget, repeat in runtime_panels():
        block = f"{model}\0{arm}\0{budget}\0{repeat}"
        for position in range(len(method_tuple)):
            rows.append(
                {
                    "model": model,
                    "arm": arm,
                    "budget": int(budget),
                    "repeat": int(repeat),
                    "method_id": method_tuple[(offset[block] + position) % len(method_tuple)],
                    "execution_position": position,
                    "schedule_seed": int(schedule_seed),
                }
            )
    return tuple(rows)


def validate_runtime_grid(rows: Sequence[Mapping[str, Any]]) -> bool:
    expected = {
        (model, arm, budget, repeat)
        for model, arm, budget, repeat in runtime_panels()
    }
    observed = {
        (
            str(row.get("model", row.get("backbone"))),
            str(row.get("arm")),
            int(row.get("budget")),
            int(row.get("repeat")),
        )
        for row in rows
    }
    if observed != expected:
        raise ValueError("runtime rows do not cover the exact frozen grid")
    if len(rows) != EXPECTED_RUNTIME_ROW_COUNT:
        raise ValueError(
            f"runtime rows must contain exactly {EXPECTED_RUNTIME_ROW_COUNT} "
            f"primitive method rows; got {len(rows)}"
        )
    identities = {
        (
            str(row.get("method_id", row.get("candidate_id"))),
            str(row.get("model", row.get("backbone"))),
            str(row.get("arm")),
            int(row.get("budget")),
            int(row.get("repeat")),
        )
        for row in rows
    }
    expected_identities = {
        (method, model, arm, budget, repeat)
        for model, arm, budget, repeat in runtime_panels()
        for method in RUNTIME_PRIMITIVE_METHODS
    }
    if identities != expected_identities:
        raise ValueError("runtime rows do not contain the exact primitive method identities")
    return True


def _artifact_payload(value: Mapping[str, Any] | os.PathLike[str] | str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    payload = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("artifact must contain a JSON object")
    return payload


_SELECTED_CHAIN_RECIPE_KEYS = frozenset(
    {
        "schema_version",
        "selected_candidate",
        "selected_chain",
        "guardrail_base_id",
        "primitive_component_ids",
        "component_recipe_family_sha256",
        "derived_recipe_identity",
        "derived_recipe_sha256",
        "deterministic_seed_rule",
    }
)
_SELECTED_CHAIN_RECIPE_COMPONENT_KEYS = frozenset(
    {"food_recipe_family_sha256", "runtime_recipe_family_sha256"}
)
_SELECTED_CHAIN_RECIPE_FOOD_SEED_RULE = {
    "selector_panel_seed": "SEED + replicate",
    "oi_fold_seed": "selector_panel_seed + fold",
    "probe_split_seed": "selector_panel_seed",
    "probe_model_random_state": "selector_panel_seed",
    "capped_subset_seed": "selector_panel_seed",
}
_SELECTED_CHAIN_RECIPE_RUNTIME_SEED_RULE = {
    "runtime_call_seed": "SEED + budget",
    "oi_fold_seed": "runtime_call_seed + fold",
    "probe_split_seed": "runtime_call_seed",
    "probe_model_random_state": "runtime_call_seed",
    "capped_subset_seed": "runtime_call_seed",
    "repeat_effect": "order_and_timing_only",
}


def _is_lower_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _expected_selected_chain_recipe(
    selected_candidate: str,
    selected_chain: str,
    guardrail_base_id: str | None,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Return the closed primitive and derived recipe identities.

    This intentionally mirrors the small, deterministic composition table in
    the analysis stage.  It is kept here as a validation table rather than an
    import of analysis, so provenance checks remain usable before numerical
    dependencies or outcome tables are loaded.
    """

    if selected_candidate not in {"M0-SW", "M1-SW"}:
        raise PermissionError("selected-chain recipe has an invalid selected_candidate")
    if selected_chain not in {"M0-SW", "M1-SW", "F", "G"}:
        raise PermissionError("selected-chain recipe has an invalid selected_chain")
    raw_candidate = "A" if selected_candidate == "M0-SW" else "B"
    if selected_chain in {"M0-SW", "M1-SW"}:
        if selected_chain != selected_candidate or guardrail_base_id is not None:
            raise PermissionError("pure selected-chain recipe has an invalid base binding")
        return [selected_candidate], {}
    if selected_chain == "F":
        if guardrail_base_id is not None:
            raise PermissionError("fusion selected-chain recipe has an invalid base binding")
        return sorted({raw_candidate, selected_candidate}), {
            "F": {
                "candidate_id": "F",
                "recipe": "equal_fractional_midrank_fusion",
                "component_ids": [raw_candidate, selected_candidate],
                "component_weights": [0.5, 0.5],
                "rank_transform": "complete_panel_fractional_midranks_[0,1]_exact_ties_averaged",
                "selection": "frozen_backbone_order_argmax_on_exact_fused_ties",
            }
        }
    if guardrail_base_id not in {selected_candidate, "F"}:
        raise PermissionError("guardrail selected-chain recipe has a mismatched base chain")
    primitives = {selected_candidate, "B", "M1-SW", "LP-CAPPED-2048"}
    derived: dict[str, dict[str, Any]] = {}
    if guardrail_base_id == "F":
        primitives.add(raw_candidate)
        derived["F"] = {
            "candidate_id": "F",
            "recipe": "equal_fractional_midrank_fusion",
            "component_ids": [raw_candidate, selected_candidate],
            "component_weights": [0.5, 0.5],
            "rank_transform": "complete_panel_fractional_midranks_[0,1]_exact_ties_averaged",
            "selection": "frozen_backbone_order_argmax_on_exact_fused_ties",
        }
    derived["G"] = {
        "candidate_id": "G",
        "recipe": "selective_capped_probe_policy",
        "base_candidate_id": guardrail_base_id,
        "diagnostic_component_ids": ["B", "M1-SW"],
        "capped_component_id": "LP-CAPPED-2048",
        "trigger": {
            "scope": "panel_wide_any_backbone",
            "B_refinement_activity_gte": 0.50,
            "absolute_B_minus_M1_score_gte": 0.05,
            "outcome_fields_allowed": False,
        },
        "untriggered_selection": (
            "F_frozen_order_argmax"
            if guardrail_base_id == "F"
            else "archived_atol_1e-12_tie_average"
        ),
        "triggered_selection": "archived_atol_1e-12_tie_average",
    }
    return sorted(primitives), derived


def _validate_selected_chain_recipe_bindings(
    payload: Mapping[str, Any],
    analysis_manifest: Mapping[str, Any],
    directory: Path,
) -> None:
    """Validate the final lock's exact, sibling-bound recipe sidecar.

    The selected-chain binding is a semantic identity, not decorative report
    metadata.  Validate its closed component set and recompute every derived
    recipe digest before accepting a final lock.  Component family values are
    already hashes of the independently validated Food/runtime recipe tables;
    here we enforce their complete key set and digest shape, while the outer
    binding hash and sibling file identity make the table output immutable.
    """

    binding = payload.get("selected_chain_recipe_bindings")
    binding_hash = payload.get("selected_chain_recipe_bindings_sha256")
    if not isinstance(binding, Mapping) or set(binding) != _SELECTED_CHAIN_RECIPE_KEYS:
        raise PermissionError(
            "final development lock lacks the exact selected-chain recipe binding"
        )
    if not _is_lower_sha256(binding_hash):
        raise PermissionError(
            "final development lock selected-chain recipe binding hash is malformed"
        )
    if binding.get("schema_version") != 1 or type(binding.get("schema_version")) is not int:
        raise PermissionError("selected-chain recipe binding schema_version is not 1")
    selected_candidate = payload.get("selected_candidate")
    selected_chain = payload.get("selected_chain")
    guardrail_base_id = binding.get("guardrail_base_id")
    if binding.get("selected_candidate") != selected_candidate:
        raise PermissionError("selected-chain recipe selected_candidate does not match lock")
    if binding.get("selected_chain") != selected_chain:
        raise PermissionError("selected-chain recipe selected_chain does not match lock")
    if guardrail_base_id is not None and not isinstance(guardrail_base_id, str):
        raise PermissionError("selected-chain recipe guardrail base is malformed")
    if "guardrail_base_id" in payload and payload.get("guardrail_base_id") != guardrail_base_id:
        raise PermissionError("selected-chain recipe guardrail base does not match lock")
    expected_components, expected_derived = _expected_selected_chain_recipe(
        str(selected_candidate), str(selected_chain),
        guardrail_base_id,
    )
    components = binding.get("primitive_component_ids")
    if (
        not isinstance(components, Sequence)
        or isinstance(components, (str, bytes))
        or list(components) != expected_components
    ):
        raise PermissionError("selected-chain recipe primitive component set is not canonical")

    component_hashes = binding.get("component_recipe_family_sha256")
    if not isinstance(component_hashes, Mapping) or set(component_hashes) != set(expected_components):
        raise PermissionError("selected-chain recipe component hashes are incomplete")
    for component in expected_components:
        value = component_hashes.get(component)
        if not isinstance(value, Mapping) or set(value) != _SELECTED_CHAIN_RECIPE_COMPONENT_KEYS:
            raise PermissionError(
                f"selected-chain recipe component hash for {component!r} is malformed"
            )
        if any(not _is_lower_sha256(value.get(key)) for key in _SELECTED_CHAIN_RECIPE_COMPONENT_KEYS):
            raise PermissionError(
                f"selected-chain recipe component hash for {component!r} is malformed"
            )

    derived_identity = binding.get("derived_recipe_identity")
    if not isinstance(derived_identity, Mapping) or set(derived_identity) != set(expected_derived):
        raise PermissionError("selected-chain recipe derived identities are incomplete")
    if canonical_json(derived_identity) != canonical_json(expected_derived):
        raise PermissionError("selected-chain recipe derived identities do not match the frozen chain")
    derived_hashes = binding.get("derived_recipe_sha256")
    expected_derived_hashes = {
        candidate_id: sha256_bytes(canonical_json(identity).encode("utf-8"))
        for candidate_id, identity in sorted(expected_derived.items())
    }
    if derived_hashes != expected_derived_hashes:
        raise PermissionError("selected-chain recipe derived hash mismatch")
    seed_rule = binding.get("deterministic_seed_rule")
    expected_seed_rule = {
        "food": _SELECTED_CHAIN_RECIPE_FOOD_SEED_RULE,
        "runtime": _SELECTED_CHAIN_RECIPE_RUNTIME_SEED_RULE,
    }
    if canonical_json(seed_rule) != canonical_json(expected_seed_rule):
        raise PermissionError("selected-chain recipe deterministic seed rule mismatch")
    if sha256_bytes(canonical_json(binding).encode("utf-8")) != binding_hash:
        raise PermissionError("selected-chain recipe binding hash mismatch")

    files = analysis_manifest.get("files")
    sidecar_name = "selected_chain_recipe_bindings.json"
    sidecar = directory / sidecar_name
    if not isinstance(files, Mapping) or sidecar_name not in files or not sidecar.is_file():
        raise PermissionError(
            "analysis manifest lacks the selected-chain recipe binding sidecar"
        )
    sidecar_identity = files.get(sidecar_name)
    sidecar_hash = sha256_path(sidecar) if sidecar.is_file() else None
    if (
        not isinstance(sidecar_identity, Mapping)
        or not _is_lower_sha256(sidecar_identity.get("sha256"))
        or sidecar_identity.get("sha256") != sidecar_hash
    ):
        raise PermissionError("analysis manifest selected-chain recipe sidecar identity is missing or mismatched")
    if analysis_manifest.get("selected_chain_recipe_bindings_sha256") != binding_hash:
        raise PermissionError("analysis manifest selected-chain recipe binding hash does not match lock")
    try:
        sidecar_payload = _artifact_payload(sidecar)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise PermissionError("selected-chain recipe binding sidecar is malformed") from exc
    if dict(sidecar_payload) != dict(binding):
        raise PermissionError("selected-chain recipe sidecar does not match lock")
    if sha256_bytes(canonical_json(sidecar_payload).encode("utf-8")) != binding_hash:
        raise PermissionError("selected-chain recipe sidecar content hash does not match lock")


def _load_analysis_bound_decision(
    decision: Mapping[str, Any] | os.PathLike[str] | str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a decision only when its sibling analysis manifest binds it.

    A free-form mapping that merely happens to contain decision-shaped keys is
    not a reproducible analysis artifact.  Runtime and confirmation therefore
    accept either an analysis directory or its ``development_lock.json`` file,
    provided the sibling ``analysis_manifest.json`` records the exact lock
    digest and file identity.
    """

    if isinstance(decision, Mapping):
        raise PermissionError(
            "decision must be an analysis artifact path, not a hand-authored mapping"
        )
    supplied = Path(decision).resolve()
    if supplied.is_dir():
        directory = supplied
        decision_path = directory / "development_lock.json"
    elif supplied.is_file():
        directory = supplied.parent
        decision_path = supplied
    else:
        raise PermissionError(f"decision artifact is unreadable: {supplied}")
    manifest_path = directory / "analysis_manifest.json"
    if decision_path.name != "development_lock.json" or not manifest_path.is_file():
        raise PermissionError(
            "decision must be development_lock.json with sibling analysis_manifest.json"
        )
    try:
        payload = dict(_artifact_payload(decision_path))
        analysis_manifest = dict(_artifact_payload(manifest_path))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise PermissionError("decision analysis artifact is malformed") from exc
    decision_hash = sha256_path(decision_path)
    if analysis_manifest.get("development_lock_sha256") != decision_hash:
        raise PermissionError("analysis manifest does not bind development_lock.json")
    files = analysis_manifest.get("files")
    if not isinstance(files, Mapping):
        raise PermissionError("analysis manifest has no file identity map")
    expected_analysis_stage = {
        "development_provisional": "development_provisional",
        "development": "development_final",
    }.get(str(payload.get("stage")))
    if expected_analysis_stage is None or analysis_manifest.get("stage") != expected_analysis_stage:
        raise PermissionError("analysis manifest stage does not match decision stage")
    actual_files = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.name != "analysis_manifest.json"
    }
    if set(files) != actual_files:
        raise PermissionError(
            "analysis manifest file identity map does not cover exactly the sibling files"
        )
    for filename, identity in files.items():
        if not isinstance(filename, str) or not isinstance(identity, Mapping):
            raise PermissionError("analysis manifest file identity map is malformed")
        file_path = (directory / filename).resolve()
        if Path(filename).name != filename or not file_path.is_file():
            raise PermissionError(f"analysis manifest names an unreadable file: {filename!r}")
        digest = identity.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise PermissionError(f"analysis manifest has malformed SHA-256 for {filename!r}")
        if sha256_path(file_path) != digest:
            raise PermissionError(f"analysis manifest file hash mismatch for {filename!r}")
    lock_identity = files.get("development_lock.json")
    if not isinstance(lock_identity, Mapping) or lock_identity.get("sha256") != decision_hash:
        raise PermissionError("analysis manifest lock file identity is missing or mismatched")
    for field in ("protocol_sha256", "code_identity_sha256"):
        if analysis_manifest.get(field) != payload.get(field):
            raise PermissionError(f"analysis manifest {field} does not match decision")
    if payload.get("stage") == "development":
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            raise PermissionError("final development lock lacks sibling summary.json")
        summary_identity = files.get("summary.json")
        summary_hash = sha256_path(summary_path)
        if not isinstance(summary_identity, Mapping) or summary_identity.get("sha256") != summary_hash:
            raise PermissionError("analysis manifest summary identity is missing or mismatched")
        try:
            summary = dict(_artifact_payload(summary_path))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
            raise PermissionError("final development summary is malformed") from exc
        resources = summary.get("resources")
        if not isinstance(resources, Mapping):
            raise PermissionError("final development summary lacks resources")
        resource_hash = sha256_bytes(canonical_json(resources).encode("utf-8"))
        if payload.get("resource_summary_sha256") != resource_hash:
            raise PermissionError("final development lock resource summary hash mismatch")
        if (
            payload.get("resource_candidate_id") != payload.get("selected_chain")
            or payload.get("resource_status") != "pass"
        ):
            raise PermissionError(
                "final development lock lacks the exact selected-chain resource binding"
            )
        if resources.get("candidate_id") != payload.get("selected_chain") or resources.get("status") != "pass":
            raise PermissionError("final development resource summary is not bound to selected_chain")
    return payload, {
        "provisional_decision_sha256": decision_hash,
        "provisional_analysis_manifest_sha256": sha256_path(manifest_path),
        "analysis_manifest_path": str(manifest_path),
        "decision_path": str(decision_path),
        "analysis_manifest": analysis_manifest,
    }


def _validate_provisional_payload(
    payload: Mapping[str, Any],
    *,
    protocol_hash: str,
    code_identity_hash: str,
) -> dict[str, Any]:
    """Validate the closed provisional runtime decision schema."""

    if payload.get("stage") != "development_provisional":
        raise PermissionError("runtime decision has an invalid provisional stage")
    if payload.get("status") != "provisional_runtime_pending":
        raise PermissionError("runtime decision is not provisional/authorized")
    _require_hash(payload, "protocol_sha256", protocol_hash, "runtime decision protocol")
    _require_hash(payload, "code_identity_sha256", code_identity_hash, "runtime decision code")
    selected = payload.get("selected_candidate")
    if selected not in {"M0-SW", "M1-SW"}:
        raise PermissionError("runtime decision must select M0-SW or M1-SW")
    # Chain selection is deliberately deferred until the runtime resource
    # gate and the conditional F/G policy checks.  A provisional artifact must
    # carry an explicit null, never a hand-authored final chain.
    if payload.get("selected_chain") is not None:
        raise PermissionError("provisional runtime decision must have selected_chain=null")
    runtime_ids = payload.get("runtime_candidate_ids")
    if not isinstance(runtime_ids, Sequence) or isinstance(runtime_ids, (str, bytes)):
        raise PermissionError("runtime decision must include runtime_candidate_ids")
    observed_ids = tuple(str(value) for value in runtime_ids)
    if observed_ids != RUNTIME_PRIMITIVE_METHODS:
        raise PermissionError(
            "runtime decision must use the exact primitive schedule "
            f"{RUNTIME_PRIMITIVE_METHODS!r}; got {observed_ids!r}"
        )
    if payload.get("runtime_pending") is not True:
        raise PermissionError("runtime decision must remain runtime_pending")
    if payload.get("runner_up_allowed") is not False:
        raise PermissionError("runtime decision cannot allow a runner-up")
    return dict(payload)


def _require_hash(payload: Mapping[str, Any], name: str, expected: str, label: str) -> None:
    observed = payload.get(name)
    if str(observed) != str(expected):
        raise PermissionError(f"{label} hash mismatch: expected {expected}, got {observed}")


def validate_development_lock(
    lock: Mapping[str, Any] | os.PathLike[str] | str,
    *,
    protocol_hash: str,
    code_identity_hash: str,
) -> Mapping[str, Any]:
    """Validate a completed, immutable development promotion lock."""

    payload, bound = _load_analysis_bound_decision(lock)
    if payload.get("stage") != "development":
        raise PermissionError("development lock has an invalid stage")
    if payload.get("status") != "locked_for_future_confirmation":
        raise PermissionError(
            "development lock must use the canonical locked_for_future_confirmation status"
        )
    _require_hash(payload, "protocol_sha256", protocol_hash, "development lock protocol")
    _require_hash(payload, "code_identity_sha256", code_identity_hash, "development lock code")
    selected_candidate = payload.get("selected_candidate")
    selected_chain = payload.get("selected_chain")
    if selected_candidate not in {"M0-SW", "M1-SW"}:
        raise PermissionError("development lock must name selected_candidate M0-SW or M1-SW")
    if selected_chain not in {"M0-SW", "M1-SW", "F", "G"}:
        raise PermissionError(
            "development lock must name an executable selected_chain in M0/M1/F/G"
        )
    expected_kind = {
        "M0-SW": "pure_oi",
        "M1-SW": "pure_oi",
        "F": "oi_fusion",
        "G": "product_guardrail",
    }[str(selected_chain)]
    if payload.get("chain_kind") != expected_kind:
        raise PermissionError("development lock chain_kind does not match selected_chain")
    algorithmic_status = payload.get("algorithmic_oi_status")
    product_status = payload.get("product_policy_status")
    if expected_kind in {"pure_oi", "oi_fusion"}:
        if algorithmic_status != "pass" or product_status != "not_applicable":
            raise PermissionError("development lock has noncanonical pure/fusion gate statuses")
    elif algorithmic_status != "fail_linear_claim" or product_status != "pass":
        raise PermissionError("development lock has noncanonical guardrail gate statuses")
    if expected_kind == "pure_oi" and selected_chain != selected_candidate:
        raise PermissionError("pure_oi development lock chain must equal selected_candidate")
    if "locked_candidate" in payload:
        raise PermissionError("development lock must not use locked_candidate alias")
    _validate_selected_chain_recipe_bindings(
        payload,
        bound["analysis_manifest"],
        Path(bound["analysis_manifest_path"]).parent,
    )
    result = dict(payload)
    result["_analysis_bound_identity"] = {
        "development_lock_sha256": bound["provisional_decision_sha256"],
        "analysis_manifest_sha256": bound["provisional_analysis_manifest_sha256"],
    }
    return result


def validate_provisional_runtime_decision(
    decision: Mapping[str, Any] | os.PathLike[str] | str,
    *,
    protocol_hash: str,
    code_identity_hash: str,
) -> Mapping[str, Any]:
    """Validate the hashed development decision consumed by runtime.

    Runtime precedes resource-gate completion, so it must consume a distinct
    provisional decision artifact rather than pretending that a final lock
    already exists.  The selected chain is explicit and all runtime IDs are
    checked against the closed method table.
    """

    payload, _bound = _load_analysis_bound_decision(decision)
    return _validate_provisional_payload(
        payload,
        protocol_hash=protocol_hash,
        code_identity_hash=code_identity_hash,
    )


def provisional_runtime_artifact(
    decision: Mapping[str, Any] | os.PathLike[str] | str,
    *,
    protocol_hash: str,
    code_identity_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a validated provisional decision and its sibling identities."""

    payload, bound = _load_analysis_bound_decision(decision)
    return (
        _validate_provisional_payload(
            payload,
            protocol_hash=protocol_hash,
            code_identity_hash=code_identity_hash,
        ),
        bound,
    )


def _registry_rows(registry: Mapping[str, Any] | os.PathLike[str] | str) -> Sequence[Mapping[str, Any]]:
    payload = _artifact_payload(registry)
    rows = payload.get("datasets")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise PermissionError("untouched confirmation registry must contain a datasets sequence")
    if any(not isinstance(row, Mapping) for row in rows):
        raise PermissionError("confirmation registry contains malformed dataset rows")
    return tuple(rows)


def validate_untouched_registry(
    registry: Mapping[str, Any] | os.PathLike[str] | str,
    *,
    minimum_panels: int = 5,
) -> tuple[Mapping[str, Any], ...]:
    """Fail closed unless registry rows prove genuinely new audited panels."""

    if type(minimum_panels) is not int or minimum_panels != 5:
        raise PermissionError(
            "confirmation registry requires exactly five audited panels"
        )
    payload = _artifact_payload(registry)
    if payload.get("registry_id") != "m50_backbone_ranking_confirmation_v1":
        raise PermissionError("confirmation registry id is not the frozen registry")
    if payload.get("status") not in {"audited", "complete", "locked"} or payload.get("ready") is not True:
        raise PermissionError("confirmation registry is not audited and complete")
    if payload.get("outcomes_inspected") is True:
        raise PermissionError("confirmation registry was inspected after outcomes")
    if tuple(payload.get("backbones", ())) != tuple(MODELS):
        raise PermissionError("confirmation registry backbone panel is not the frozen panel")
    if tuple(int(value) for value in payload.get("budgets_per_class", ())) != (32, 64):
        raise PermissionError("confirmation registry budgets do not match the frozen grid")
    if tuple(int(value) for value in payload.get("replicate_seeds", ())) != tuple(
        2026082710 + value for value in range(5)
    ):
        raise PermissionError("confirmation registry replicate seeds do not match the frozen grid")
    rows = _registry_rows(payload)
    if len(rows) != int(minimum_panels):
        raise PermissionError(
            f"confirmation registry requires exactly {minimum_panels} panels"
        )
    observed_ids = tuple(str(row.get("id", "")) for row in rows)
    if len(set(observed_ids)) != len(observed_ids):
        raise PermissionError("confirmation registry panel identities are missing/duplicated")
    if observed_ids != CONFIRMATION_DATASET_IDS:
        raise PermissionError(
            "confirmation registry dataset IDs do not match the frozen five-panel order"
        )
    forbidden = {
        "food101",
        "cifar100",
        "cifar100_coarse",
        "cifar100-coarse",
        "oxford_iiit_pet",
        "oxford-iiit-pet",
        "oxford",
        "prior_synthetic_panels",
        "prior_synthetic",
        "prior-synthetic",
        "synthetic",
    }
    ids: set[str] = set()
    for row in rows:
        panel_id = str(row.get("id", "")).strip()
        dataset = panel_id.lower()
        if not panel_id or panel_id in ids:
            raise PermissionError("confirmation registry panel identities are missing/duplicated")
        ids.add(panel_id)
        # ``torchvision_cifar10`` is a genuinely new explicitly allowed panel;
        # reject only the exact known/archived families, not every string that
        # happens to contain ``cifar``.
        if not dataset or dataset in forbidden:
            raise PermissionError("confirmation registry contains a known or forbidden dataset")
        expected_spec = CONFIRMATION_DATASET_SPECS.get(panel_id)
        if expected_spec is None:
            raise PermissionError(
                "confirmation registry contains a dataset outside the frozen panel"
            )
        for field, expected in expected_spec.items():
            if row.get(field) != expected:
                raise PermissionError(
                    f"confirmation registry {panel_id!r} has a non-frozen {field}"
                )
        if (
            row.get("minimum_training_examples_per_class")
            != CONFIRMATION_MIN_TRAINING_EXAMPLES_PER_CLASS
            or row.get("minimum_evaluation_examples_per_class")
            != CONFIRMATION_MIN_EVALUATION_EXAMPLES_PER_CLASS
        ):
            raise PermissionError(
                f"confirmation registry {panel_id!r} has non-frozen minimum counts"
            )
        if row.get("new") is not True or row.get("audited") is not True:
            raise PermissionError("confirmation registry must mark every panel new and audited")
        embedding_manifest = row.get("embedding_manifest")
        if not isinstance(embedding_manifest, Mapping):
            raise PermissionError("confirmation registry requires an embedding manifest object")
        manifest_hash = embedding_manifest.get("sha256")
        matrix_hash = embedding_manifest.get("matrix_sha256")
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in (manifest_hash, matrix_hash)
        ):
            raise PermissionError("confirmation registry requires manifest and matrix SHA-256")
        for field in (
            "version",
            "training_split",
            "evaluation_split",
            "minimum_training_examples_per_class",
            "minimum_evaluation_examples_per_class",
        ):
            if field not in row or row[field] in (None, ""):
                raise PermissionError(f"confirmation registry row lacks {field}")
        if not row.get("source") or not row.get("provenance"):
            raise PermissionError("confirmation registry requires source and provenance")
    return rows


def environment(provenance: Mapping[str, Any]) -> dict[str, Any]:
    import importlib.metadata as metadata

    packages: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "scikit-learn", "overlapindex"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            # The local checkout is intentionally not required to be installed
            # as a wheel.  Preserve the other package versions instead of
            # collapsing the complete environment surface to an empty map.
            packages[name] = None
        except Exception:  # pragma: no cover - diagnostic fallback.
            packages[name] = None
    project_version: str | None = None
    pyproject = ROOT / "pyproject.toml"
    if pyproject.is_file():
        try:
            for line in pyproject.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("version") and "=" in stripped:
                    candidate = stripped.split("=", 1)[1].strip().strip('"\'')
                    if candidate:
                        project_version = candidate
                        break
        except (OSError, UnicodeError):  # pragma: no cover - diagnostic only.
            project_version = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "project": {"name": "overlapindex", "version": project_version},
        "starting_commit": EXPECTED_STARTING_COMMIT,
        "original_develop_base": EXPECTED_ORIGINAL_DEVELOP_BASE,
        "experiment_commit": provenance.get("experiment_commit"),
        "git_dirty": provenance.get("git_dirty"),
    }


__all__ = [
    "ARMS", "BUDGETS", "CANDIDATE_IDS", "DEFAULT_RUNTIME_METHODS", "DECLARED_SOURCE_PATHS", "EXPECTED_BRANCH",
    "EXPECTED_DEVELOPMENT_PANEL_COUNT", "EXPECTED_DEVELOPMENT_ROW_COUNT",
    "EXPECTED_ORIGINAL_DEVELOP_BASE", "EXPECTED_RUNTIME_PANEL_COUNT", "EXPECTED_RUNTIME_ROW_COUNT",
    "EXPECTED_STARTING_COMMIT", "FoodPanel", "FOLDS", "K", "MODELS", "OI_METHODS",
    "ORIGINAL_DEVELOP_BASE", "OUTPUT_ROOTS", "PROTOCOL_PATH", "REPLICATES",
    "RUNTIME_ARMS", "RUNTIME_BUDGETS", "RUNTIME_REPEATS", "RUNTIME_PRIMITIVE_METHODS", "SCHEMA_VERSION", "SCHEDULE_SEED",
    "SELECTOR_METHODS", "STARTING_COMMIT", "canonical_json", "code_identity_sha256",
    "development_panels", "environment",
    "panel_grid_sha256", "planned_execution_order", "protocol_sha256",
    "recorded_protocol_sha256", "repository_provenance",
    "runtime_execution_order", "runtime_panels", "sha256_bytes", "sha256_path", "source_hashes",
    "validate_development_lock", "validate_development_panels", "validate_execution_order",
    "provisional_runtime_artifact", "validate_provisional_runtime_decision",
    "validate_runtime_grid", "validate_untouched_registry", "verify_protocol_hash",
    "verify_protocol_source_evidence", "working_tree_hash",
]

# Keep this derived constant here rather than importing candidate definitions
# during manifest import; it makes malformed candidate modules fail at their
# own boundary and keeps provenance usable for preflight checks.
CANDIDATE_IDS = SELECTOR_METHODS
