"""Fail-closed staged analysis for the M50 backbone-ranking experiment.

Commands
--------
Development (retrospective Food evidence)::

    python -m experiments.m50_backbone_ranking.analysis development \
      --input artifacts/m50_backbone_ranking/development \
      --output artifacts/m50_backbone_ranking/development_analysis

Runtime finalization::

    python -m experiments.m50_backbone_ranking.analysis finalize \
      --provisional artifacts/m50_backbone_ranking/development_analysis \
      --runtime artifacts/m50_backbone_ranking/runtime \
      --output artifacts/m50_backbone_ranking/final_analysis

V1 intentionally has no confirmation-analysis command. It ends with a
retrospective development/resource lock; future V2 confirmation requires a
new lineage-aware protocol and implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import fusion, manifest, reporting, statistics


FOOD_RECIPE_SEED_RULE = {
    "selector_panel_seed": "SEED + replicate",
    "oi_fold_seed": "selector_panel_seed + fold",
    "probe_split_seed": "selector_panel_seed",
    "probe_model_random_state": "selector_panel_seed",
    "capped_subset_seed": "selector_panel_seed",
}
RUNTIME_RECIPE_SEED_RULE = {
    "runtime_call_seed": "SEED + budget",
    "oi_fold_seed": "runtime_call_seed + fold",
    "probe_split_seed": "runtime_call_seed",
    "probe_model_random_state": "runtime_call_seed",
    "capped_subset_seed": "runtime_call_seed",
    "repeat_effect": "order_and_timing_only",
}


def _read_object(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path | str) -> str:
    return manifest.sha256_path(path)


def _object_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(manifest.canonical_json(dict(value)).encode("utf-8")).hexdigest()


def canonical_table_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash the canonical JSONL representation used by stage writers."""

    payload = "".join(
        manifest.canonical_json(dict(row)) + "\n" for row in rows
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_canonical_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    payload = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise ValueError(f"{path} contains an empty JSONL row at line {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} row {line_number} is not a JSON object")
        rows.append(value)
    canonical = "".join(manifest.canonical_json(row) + "\n" for row in rows).encode(
        "utf-8"
    )
    if payload != canonical:
        raise ValueError(f"{path} is not encoded as canonical JSONL")
    return rows, payload


def verify_completed_artifact(
    root: Path | str,
    *,
    table_names: Sequence[str],
    expected_study: str | None = None,
    verify_current_identity: bool = True,
) -> dict[str, Any]:
    """Load a completed bundle only after manifest/table/hash validation."""

    directory = Path(root).resolve()
    manifest_path = directory / "manifest.json"
    raw_path = directory / "raw_results.json"
    artifact_manifest = _read_object(manifest_path)
    raw_identity = artifact_manifest.get("raw_results")
    if not isinstance(raw_identity, Mapping) or set(raw_identity) != {
        "sha256",
        "size_bytes",
    }:
        raise ValueError("manifest requires the exact raw_results SHA-256 and byte size")
    raw_payload = raw_path.read_bytes()
    if raw_identity.get("size_bytes") != len(raw_payload) or raw_identity.get(
        "sha256"
    ) != hashlib.sha256(raw_payload).hexdigest():
        raise ValueError("raw_results identity does not match the exact file bytes")
    raw_value = json.loads(raw_payload)
    if not isinstance(raw_value, dict):
        raise ValueError("raw_results.json must contain a JSON object")
    raw = raw_value
    for key, manifest_value in artifact_manifest.items():
        if key == "raw_results":
            continue
        if key not in raw or raw[key] != manifest_value:
            raise ValueError(
                f"manifest metadata {key!r} does not exactly match hash-bound raw_results"
            )
    if artifact_manifest.get("artifact_status") != "completed" or raw.get(
        "artifact_status"
    ) != "completed":
        raise ValueError("analysis requires a completed artifact and completed manifest")
    if expected_study is not None and (
        artifact_manifest.get("study") != expected_study
        or raw.get("study") != expected_study
    ):
        raise ValueError(f"artifact is not the expected study {expected_study!r}")
    tables = artifact_manifest.get("tables")
    if not isinstance(tables, Mapping) or set(tables) != set(table_names):
        raise ValueError("manifest must contain exactly the required table identities")
    normalized_tables: dict[str, list[dict[str, Any]]] = {}
    for name in table_names:
        rows = raw.get(name)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError(f"raw artifact table {name!r} is missing or malformed")
        materialized = [dict(row) for row in rows if isinstance(row, Mapping)]
        if len(materialized) != len(rows):
            raise ValueError(f"raw artifact table {name!r} contains non-object rows")
        identity = tables[name]
        if not isinstance(identity, Mapping):
            raise ValueError(f"manifest table identity {name!r} is malformed")
        if identity.get("encoding") != "canonical_jsonl_utf8":
            raise ValueError(f"manifest table encoding mismatch for {name!r}")
        if identity.get("row_count") != len(materialized):
            raise ValueError(f"manifest row count mismatch for {name!r}")
        table_path = directory / f"{name}.jsonl"
        if not table_path.is_file():
            raise ValueError(f"canonical JSONL table {name!r} is missing")
        file_rows, table_payload = _read_canonical_jsonl(table_path)
        if file_rows != materialized:
            raise ValueError(
                f"canonical JSONL table {name!r} does not match raw_results.json"
            )
        observed = hashlib.sha256(table_payload).hexdigest()
        if identity.get("sha256") != observed:
            raise ValueError(f"manifest table SHA-256 mismatch for {name!r}")
        normalized_tables[name] = materialized

    protocol_payload = artifact_manifest.get("protocol")
    provenance = artifact_manifest.get("repository_provenance")
    if not isinstance(protocol_payload, Mapping) or not isinstance(provenance, Mapping):
        # Runtime artifacts place hashes under their strict identity mapping.
        identity = artifact_manifest.get("identity")
        if not isinstance(identity, Mapping):
            raise ValueError("artifact manifest lacks protocol/code identity")
        protocol_hash = identity.get("protocol_sha256")
        code_hash = identity.get("code_identity_sha256")
    else:
        protocol_hash = protocol_payload.get("sha256")
        code_hash = provenance.get("code_identity_sha256")
    if not isinstance(protocol_hash, str) or not isinstance(code_hash, str):
        raise ValueError("artifact protocol/code hashes are missing")
    if verify_current_identity:
        current_protocol = manifest.verify_protocol_hash()
        current_code = manifest.code_identity_sha256(require_existing=True)
        if protocol_hash != current_protocol:
            raise ValueError("artifact protocol hash does not match current frozen protocol")
        if code_hash != current_code:
            raise ValueError("artifact code identity does not match current frozen sources")
    return {
        "root": directory,
        "manifest": artifact_manifest,
        "raw": raw,
        "tables": normalized_tables,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "protocol_sha256": protocol_hash,
        "code_identity_sha256": code_hash,
    }


def _flatten_factorial(factorial_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell, payload in sorted(factorial_summary.get("effects", {}).items()):
        for effect in (
            "raw_refinement_effect",
            "conditioned_refinement_effect",
            "interaction",
            "cb_refinement_diagnostic",
        ):
            rows.append({"cell": cell, "effect": effect, **dict(payload[effect])})
    return rows


def _group_panel_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["replicate"]), str(row["arm"]), int(row["budget"]))].append(
            dict(row)
        )
    return grouped


def _validate_recipe_bindings(
    bindings: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_candidates: set[str],
    expected_seed_rule: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate resolved fold configuration hashes and their family hashes."""

    if not isinstance(bindings, Mapping) or set(bindings) != {
        "candidate_config_sha256",
        "recipe_family_sha256",
        "deterministic_seed_rule",
    }:
        raise ValueError("recipe_bindings must use the exact closed schema")
    configs = bindings.get("candidate_config_sha256")
    families = bindings.get("recipe_family_sha256")
    seed_rule = bindings.get("deterministic_seed_rule")
    if (
        not isinstance(configs, Mapping)
        or not isinstance(families, Mapping)
        or set(configs) != expected_candidates
        or set(families) != expected_candidates
        or seed_rule != expected_seed_rule
    ):
        raise ValueError("recipe_bindings candidates or deterministic seed rule mismatch")

    observed: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        candidate_id = str(row.get("candidate_id"))
        if candidate_id not in expected_candidates:
            continue
        folds = row.get("folds")
        if (
            not isinstance(folds, Sequence)
            or isinstance(folds, (str, bytes))
            or len(folds) != 5
        ):
            raise ValueError(f"recipe row for {candidate_id!r} lacks resolved folds")
        if {
            fold.get("fold") for fold in folds if isinstance(fold, Mapping)
        } != set(range(5)):
            raise ValueError(f"recipe row for {candidate_id!r} has an incomplete fold grid")
        for fold in folds:
            if not isinstance(fold, Mapping):
                raise ValueError(f"recipe fold for {candidate_id!r} is malformed")
            identity = fold.get("candidate_config_identity")
            digest = fold.get("candidate_config_sha256")
            if (
                not isinstance(identity, Mapping)
                or identity.get("candidate_id") != candidate_id
                or not isinstance(digest, str)
            ):
                raise ValueError(f"recipe fold for {candidate_id!r} lacks config identity/hash")
            computed = hashlib.sha256(
                manifest.canonical_json(identity).encode("utf-8")
            ).hexdigest()
            if digest != computed:
                raise ValueError(f"recipe fold hash mismatch for {candidate_id!r}")
            observed[candidate_id].add(digest)

    normalized_configs: dict[str, list[str]] = {}
    normalized_families: dict[str, str] = {}
    for candidate_id in sorted(expected_candidates):
        values = configs.get(candidate_id)
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or list(values) != sorted(observed.get(candidate_id, set()))
            or not values
        ):
            raise ValueError(f"recipe config hash set mismatch for {candidate_id!r}")
        normalized = [str(value) for value in values]
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in normalized
        ):
            raise ValueError(f"recipe config hash is malformed for {candidate_id!r}")
        family = hashlib.sha256(
            manifest.canonical_json(
                {
                    "candidate_id": candidate_id,
                    "candidate_config_sha256": normalized,
                }
            ).encode("utf-8")
        ).hexdigest()
        if families.get(candidate_id) != family:
            raise ValueError(f"recipe family hash mismatch for {candidate_id!r}")
        normalized_configs[candidate_id] = normalized
        normalized_families[candidate_id] = family
    return {
        "candidate_config_sha256": normalized_configs,
        "recipe_family_sha256": normalized_families,
        "deterministic_seed_rule": dict(expected_seed_rule),
    }


def _validate_food_completed_artifact(loaded: Mapping[str, Any]) -> None:
    """Independently enforce the frozen complete Food grid and stop gates."""

    tables = loaded["tables"]
    selectors = tables["selector_rows"]
    references = tables["reference_rows"]
    probes = tables["probe_rows"]
    capped = tables["capped_probe_rows"]
    parity = tables["baseline_parity_rows"]
    expected_panels = {
        (model, replicate, arm, budget)
        for model in statistics.FOOD_BACKBONES
        for replicate in statistics.FOOD_REPLICATES
        for arm in statistics.FOOD_ARMS
        for budget in statistics.FOOD_BUDGETS
    }
    refinement_by_candidate = {
        "A": False,
        "B": True,
        "M0-SW": False,
        "M1-SW": True,
        "M0-CB": False,
        "M1-CB": True,
        "LP-FULL": False,
        "LP-CAPPED-2048": False,
    }
    required_numeric_fields = (
        "score",
        "fit_wall_seconds",
        "fit_cpu_seconds",
        "score_fixed_wall_seconds",
        "score_fixed_cpu_seconds",
        "total_wall_seconds",
        "total_cpu_seconds",
    )

    def row_identity(
        row: Mapping[str, Any], *, table: str
    ) -> tuple[str, int, str, int, str]:
        model = row.get("model")
        if not isinstance(model, str) or row.get("backbone") != model:
            raise ValueError(f"Food {table} row has malformed model/backbone identity")
        replicate = row.get("replicate")
        budget = row.get("budget")
        if type(replicate) is not int or type(budget) is not int:
            raise ValueError(f"Food {table} row has non-integer panel identity")
        arm = row.get("arm")
        candidate_id = row.get("candidate_id")
        if not isinstance(arm, str) or not isinstance(candidate_id, str):
            raise ValueError(f"Food {table} row has malformed arm/candidate identity")
        return model, replicate, arm, budget, candidate_id

    def validate_outcome_row(
        row: Mapping[str, Any], *, table: str, expected_candidate: str | None = None
    ) -> tuple[str, int, str, int, str]:
        identity = row_identity(row, table=table)
        candidate_id = identity[-1]
        if candidate_id not in refinement_by_candidate or (
            expected_candidate is not None and candidate_id != expected_candidate
        ):
            raise ValueError(f"Food {table} row has an unknown candidate identity")
        if row.get("status") != "ok":
            raise ValueError(f"Food {table} contains a non-ok row")
        refinement = row.get("prototype_refinement_enabled")
        if type(refinement) is not bool or refinement is not refinement_by_candidate[candidate_id]:
            raise ValueError(f"Food {table} row has a malformed refinement flag")
        if row.get("warmup_excluded") is not True:
            raise ValueError(f"Food {table} row does not exclude warm-up from clocks")
        for field in required_numeric_fields:
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(
                    f"Food {table} row has invalid required score/timing field {field!r}"
                )
        return identity

    selector_ids = {
        validate_outcome_row(row, table="selector table") for row in selectors
    }
    expected_selector_ids = {
        (*panel, candidate_id)
        for panel in expected_panels
        for candidate_id in manifest.SELECTOR_METHODS
    }
    if len(selectors) != 4_200 or selector_ids != expected_selector_ids:
        raise ValueError("Food selector table is not the exact frozen 4,200-row grid")
    reference_ids = {
        (
            str(row.get("model", row.get("backbone"))),
            int(row["replicate"]),
            str(row["arm"]),
            str(row["head"]),
        )
        for row in references
    }
    expected_reference_ids = {
        (model, replicate, arm, head)
        for model in statistics.FOOD_BACKBONES
        for replicate in statistics.FOOD_REPLICATES
        for arm in statistics.FOOD_ARMS
        for head in statistics.REFERENCE_HEADS
    }
    if len(references) != 600 or reference_ids != expected_reference_ids:
        raise ValueError("Food reference table is not the exact frozen 600-row grid")
    if any(
        isinstance(row.get("test_accuracy"), bool)
        or not isinstance(row.get("test_accuracy"), (int, float))
        or not np.isfinite(float(row["test_accuracy"]))
        for row in references
    ):
        raise ValueError("Food reference table contains a non-finite test accuracy")

    probe_ids = {
        validate_outcome_row(
            row, table="full-probe table", expected_candidate="LP-FULL"
        )
        for row in probes
    }
    expected_probe_ids = {(*panel, "LP-FULL") for panel in expected_panels}
    if len(probes) != 600 or probe_ids != expected_probe_ids:
        raise ValueError("Food full-probe table is incomplete or duplicated")
    capped_ids = {
        validate_outcome_row(
            row, table="capped-probe table", expected_candidate="LP-CAPPED-2048"
        )
        for row in capped
    }
    expected_capped_ids = {(*panel, "LP-CAPPED-2048") for panel in expected_panels}
    if len(capped) != 600 or capped_ids != expected_capped_ids:
        raise ValueError("Food capped-probe table is incomplete or duplicated")

    selector_lp = {
        row_identity(row, table="selector table"): dict(row)
        for row in selectors
        if row.get("candidate_id") == "LP-FULL"
    }
    probe_lp = {
        row_identity(row, table="full-probe table"): dict(row) for row in probes
    }
    if selector_lp != probe_lp:
        raise ValueError(
            "Food duplicated LP-FULL selector/probe rows do not agree exactly"
        )
    expected_parity_ids = {
        (*panel, candidate_id)
        for panel in expected_panels
        for candidate_id in ("A", "B", "LP-FULL")
    }
    parity_ids = {
        (
            str(row.get("model", row.get("backbone"))),
            int(row["replicate"]),
            str(row["arm"]),
            int(row["budget"]),
            str(row["candidate_id"]),
        )
        for row in parity
    }
    if (
        len(parity) != 1_800
        or parity_ids != expected_parity_ids
        or any(row.get("exact") is not True or float(row.get("delta", 1.0)) != 0.0 for row in parity)
    ):
        raise ValueError("Food A/B/LP-FULL exact parity gate is not satisfied")
    _validate_recipe_bindings(
        loaded["raw"].get("recipe_bindings"),
        [*selectors, *capped],
        expected_candidates={*manifest.SELECTOR_METHODS, "LP-CAPPED-2048"},
        expected_seed_rule=FOOD_RECIPE_SEED_RULE,
    )
    deterministic = loaded["raw"].get("determinism_verification")
    expected_determinism_methods = {
        *manifest.SELECTOR_METHODS,
        "LP-CAPPED-2048",
    }
    frozen_panels = manifest.development_panels()
    panel_order = {
        panel.panel_id: position for position, panel in enumerate(frozen_panels)
    }
    first_panel_ids: dict[str, str] = {}
    frozen_schedule = sorted(
        manifest.planned_execution_order(
            frozen_panels,
            manifest.SELECTOR_METHODS,
            schedule_seed=manifest.SCHEDULE_SEED,
        ),
        key=lambda row: (
            panel_order[str(row["panel_id"])],
            int(row["execution_position"]),
        ),
    )
    for schedule_row in frozen_schedule:
        candidate_id = str(schedule_row["method_id"])
        first_panel_ids.setdefault(candidate_id, str(schedule_row["panel_id"]))
    first_panel_ids["LP-CAPPED-2048"] = frozen_panels[0].panel_id
    if set(first_panel_ids) != expected_determinism_methods:
        raise ValueError("frozen determinism schedule lacks an expected method")
    first_determinism_panels = {
        candidate_id: next(
            panel
            for panel in frozen_panels
            if panel.panel_id == panel_id
        ).identity()
        for candidate_id, panel_id in first_panel_ids.items()
    }
    hash_fields = ("first_signature_sha256", "second_signature_sha256")

    def valid_sha256(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    def valid_determinism(candidate_id: str, value: Any) -> bool:
        if not isinstance(value, Mapping) or set(value) != {
            "exact",
            *hash_fields,
            "runtime_fields_excluded",
            "panel_identity",
        }:
            return False
        panel = value.get("panel_identity")
        if not isinstance(panel, Mapping) or set(panel) != {
            "model",
            "replicate",
            "arm",
            "budget",
            "candidate_id",
        }:
            return False
        panel_key = (
            panel.get("model"),
            panel.get("replicate"),
            panel.get("arm"),
            panel.get("budget"),
        )
        return (
            value.get("exact") is True
            and value.get("runtime_fields_excluded") is True
            and all(valid_sha256(value.get(field)) for field in hash_fields)
            and value.get(hash_fields[0]) == value.get(hash_fields[1])
            and type(panel.get("replicate")) is int
            and type(panel.get("budget")) is int
            and panel.get("candidate_id") == candidate_id
            and panel_key in expected_panels
            and {
                "model": panel.get("model"),
                "replicate": panel.get("replicate"),
                "arm": panel.get("arm"),
                "budget": panel.get("budget"),
            }
            == first_determinism_panels.get(candidate_id)
        )

    if (
        not isinstance(deterministic, Mapping)
        or set(deterministic) != expected_determinism_methods
        or any(
            not valid_determinism(str(candidate_id), value)
            for candidate_id, value in deterministic.items()
        )
    ):
        raise ValueError("Food deterministic-repeat gate is not satisfied")


def _derive_guardrail_all_panels(
    selector_rows: Sequence[Mapping[str, Any]],
    capped_rows: Sequence[Mapping[str, Any]],
    *,
    base_rows: Sequence[Mapping[str, Any]],
    base_candidate_id: str,
) -> list[dict[str, Any]]:
    selectors = _group_panel_rows(selector_rows)
    capped = _group_panel_rows(capped_rows)
    bases = _group_panel_rows(base_rows)
    expected_panels = {
        (replicate, arm, budget)
        for replicate in statistics.FOOD_REPLICATES
        for arm in statistics.FOOD_ARMS
        for budget in statistics.FOOD_BUDGETS
    }
    if set(selectors) != expected_panels or set(capped) != expected_panels or set(bases) != expected_panels:
        raise ValueError("G requires complete diagnostic, capped-probe, and base panels")
    result: list[dict[str, Any]] = []
    for panel in sorted(expected_panels, key=repr):
        diagnostic_rows = [
            row
            for row in selectors[panel]
            if row.get("candidate_id") in {"B", "M1-SW"}
        ]
        panel_capped = [
            row for row in capped[panel] if row.get("candidate_id") == "LP-CAPPED-2048"
        ]
        result.extend(
            fusion.derive_guardrail_rows(
                diagnostic_rows=diagnostic_rows,
                base_rows=bases[panel],
                capped_probe_rows=panel_capped,
                backbones=statistics.FOOD_BACKBONES,
                base_candidate_id=base_candidate_id,
            )
        )
    return result


def analyze_development(
    input_root: Path | str,
    output: Path | str,
    *,
    n_resamples: int = statistics.BOOTSTRAP_RESAMPLES,
    verify_current_identity: bool = True,
) -> Path:
    loaded = verify_completed_artifact(
        input_root,
        table_names=(
            "selector_rows",
            "reference_rows",
            "probe_rows",
            "capped_probe_rows",
            "baseline_parity_rows",
        ),
        expected_study="food101_m50_backbone_ranking",
        verify_current_identity=verify_current_identity,
    )
    _validate_food_completed_artifact(loaded)
    selectors = loaded["tables"]["selector_rows"]
    references = loaded["tables"]["reference_rows"]
    base_metrics = statistics.food_panel_metrics(
        selectors,
        references,
        candidate_ids=manifest.SELECTOR_METHODS,
    )
    all_metrics = list(base_metrics)
    all_rank_auc = statistics.rank_auc_rows(base_metrics)
    factorial = statistics.factorial_effects(base_metrics, n_resamples=n_resamples)
    provisional = statistics.provisional_development_lock(
        base_metrics, n_resamples=n_resamples
    )
    provisional.update(
        {
            "stage": "development_provisional",
            "protocol_sha256": loaded["protocol_sha256"],
            "code_identity_sha256": loaded["code_identity_sha256"],
            "input_manifest_sha256": loaded["manifest_sha256"],
            "development_input_root": str(loaded["root"]),
            "fusion_gate_summary": None,
            "guardrail_gate_summary": None,
            "derived_candidates_evaluated": [],
            "retrospective": True,
        }
    )
    summary = {
        "stage": "development_provisional",
        "retrospective": True,
        "candidate_gates": provisional.get("candidate_gates", {}),
        "factorial": factorial,
        "refinement_noninferiority": provisional.get("refinement_noninferiority"),
        "fusion_gates": None,
        "guardrail_gates": None,
        "derived_candidates_status": "deferred_until_runtime_settles_refinement",
        "selection_rates": statistics.selection_rate_summary(all_metrics),
        "rank_auc_summary": statistics.rank_auc_summary(all_rank_auc),
        "resources": {"status": "pending"},
        "decision_status": provisional["status"],
    }
    return reporting.write_report_bundle(
        output,
        summary=summary,
        development_lock=provisional,
        panel_metrics=all_metrics,
        rank_auc=all_rank_auc,
        factorial_rows=_flatten_factorial(factorial),
        resource_rows=(),
        full_panel_resource_rows=(),
        lineage={
            "stage": "development_provisional",
            "status": str(provisional["status"]),
            "protocol_sha256": loaded["protocol_sha256"],
            "code_identity_sha256": loaded["code_identity_sha256"],
            "input_manifest_sha256": loaded["manifest_sha256"],
        },
    )


def _csv_data_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    return max(0, len(rows) - 1) if rows else 0


def _verify_analysis_bundle(
    root: Path | str,
    *,
    expected_stage: str,
) -> tuple[dict[str, Any], str]:
    directory = Path(root).resolve()
    manifest_path = directory / "analysis_manifest.json"
    analysis_manifest = _read_object(manifest_path)
    if analysis_manifest.get("stage") != expected_stage:
        raise ValueError(f"analysis bundle must have stage {expected_stage!r}")
    files = analysis_manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("analysis manifest files mapping is missing or malformed")
    actual_files = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.name != "analysis_manifest.json"
    }
    if set(files) != actual_files:
        raise ValueError("analysis manifest must identify every and only sibling file")
    for name, identity in files.items():
        if not isinstance(identity, Mapping) or set(identity) != {"sha256", "row_count"}:
            raise ValueError(f"analysis file identity {name!r} is malformed")
        path = directory / str(name)
        if identity.get("sha256") != _sha256(path):
            raise ValueError(f"analysis file SHA-256 mismatch for {name!r}")
        row_count = identity.get("row_count")
        if row_count is not None:
            if path.suffix != ".csv" or type(row_count) is not int or row_count < 0:
                raise ValueError(f"analysis row-count identity {name!r} is malformed")
            if _csv_data_row_count(path) != row_count:
                raise ValueError(f"analysis CSV row-count mismatch for {name!r}")
    decision_path = directory / "development_lock.json"
    if analysis_manifest.get("development_lock_sha256") != _sha256(decision_path):
        raise ValueError("analysis development-lock hash mismatch")
    decision = _read_object(decision_path)
    recipe_manifest_hash = analysis_manifest.get(
        "selected_chain_recipe_bindings_sha256"
    )
    recipe_path = directory / "selected_chain_recipe_bindings.json"
    if recipe_manifest_hash is not None:
        if not isinstance(recipe_manifest_hash, str) or not recipe_path.is_file():
            raise ValueError("analysis selected-chain recipe binding is malformed")
        recipe = _read_object(recipe_path)
        if (
            _object_sha256(recipe) != recipe_manifest_hash
            or decision.get("selected_chain_recipe_bindings") != recipe
            or decision.get("selected_chain_recipe_bindings_sha256")
            != recipe_manifest_hash
        ):
            raise ValueError("analysis selected-chain recipe binding hash mismatch")
    elif recipe_path.exists() or "selected_chain_recipe_bindings" in decision:
        raise ValueError("analysis selected-chain recipe binding is not manifest-bound")
    return analysis_manifest, _sha256(manifest_path)


def _load_provisional(root: Path | str) -> tuple[dict[str, Any], dict[str, Any], str]:
    directory = Path(root).resolve()
    decision_path = directory / "development_lock.json"
    analysis_manifest, analysis_manifest_hash = _verify_analysis_bundle(
        directory, expected_stage="development_provisional"
    )
    decision = _read_object(decision_path)
    decision_hash = _sha256(decision_path)
    if decision.get("stage") != "development_provisional":
        raise ValueError("runtime finalization requires a development provisional decision")
    analysis_manifest = {
        **analysis_manifest,
        "analysis_manifest_sha256": analysis_manifest_hash,
    }
    return decision, analysis_manifest, decision_hash


def _paired_runtime_ratio(
    rows: Sequence[Mapping[str, Any]],
    numerator: str,
    denominator: str,
) -> dict[str, float]:
    lookup: dict[tuple[str, str, str, int, int], float] = {}
    for row in rows:
        candidate = str(row.get("candidate_id"))
        if candidate not in {numerator, denominator}:
            continue
        key = (
            candidate,
            str(row.get("backbone")),
            str(row.get("arm")),
            int(row.get("budget")),
            int(row.get("repeat")),
        )
        if key in lookup:
            raise ValueError(f"duplicate runtime identity {key!r}")
        lookup[key] = float(row["total_wall_seconds"])
    ratios_by_arm: dict[str, list[float]] = {
        arm: [] for arm in statistics.FOOD_ARMS
    }
    for arm in statistics.FOOD_ARMS:
        for backbone in statistics.FOOD_BACKBONES:
            for budget in statistics.FOOD_RUNTIME_BUDGETS:
                if budget < 128:
                    continue
                for repeat in statistics.FOOD_REPLICATES:
                    top = (numerator, backbone, arm, budget, repeat)
                    bottom = (denominator, backbone, arm, budget, repeat)
                    if top not in lookup or bottom not in lookup or lookup[bottom] <= 0.0:
                        raise ValueError("incomplete paired refinement runtime evidence")
                    ratios_by_arm[arm].append(lookup[top] / lookup[bottom])
    return {
        arm: float(np.median(np.asarray(values, dtype=np.float64)))
        for arm, values in ratios_by_arm.items()
    }


def _validate_derived_gate_summary(summary: Mapping[str, Any], candidate_id: str) -> None:
    if candidate_id not in {"F", "G"} or summary.get("candidate_id") != candidate_id:
        raise ValueError("derived gate summary has the wrong canonical candidate identity")
    expected = {f"{arm}:{head}" for arm, head in statistics.PRIMARY_CELLS}
    gates = summary.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != expected:
        raise ValueError("derived gate summary must contain the exact five frozen cells")
    if any(
        not isinstance(gate, Mapping)
        or gate.get("candidate_id") != candidate_id
        or gate.get("comparator_id") != "LP-FULL"
        for gate in gates.values()
    ):
        raise ValueError("derived gate cells have malformed candidate/comparator identities")


def _validate_runtime_primitives(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str, int, int], dict[str, Any]]:
    expected_methods = set(statistics.RUNTIME_PRIMITIVE_IDS)
    expected = {
        (method, backbone, arm, budget, repeat)
        for method in expected_methods
        for backbone in statistics.FOOD_BACKBONES
        for arm in statistics.FOOD_ARMS
        for budget in statistics.FOOD_RUNTIME_BUDGETS
        for repeat in statistics.FOOD_REPLICATES
    }
    lookup: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    numeric_fields = (
        "fit_wall_seconds",
        "fit_cpu_seconds",
        "conditioning_fit_wall_seconds",
        "conditioning_fit_cpu_seconds",
        "score_fixed_wall_seconds",
        "score_fixed_cpu_seconds",
        "total_wall_seconds",
        "total_cpu_seconds",
        "peak_memory_mb",
    )
    dataset_by_block: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    dataset_by_panel: dict[tuple[str, str, int], dict[str, Any]] = {}
    structure_by_timing_panel: dict[
        tuple[str, str, str, int], dict[str, Any]
    ] = {}
    for value in rows:
        row = dict(value)
        identity = (
            str(row.get("candidate_id")),
            str(row.get("backbone")),
            str(row.get("arm")),
            int(row.get("budget")),
            int(row.get("repeat")),
        )
        if identity in lookup:
            raise ValueError(f"duplicate runtime primitive identity {identity!r}")
        if identity not in expected:
            raise ValueError(f"unexpected runtime primitive identity {identity!r}")
        if row.get("warmup_excluded") is not True:
            raise ValueError("runtime rows must prove warm-up exclusion")
        for field in numeric_fields:
            value_float = float(row.get(field))
            if not np.isfinite(value_float) or value_float < 0.0:
                raise ValueError(f"runtime field {field!r} must be finite and nonnegative")
        candidate_id = identity[0]
        for field in ("oi_fit_wall_seconds", "oi_fit_cpu_seconds"):
            value = row.get(field)
            if value is None and candidate_id in {"LP-FULL", "LP-CAPPED-2048"}:
                continue
            value_float = float(value)
            if not np.isfinite(value_float) or value_float < 0.0:
                raise ValueError(
                    f"runtime OI stage field {field!r} must be finite and nonnegative"
                )
        dataset_identity = row.get("dataset_identity")
        required_dataset_fields = {
            "values_sha256",
            "labels_sha256",
            "row_count",
            "feature_count",
            "metadata",
        }
        if not isinstance(dataset_identity, Mapping) or set(dataset_identity) != required_dataset_fields:
            raise ValueError("runtime dataset_identity must use the exact closed schema")
        normalized_dataset = dict(dataset_identity)
        for digest_field in ("values_sha256", "labels_sha256"):
            digest = normalized_dataset.get(digest_field)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"runtime dataset_identity has malformed {digest_field}")
        row_count = normalized_dataset.get("row_count")
        feature_count = normalized_dataset.get("feature_count")
        if (
            type(row_count) is not int
            or row_count != 40 * identity[3]
            or type(feature_count) is not int
            or feature_count <= 0
        ):
            raise ValueError("runtime dataset_identity row/feature counts are invalid")
        metadata = normalized_dataset.get("metadata")
        if not isinstance(metadata, Mapping) or set(
            ("bridge_arm", "bridge_lambda", "bridge_nuisance_strength")
        ) - set(metadata):
            raise ValueError("runtime dataset_identity metadata is incomplete")
        if metadata.get("bridge_arm") != identity[2]:
            raise ValueError("runtime dataset_identity bridge arm is misaligned")
        for field in ("bridge_lambda", "bridge_nuisance_strength"):
            value_float = float(metadata.get(field))
            if not np.isfinite(value_float):
                raise ValueError(f"runtime dataset_identity metadata {field} is nonfinite")
        try:
            manifest.canonical_json(normalized_dataset)
        except (TypeError, ValueError) as exc:
            raise ValueError("runtime dataset_identity is not canonical JSON-safe") from exc
        block = identity[1:]
        prior_dataset = dataset_by_block.setdefault(block, normalized_dataset)
        if prior_dataset != normalized_dataset:
            raise ValueError(
                "runtime primitive methods within a block have mixed dataset identities"
            )
        panel = identity[1:4]
        prior_panel_dataset = dataset_by_panel.setdefault(panel, normalized_dataset)
        if prior_panel_dataset != normalized_dataset:
            raise ValueError(
                "runtime dataset_identity changed across timing repeats for one panel"
            )
        candidate_score = row.get("candidate_score")
        if isinstance(candidate_score, (bool, np.bool_)) or not np.isfinite(
            float(candidate_score)
        ):
            raise ValueError("runtime candidate_score must be finite")
        refinement = row.get("prototype_refinement")
        conditioning = row.get("conditioning_diagnostics")
        if not isinstance(refinement, Mapping) or not isinstance(
            conditioning, (Mapping, list)
        ):
            raise ValueError(
                "runtime structural conditioning/refinement diagnostics are malformed"
            )
        structural = {
            "candidate_score": float(candidate_score),
            "prototype_refinement": refinement,
            "conditioning_diagnostics": conditioning,
        }
        try:
            manifest.canonical_json(structural)
        except (TypeError, ValueError) as exc:
            raise ValueError("runtime structural diagnostics are not JSON-safe") from exc
        timing_panel = identity[:4]
        prior_structure = structure_by_timing_panel.setdefault(timing_panel, structural)
        if prior_structure != structural:
            raise ValueError(
                "runtime candidate score or structural diagnostics changed across repeats"
            )
        lookup[identity] = row
    if set(lookup) != expected:
        raise ValueError(
            "runtime artifact lacks the exact six-primitive three-arm five-budget grid"
        )
    return lookup


_RUNTIME_SUM_FIELDS = (
    "fit_wall_seconds",
    "fit_cpu_seconds",
    "conditioning_fit_wall_seconds",
    "conditioning_fit_cpu_seconds",
    "oi_fit_wall_seconds",
    "oi_fit_cpu_seconds",
    "score_fixed_wall_seconds",
    "score_fixed_cpu_seconds",
    "total_wall_seconds",
    "total_cpu_seconds",
)


def _primitive_component(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        **{
            field: float(
                row.get(field, 0.0)
                if field in {"oi_fit_wall_seconds", "oi_fit_cpu_seconds"}
                else row[field]
            )
            for field in _RUNTIME_SUM_FIELDS
        },
        "peak_memory_mb": float(row["peak_memory_mb"]),
    }


def _compose_runtime_row(
    *,
    candidate_id: str,
    backbone: str,
    arm: str,
    budget: int,
    repeat: int,
    components: Mapping[str, Mapping[str, Any]],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if candidate_id not in {"F", "G"} or not components:
        raise ValueError("only derived F/G runtime rows may be composed")
    normalized = {
        str(component): _primitive_component(value)
        for component, value in components.items()
    }
    return {
        "candidate_id": candidate_id,
        "backbone": str(backbone),
        "arm": str(arm),
        "budget": int(budget),
        "repeat": int(repeat),
        **{
            field: float(sum(value[field] for value in normalized.values()))
            for field in _RUNTIME_SUM_FIELDS
        },
        "peak_memory_mb": float(
            max(value["peak_memory_mb"] for value in normalized.values())
        ),
        "primitive_components": normalized,
        "warmup_excluded": True,
        **({} if extra is None else dict(extra)),
    }


def _derive_f_runtime_rows(
    lookup: Mapping[tuple[str, str, str, int, int], Mapping[str, Any]],
    *,
    selected_candidate: str,
) -> list[dict[str, Any]]:
    if selected_candidate not in statistics.SW_CANDIDATES:
        raise ValueError("F requires an exactly settled SW candidate")
    raw = "A" if selected_candidate == "M0-SW" else "B"
    rows = []
    for backbone in statistics.FOOD_BACKBONES:
        for arm in statistics.FOOD_ARMS:
            for budget in statistics.FOOD_RUNTIME_BUDGETS:
                for repeat in statistics.FOOD_REPLICATES:
                    rows.append(
                        _compose_runtime_row(
                            candidate_id="F",
                            backbone=backbone,
                            arm=arm,
                            budget=budget,
                            repeat=repeat,
                            components={
                                raw: lookup[(raw, backbone, arm, budget, repeat)],
                                selected_candidate: lookup[
                                    (selected_candidate, backbone, arm, budget, repeat)
                                ],
                            },
                            extra={
                                "raw_candidate_id": raw,
                                "m50_candidate_id": selected_candidate,
                            },
                        )
                    )
    return rows


def _derive_g_runtime_rows(
    lookup: Mapping[tuple[str, str, str, int, int], Mapping[str, Any]],
    *,
    base_candidate_id: str,
    base_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if base_candidate_id not in {"M0-SW", "M1-SW", "F"}:
        raise ValueError("G requires an exact settled M0-SW, M1-SW, or F base")
    base_lookup = {
        (
            str(row.get("backbone")),
            str(row.get("arm")),
            int(row.get("budget")),
            int(row.get("repeat")),
        ): dict(row)
        for row in base_rows
    }
    expected_base = {
        (backbone, arm, budget, repeat)
        for backbone in statistics.FOOD_BACKBONES
        for arm in statistics.FOOD_ARMS
        for budget in statistics.FOOD_RUNTIME_BUDGETS
        for repeat in statistics.FOOD_REPLICATES
    }
    if set(base_lookup) != expected_base or len(base_rows) != len(expected_base):
        raise ValueError("G base runtime rows must form one exact complete grid")
    result = []
    for arm in statistics.FOOD_ARMS:
        for budget in statistics.FOOD_RUNTIME_BUDGETS:
            for repeat in statistics.FOOD_REPLICATES:
                diagnostics = []
                for backbone in statistics.FOOD_BACKBONES:
                    b = lookup[("B", backbone, arm, budget, repeat)]
                    m1 = lookup[("M1-SW", backbone, arm, budget, repeat)]
                    refinement = b.get("prototype_refinement")
                    if not isinstance(refinement, Mapping):
                        raise ValueError("runtime B row lacks refinement diagnostics")
                    diagnostics.extend(
                        [
                            {
                                "candidate_id": "B",
                                "backbone": backbone,
                                "score": b.get("candidate_score"),
                                "prototype_refinement": {
                                    "applied_rate": refinement.get("applied_rate")
                                },
                            },
                            {
                                "candidate_id": "M1-SW",
                                "backbone": backbone,
                                "score": m1.get("candidate_score"),
                            },
                        ]
                    )
                trigger = fusion.guardrail_trigger(
                    diagnostics, backbones=statistics.FOOD_BACKBONES
                )
                for backbone in statistics.FOOD_BACKBONES:
                    base = base_lookup[(backbone, arm, budget, repeat)]
                    base_components = base.get("primitive_components")
                    if base_candidate_id == "F":
                        if not isinstance(base_components, Mapping):
                            raise ValueError("derived F base lacks primitive components")
                        component_rows = {
                            str(name): dict(value)
                            for name, value in base_components.items()
                        }
                    else:
                        component_rows = {
                            base_candidate_id: lookup[
                                (base_candidate_id, backbone, arm, budget, repeat)
                            ]
                        }
                    for diagnostic_id in ("B", "M1-SW"):
                        diagnostic_row = lookup[
                            (diagnostic_id, backbone, arm, budget, repeat)
                        ]
                        if diagnostic_id in component_rows and _primitive_component(
                            component_rows[diagnostic_id]
                        ) != _primitive_component(diagnostic_row):
                            raise ValueError(
                                f"conflicting G timing for primitive {diagnostic_id!r}"
                            )
                        component_rows[diagnostic_id] = diagnostic_row
                    if trigger["triggered"]:
                        component_rows["LP-CAPPED-2048"] = lookup[
                            ("LP-CAPPED-2048", backbone, arm, budget, repeat)
                        ]
                    result.append(
                        _compose_runtime_row(
                            candidate_id="G",
                            backbone=backbone,
                            arm=arm,
                            budget=budget,
                            repeat=repeat,
                            components=component_rows,
                            extra={
                                "base_candidate_id": base_candidate_id,
                                "triggered": bool(trigger["triggered"]),
                                "trigger": trigger,
                            },
                        )
                    )
    return result


def _selected_chain_recipe_binding(
    *,
    selected_candidate: str,
    selected_chain: str,
    guardrail_base_id: str | None,
    food_bindings: Mapping[str, Any],
    runtime_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every executable chain component to resolved recipe families."""

    if selected_candidate not in statistics.SW_CANDIDATES:
        raise ValueError("recipe binding requires the exact selected M0/M1 base")
    if selected_chain not in {*statistics.SW_CANDIDATES, "F", "G"}:
        raise ValueError("recipe binding requires an executable M/F/G chain")
    raw_candidate = "A" if selected_candidate == "M0-SW" else "B"
    primitives = {selected_candidate}
    derived_identities: dict[str, dict[str, Any]] = {}
    if selected_chain == "F" or guardrail_base_id == "F":
        primitives.add(raw_candidate)
        derived_identities["F"] = {
            "candidate_id": "F",
            "recipe": "equal_fractional_midrank_fusion",
            "component_ids": [raw_candidate, selected_candidate],
            "component_weights": [0.5, 0.5],
            "rank_transform": "complete_panel_fractional_midranks_[0,1]_exact_ties_averaged",
            "selection": "frozen_backbone_order_argmax_on_exact_fused_ties",
        }
    if selected_chain == "G":
        if guardrail_base_id not in {selected_candidate, "F"}:
            raise ValueError("G recipe binding has a mismatched base chain")
        primitives.update({"B", "M1-SW", "LP-CAPPED-2048"})
        derived_identities["G"] = {
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

    food_families = food_bindings.get("recipe_family_sha256")
    runtime_families = runtime_bindings.get("recipe_family_sha256")
    if not isinstance(food_families, Mapping) or not isinstance(
        runtime_families, Mapping
    ):
        raise ValueError("validated recipe families are missing")
    component_hashes: dict[str, dict[str, str]] = {}
    for component in sorted(primitives):
        food_hash = food_families.get(component)
        runtime_hash = runtime_families.get(component)
        if not isinstance(food_hash, str) or not isinstance(runtime_hash, str):
            raise ValueError(f"selected chain recipe is missing component {component!r}")
        component_hashes[component] = {
            "food_recipe_family_sha256": food_hash,
            "runtime_recipe_family_sha256": runtime_hash,
        }
    derived_hashes = {
        candidate_id: hashlib.sha256(
            manifest.canonical_json(identity).encode("utf-8")
        ).hexdigest()
        for candidate_id, identity in sorted(derived_identities.items())
    }
    return {
        "schema_version": 1,
        "selected_candidate": selected_candidate,
        "selected_chain": selected_chain,
        "guardrail_base_id": guardrail_base_id,
        "primitive_component_ids": sorted(primitives),
        "component_recipe_family_sha256": component_hashes,
        "derived_recipe_identity": derived_identities,
        "derived_recipe_sha256": derived_hashes,
        "deterministic_seed_rule": {
            "food": dict(food_bindings["deterministic_seed_rule"]),
            "runtime": dict(runtime_bindings["deterministic_seed_rule"]),
        },
    }


def finalize_development(
    provisional_root: Path | str,
    runtime_root: Path | str,
    output: Path | str,
    *,
    n_resamples: int = statistics.BOOTSTRAP_RESAMPLES,
    verify_current_identity: bool = True,
) -> Path:
    provisional, provisional_manifest, provisional_hash = _load_provisional(
        provisional_root
    )
    development_root = provisional.get("development_input_root")
    if not isinstance(development_root, str) or not development_root:
        raise ValueError("provisional decision lacks the exact development input root")
    development = verify_completed_artifact(
        development_root,
        table_names=(
            "selector_rows",
            "reference_rows",
            "probe_rows",
            "capped_probe_rows",
            "baseline_parity_rows",
        ),
        expected_study="food101_m50_backbone_ranking",
        verify_current_identity=verify_current_identity,
    )
    _validate_food_completed_artifact(development)
    food_recipe_bindings = _validate_recipe_bindings(
        development["raw"].get("recipe_bindings"),
        [
            *development["tables"]["selector_rows"],
            *development["tables"]["capped_probe_rows"],
        ],
        expected_candidates={*manifest.SELECTOR_METHODS, "LP-CAPPED-2048"},
        expected_seed_rule=FOOD_RECIPE_SEED_RULE,
    )
    if development["manifest_sha256"] != provisional.get("input_manifest_sha256"):
        raise ValueError("development input manifest changed after provisional analysis")
    runtime = verify_completed_artifact(
        runtime_root,
        table_names=("runtime_rows",),
        expected_study="m50_backbone_ranking_runtime",
        verify_current_identity=verify_current_identity,
    )
    identity = runtime["manifest"].get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("runtime artifact lacks its exact provisional identity")
    if identity.get("deterministic_seed_rule") != RUNTIME_RECIPE_SEED_RULE:
        raise ValueError("runtime artifact has a mismatched deterministic seed rule")
    if identity.get("provisional_decision_sha256") != provisional_hash:
        raise ValueError("runtime artifact does not consume the exact provisional decision")
    if identity.get("provisional_analysis_manifest_sha256") != provisional_manifest.get(
        "analysis_manifest_sha256"
    ):
        raise ValueError("runtime artifact does not bind the provisional analysis manifest")
    methods = identity.get("methods")
    if not isinstance(methods, Sequence) or isinstance(methods, (str, bytes)) or tuple(
        str(value) for value in methods
    ) != statistics.RUNTIME_PRIMITIVE_IDS:
        raise ValueError("runtime artifact must measure the exact frozen primitive set")
    runtime_arms = identity.get("arms")
    if (
        not isinstance(runtime_arms, Sequence)
        or isinstance(runtime_arms, (str, bytes))
        or tuple(str(value) for value in runtime_arms) != statistics.FOOD_ARMS
    ):
        raise ValueError("runtime artifact must bind the exact frozen three-arm order")
    for field in ("protocol_sha256", "code_identity_sha256"):
        if provisional.get(field) != runtime[field]:
            raise ValueError(f"runtime/provisional {field} mismatch")
    runtime_rows = runtime["tables"]["runtime_rows"]
    runtime_recipe_bindings = _validate_recipe_bindings(
        runtime["raw"].get("recipe_bindings"),
        runtime_rows,
        expected_candidates=set(statistics.RUNTIME_PRIMITIVE_IDS),
        expected_seed_rule=RUNTIME_RECIPE_SEED_RULE,
    )
    runtime_lookup = _validate_runtime_primitives(runtime_rows)
    speed_ratio_by_arm = None
    effective_provisional = dict(provisional)
    if provisional.get("unrefined_requires_speed_ratio"):
        speed_ratio_by_arm = _paired_runtime_ratio(runtime_rows, "M0-SW", "M1-SW")
        if any(
            value > statistics.REFINEMENT_SPEED_RATIO_MAX
            for value in speed_ratio_by_arm.values()
        ):
            effective_provisional["selected_candidate"] = "M1-SW"
    selected = str(effective_provisional.get("selected_candidate"))
    if selected not in statistics.SW_CANDIDATES:
        raise ValueError("runtime did not settle an exact M0-SW/M1-SW candidate")

    selectors = development["tables"]["selector_rows"]
    references = development["tables"]["reference_rows"]
    capped = development["tables"]["capped_probe_rows"]
    base_metrics = statistics.food_panel_metrics(
        selectors, references, candidate_ids=manifest.SELECTOR_METHODS
    )
    all_metrics = list(base_metrics)
    all_rank_auc = statistics.rank_auc_rows(base_metrics)
    fusion_rows: list[dict[str, Any]] = []
    guardrail_rows: list[dict[str, Any]] = []
    fusion_gates: dict[str, Any] | None = None
    guardrail_gates: dict[str, Any] | None = None
    selected_gate = effective_provisional.get("candidate_gates", {}).get(selected)
    if not isinstance(selected_gate, Mapping):
        raise ValueError("settled candidate lacks its frozen development gate")
    gate_status = str(selected_gate.get("status"))
    if gate_status not in {"pass", "fusion_required", "guardrail_eligible"}:
        raise ValueError("settled candidate is not eligible for any frozen chain")
    chain = selected

    probe_metrics = [
        row for row in base_metrics if row["candidate_id"] == "LP-FULL"
    ]
    if gate_status == "fusion_required":
        raw = "A" if selected == "M0-SW" else "B"
        fusion_rows = fusion.derive_fusion_rows(
            selectors,
            raw_candidate_id=raw,
            m50_candidate_id=selected,
            backbones=statistics.FOOD_BACKBONES,
        )
        derived_metrics = statistics.food_panel_metrics(
            fusion_rows, references, candidate_ids=("F",)
        )
        fusion_gates = statistics.candidate_product_gates(
            [*derived_metrics, *probe_metrics],
            candidate_id="F",
            n_resamples=n_resamples,
        )
        _validate_derived_gate_summary(fusion_gates, "F")
        all_metrics.extend(derived_metrics)
        all_rank_auc.extend(statistics.rank_auc_rows(derived_metrics))
        if fusion_gates["status"] == "pass":
            chain = "F"
        elif fusion_gates["status"] == "guardrail_eligible":
            chain = "G"
        else:
            chain = "stopped_fusion_failure"
    elif gate_status == "guardrail_eligible":
        chain = "G"

    if chain == "G":
        policy_base_id = "F" if fusion_rows else selected
        policy_base_rows = fusion_rows or [
            row for row in selectors if row.get("candidate_id") == selected
        ]
        guardrail_rows = _derive_guardrail_all_panels(
            selectors,
            capped,
            base_rows=policy_base_rows,
            base_candidate_id=policy_base_id,
        )
        derived_metrics = statistics.food_panel_metrics(
            guardrail_rows, references, candidate_ids=("G",)
        )
        guardrail_gates = statistics.candidate_product_gates(
            [*derived_metrics, *probe_metrics],
            candidate_id="G",
            n_resamples=n_resamples,
        )
        _validate_derived_gate_summary(guardrail_gates, "G")
        all_metrics.extend(derived_metrics)
        all_rank_auc.extend(statistics.rank_auc_rows(derived_metrics))
        if guardrail_gates["status"] != "pass":
            chain = "stopped_guardrail_failure"

    if chain == "stopped_fusion_failure":
        resources = {
            "candidate_id": None,
            "status": "not_run_algorithmic_failure",
            "arm_gates": {},
            "full_panel_arm_gates": {},
            "pooled_summary": {},
            "ratio_rows": [],
            "full_panel_ratio_rows": [],
            "scaling_by_budget": [],
            "secondary_runtime": {
                "inferential": False,
                "promotion_gate": False,
                "arm_summary": {},
                "scaling_by_arm_budget": [],
            },
        }
    elif chain == "stopped_guardrail_failure":
        resources = {
            "candidate_id": None,
            "status": "not_run_policy_failure",
            "arm_gates": {},
            "full_panel_arm_gates": {},
            "pooled_summary": {},
            "ratio_rows": [],
            "full_panel_ratio_rows": [],
            "scaling_by_budget": [],
            "secondary_runtime": {
                "inferential": False,
                "promotion_gate": False,
                "arm_summary": {},
                "scaling_by_arm_budget": [],
            },
        }
    else:
        if chain == "F":
            candidate_runtime = _derive_f_runtime_rows(
                runtime_lookup, selected_candidate=selected
            )
        elif chain == "G":
            if fusion_rows:
                base_runtime = _derive_f_runtime_rows(
                    runtime_lookup, selected_candidate=selected
                )
                base_runtime_id = "F"
            else:
                base_runtime = [
                    row
                    for row in runtime_rows
                    if str(row.get("candidate_id")) == selected
                ]
                base_runtime_id = selected
            candidate_runtime = _derive_g_runtime_rows(
                runtime_lookup,
                base_candidate_id=base_runtime_id,
                base_rows=base_runtime,
            )
        else:
            candidate_runtime = []
        resources = statistics.resource_gates(
            [*runtime_rows, *candidate_runtime],
            candidate_id=str(chain),
            n_resamples=n_resamples,
        )
    finalized = statistics.finalize_development_lock(
        effective_provisional,
        resource_summary=resources,
        unrefined_vs_refined_runtime_ratio_by_arm=speed_ratio_by_arm,
        fusion_gate_summary=fusion_gates,
        guardrail_gate_summary=guardrail_gates,
    )
    selected_recipe_binding = None
    if finalized.get("status") == "locked_for_future_confirmation":
        selected_chain = str(finalized.get("selected_chain"))
        guardrail_base = (
            ("F" if fusion_rows else selected)
            if selected_chain == "G"
            else None
        )
        selected_recipe_binding = _selected_chain_recipe_binding(
            selected_candidate=str(finalized.get("selected_candidate")),
            selected_chain=selected_chain,
            guardrail_base_id=guardrail_base,
            food_bindings=food_recipe_bindings,
            runtime_bindings=runtime_recipe_bindings,
        )
        finalized["selected_chain_recipe_bindings"] = selected_recipe_binding
        finalized["selected_chain_recipe_bindings_sha256"] = _object_sha256(
            selected_recipe_binding
        )
    finalized.update(
        {
            "stage": "development",
            "protocol_sha256": runtime["protocol_sha256"],
            "code_identity_sha256": runtime["code_identity_sha256"],
            "provisional_decision_sha256": provisional_hash,
            "runtime_manifest_sha256": runtime["manifest_sha256"],
            "provisional_analysis_manifest_sha256": provisional_manifest[
                "analysis_manifest_sha256"
            ],
            "derived_candidates_evaluated": [
                value
                for value, rows in (("F", fusion_rows), ("G", guardrail_rows))
                if rows
            ],
            "runner_up_allowed": False,
            "resource_candidate_id": resources.get("candidate_id"),
            "resource_summary_sha256": _object_sha256(resources),
        }
    )
    prior_summary = _read_object(Path(provisional_root) / "summary.json")
    prior_summary["stage"] = "development_final"
    prior_summary["resources"] = resources
    prior_summary["fusion_gates"] = fusion_gates
    prior_summary["guardrail_gates"] = guardrail_gates
    prior_summary["selection_rates"] = statistics.selection_rate_summary(all_metrics)
    prior_summary["rank_auc_summary"] = statistics.rank_auc_summary(all_rank_auc)
    prior_summary["decision_status"] = finalized["status"]
    return reporting.write_report_bundle(
        output,
        summary=prior_summary,
        development_lock=finalized,
        panel_metrics=all_metrics,
        rank_auc=all_rank_auc,
        factorial_rows=_flatten_factorial(prior_summary["factorial"]),
        resource_rows=resources["ratio_rows"],
        full_panel_resource_rows=resources["full_panel_ratio_rows"],
        selected_chain_recipe_bindings=selected_recipe_binding,
        linked_development_files={
            name: Path(provisional_root) / name
            for name in (
                "summary.json",
                "panel_metrics.csv",
                "rank_auc.csv",
                "factorial.csv",
                "report.md",
                "selector_regret.svg",
            )
        },
        lineage={
            "stage": "development_final",
            "status": str(finalized["status"]),
            "protocol_sha256": runtime["protocol_sha256"],
            "code_identity_sha256": runtime["code_identity_sha256"],
            "input_manifest_sha256": runtime["manifest_sha256"],
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    development = subparsers.add_parser("development")
    development.add_argument("--input", type=Path, required=True)
    development.add_argument("--output", type=Path, required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--provisional", type=Path, required=True)
    finalize.add_argument("--runtime", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "development":
        analyze_development(args.input, args.output)
    else:
        finalize_development(args.provisional, args.runtime, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "analyze_development",
    "canonical_table_sha256",
    "finalize_development",
    "main",
    "verify_completed_artifact",
]
