"""Fail-closed admission and outcome validator for a future continuous benchmark.

This module intentionally performs no model fitting, data loading, or outcome
comparison. It validates a frozen mechanism record, a benchmark plan, and (in a
future run) complete uncertainty-aware summary fields.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{field} must be finite")
    return result


def _unique_strings(values: Any, field: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{field} must be a sequence of strings")
    result = [str(value) for value in values]
    if not result or any(not value for value in result) or len(set(result)) != len(result):
        raise ValueError(f"{field} must contain unique non-empty strings")
    return result


def _integer_set(values: Any, field: str) -> set[int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{field} must be a sequence of integers")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError(f"{field} must contain measured integer values")
    result = set(values)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{field} must contain positive integers")
    return result


def load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    actual = sha256_path(PROTOCOL_PATH)
    if len(expected) != 64 or actual != expected:
        raise ValueError("practical-value protocol SHA-256 sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("status") != "frozen_prospective_practical_value_gate_before_outcomes":
        raise ValueError("practical-value gate is not frozen before outcomes")
    scope = protocol.get("scope", {})
    for field in ("overlapindex_evaluated", "outcome_comparison_executed", "public_library_modified", "prior_artifacts_modified"):
        if scope.get(field) is not False:
            raise ValueError(f"scope.{field} must remain false")
    estimand = protocol.get("estimand", {})
    if estimand.get("comparator") != "linear_cv_training_only":
        raise ValueError("the deployable comparator must be training-only linear CV")
    if estimand.get("primary_metric") != "top1_selection_regret_r2":
        raise ValueError("top-one selection regret must be primary")
    uncertainty = protocol.get("uncertainty", {})
    if float(uncertainty.get("confidence_level", 0.0)) != 0.95:
        raise ValueError("the gate requires 95% uncertainty bounds")
    if float(uncertainty.get("material_pair_difference_r2", 0.0)) != 0.01:
        raise ValueError("material pair difference must remain 0.01 R2")
    entry = protocol.get("step2_entry", {})
    if entry.get("required_status") != "passed":
        raise ValueError("Step 2 must pass before entry")
    if int(entry.get("maximum_formulation_changes_after_mechanism", -1)) != 1:
        raise ValueError("the one-formulation-change limit is missing")
    data = protocol.get("data_design", {})
    if int(data.get("minimum_new_dataset_groups", 0)) < 2:
        raise ValueError("at least two fresh dataset groups are required")
    if int(data.get("minimum_reserved_backbones", 0)) < 2:
        raise ValueError("at least two reserved backbones are required")
    grid = protocol.get("capacity_grid", {})
    if set(grid.get("embedding_dimensions", [])) != {384, 768, 1536}:
        raise ValueError("the intended embedding-dimension grid changed")
    if set(grid.get("representative_synthetic_dimensions", [])) != {128, 512, 1024}:
        raise ValueError("the representative synthetic dimension grid changed")
    if set(grid.get("sample_sizes", [])) != {300, 600, 1200}:
        raise ValueError("the intended sample-size grid changed")
    runtime = protocol.get("runtime", {})
    if int(runtime.get("timed_repeats", 0)) != 5 or int(runtime.get("warmup_runs_excluded", -1)) != 1:
        raise ValueError("runtime repeat/warmup policy changed")
    if float(runtime.get("time_competitive_geomean_ratio_maximum", 0.0)) != 1.0:
        raise ValueError("time-competitive geomean ceiling changed")
    if float(runtime.get("preferred_geomean_ratio_maximum", 0.0)) != 0.5:
        raise ValueError("preferred runtime target changed")


def validate_step2_record(record: Mapping[str, Any], method_id: str, protocol: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("status") != protocol["step2_entry"]["required_status"]:
        raise ValueError("candidate did not survive the Step 2 mechanism diagnostic")
    if str(record.get("method_id")) != str(method_id):
        raise ValueError("Step 2 method identity does not match the candidate")
    if record.get("method_config_frozen_before_outcomes") is not True:
        raise ValueError("Step 2 method configuration was not frozen before outcomes")
    if record.get("outcomes_used_for_tuning") is not False:
        raise ValueError("Step 2 outcomes cannot tune the candidate")
    config_hash = record.get("locked_method_config_sha256")
    if not isinstance(config_hash, str) or len(config_hash) != 64:
        raise ValueError("Step 2 must provide a locked method-config SHA-256")
    changes = int(record.get("formulation_changes_after_mechanism", -1))
    if changes < 0 or changes > int(protocol["step2_entry"]["maximum_formulation_changes_after_mechanism"]):
        raise ValueError("the post-mechanism formulation-change limit was exceeded")
    if changes == 1:
        mapping = record.get("diagnosis_to_change_mapping")
        if not isinstance(mapping, Mapping):
            raise ValueError("the single allowed change needs a diagnosis-to-change mapping")
        if not str(mapping.get("diagnosis_id", "")).strip():
            raise ValueError("diagnosis-to-change mapping needs a diagnosis_id")
        if mapping.get("change_id") != protocol["step2_entry"]["diagnosis_to_change_mapping"]["allowed_change_id"]:
            raise ValueError("formulation change is not the predeclared diagnosis-to-change mapping")
        if mapping.get("declared_before_step2_outcomes") is not True:
            raise ValueError("diagnosis-to-change mapping was declared too late")
        if record.get("post_change_outcomes_used_for_tuning") is not False:
            raise ValueError("Step 2 outcomes cannot tune the post-diagnostic change")
        changed_hash = record.get("post_change_locked_method_config_sha256")
        if not isinstance(changed_hash, str) or len(changed_hash) != 64 or changed_hash == config_hash:
            raise ValueError("the formulation change needs a distinct locked method-config SHA-256")
    return {
        "method_id": str(method_id),
        "formulation_changes_after_mechanism": changes,
        "locked_method_config_sha256": config_hash,
    }


def validate_reference_headroom(reference: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    admission = protocol["reference_admission"]
    if reference.get("selector") != "linear_cv_training_only":
        raise ValueError("reference headroom must be measured by training-only Linear CV")
    fields = (
        "best_reference_r2_point",
        "best_reference_r2_lower95",
        "nonlinear_gain_lower95_r2",
        "training_only_linear_cv_top1_regret_lower95_r2",
        "training_only_linear_cv_material_pair_reversal_count",
    )
    for field in fields:
        _number(reference.get(field), f"reference.{field}")
    if reference.get("top3_boundary_identified") is not True:
        raise ValueError("the top-three boundary is not identifiable")
    checks = {
        "recoverable": _number(reference["best_reference_r2_point"], "best_reference_r2_point") >= float(admission["minimum_best_reference_r2_point"]),
        "positive_lower_bound": _number(reference["best_reference_r2_lower95"], "best_reference_r2_lower95") > float(admission["minimum_best_reference_r2_lower95"]),
        "nonlinear_gain": _number(reference["nonlinear_gain_lower95_r2"], "nonlinear_gain_lower95_r2") > float(admission["minimum_nonlinear_gain_lower95_r2"]),
        "lp_headroom": _number(reference["training_only_linear_cv_top1_regret_lower95_r2"], "training_only_linear_cv_top1_regret_lower95_r2") > float(admission["minimum_training_only_linear_cv_top1_regret_lower95_r2"]),
        "material_rank_change": _number(reference["training_only_linear_cv_top1_regret_lower95_r2"], "training_only_linear_cv_top1_regret_lower95_r2") > float(admission["minimum_training_only_linear_cv_top1_regret_lower95_r2"]) or int(reference["training_only_linear_cv_material_pair_reversal_count"]) >= int(admission["minimum_training_only_linear_cv_material_pair_reversals"]),
    }
    checks["nonlinear_opportunity"] = bool(all(checks.values()))
    return checks


def validate_benchmark_plan(plan: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    data = protocol["data_design"]
    groups = plan.get("dataset_groups")
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        raise ValueError("dataset_groups must be a sequence")
    group_ids = []
    roles = []
    source_ids = []
    for group in groups:
        if not isinstance(group, Mapping) or group.get("is_new") is not True:
            raise ValueError("every admitted dataset group must be newly reserved")
        group_ids.append(str(group.get("group_id")))
        roles.append(str(group.get("role")))
        source_id = str(group.get("source_id"))
        if not source_id or source_id == "None":
            raise ValueError("every dataset group needs a source_id")
        source_ids.append(source_id)
    if len(group_ids) < int(data["minimum_new_dataset_groups"]) or len(set(group_ids)) != len(group_ids):
        raise ValueError("the benchmark lacks the required number of distinct new dataset groups")
    nonlinear_sources = {source for source, role in zip(source_ids, roles) if role == "nonlinear_opportunity"}
    linear_sources = {source for source, role in zip(source_ids, roles) if role == "linear_control"}
    if len(nonlinear_sources) < int(data["minimum_new_qualified_nonlinear_source_datasets"]):
        raise ValueError("the benchmark lacks new qualified nonlinear source groups")
    if len(linear_sources) < int(data["minimum_new_linear_control_source_datasets"]):
        raise ValueError("the benchmark lacks new linear-control source groups")
    reserved = _unique_strings(plan.get("reserved_backbones"), "reserved_backbones")
    if len(reserved) < int(data["minimum_reserved_backbones"]):
        raise ValueError("the benchmark lacks reserved backbones")
    if set(reserved).intersection(set(_unique_strings(plan.get("scout_backbones"), "scout_backbones"))):
        raise ValueError("reserved backbones overlap scout backbones")
    dimensions = _integer_set(plan.get("embedding_dimensions", []), "embedding_dimensions")
    samples = _integer_set(plan.get("sample_sizes", []), "sample_sizes")
    if dimensions != set(protocol["capacity_grid"]["embedding_dimensions"]):
        raise ValueError("the intended embedding dimensions are incomplete")
    native = plan.get("native_dimensions")
    if not isinstance(native, Mapping) or not native:
        raise ValueError("native embedding dimensions must be measured per backbone")
    expected_backbones = set(_unique_strings(plan.get("scout_backbones"), "scout_backbones")) | set(reserved)
    if set(str(key) for key in native) != expected_backbones:
        raise ValueError("native dimensions must be recorded for every scout and reserved backbone")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in native.values()):
        raise ValueError("native embedding dimensions must be positive measured integers")
    if samples != set(protocol["capacity_grid"]["sample_sizes"]):
        raise ValueError("the intended sample sizes are incomplete")
    if plan.get("independent_evaluation") is not True or plan.get("method_parameters_locked") is not True:
        raise ValueError("the broad benchmark must be independent and method-locked")
    if plan.get("outcome_free") is not True:
        raise ValueError("the admission plan must be outcome-free")
    runtime = plan.get("runtime", {})
    required_runtime = ("includes", "excludes", "warmup_runs_excluded", "timed_repeats", "thread_control", "complete_grid")
    if any(field not in runtime for field in required_runtime):
        raise ValueError("runtime accounting is incomplete")
    if runtime["warmup_runs_excluded"] != 1 or runtime["timed_repeats"] != 5 or runtime["complete_grid"] is not True:
        raise ValueError("runtime controls do not match the frozen gate")
    if not plan.get("scorer_runtime_includes_all"): 
        raise ValueError("runtime must include the complete deployable selector")
    return {
        "new_dataset_group_count": len(group_ids),
        "reserved_backbone_count": len(reserved),
        "embedding_dimensions": sorted(dimensions),
        "sample_sizes": sorted(samples),
    }


def admission_decision(
    method_id: str,
    step2_record: Mapping[str, Any],
    reference: Mapping[str, Any],
    plan: Mapping[str, Any],
    protocol: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    protocol = load_protocol() if protocol is None else protocol
    checks: dict[str, Any] = {}
    try:
        checks["step2"] = validate_step2_record(step2_record, method_id, protocol)
        checks["reference"] = validate_reference_headroom(reference, protocol)
        checks["plan"] = validate_benchmark_plan(plan, protocol)
        passed = bool(checks["reference"]["nonlinear_opportunity"])
        if not passed:
            checks["failure"] = "reference does not establish a stable, recoverable nonlinear ranking opportunity with LP headroom"
    except (KeyError, TypeError, ValueError) as exc:
        return {"admitted": False, "checks": checks, "failure": str(exc), "next_action": "stop"}
    return {
        "admitted": passed,
        "checks": checks,
        "failure": None if passed else checks["failure"],
        "next_action": "independent_evaluation" if passed else "stop",
    }


def outcome_decision(outcomes: Mapping[str, Any], protocol: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Apply the frozen post-evaluation gate to already-complete summaries."""
    protocol = load_protocol() if protocol is None else protocol
    required = {
        "linear_control": (
            "upper95_excess_top1_regret_r2",
            "upper95_excess_top3_opportunity_regret_r2",
            "upper95_material_pair_accuracy_loss",
        ),
        "nonlinear_opportunity": (
            "upper95_method_minus_linear_top1_regret_r2",
            "upper95_method_minus_linear_top3_opportunity_regret_r2",
        ),
        "broad_panel": (
            "upper95_excess_top1_regret_r2",
            "upper95_excess_top3_opportunity_regret_r2",
        ),
        "runtime": ("geometric_mean_ratio", "p90_ratio", "complete_grid", "peak_memory_recorded"),
    }
    checks: dict[str, bool] = {}
    try:
        for section, fields in required.items():
            record = outcomes[section]
            for field in fields:
                if field not in record:
                    raise ValueError(f"outcome.{section}.{field} is missing")
                if field not in ("complete_grid", "peak_memory_recorded"):
                    _number(record[field], f"outcome.{section}.{field}")
        linear = outcomes["linear_control"]
        linear_gate = protocol["success"]["linear_control"]
        checks["linear_control_top1"] = float(linear["upper95_excess_top1_regret_r2"]) <= float(linear_gate["upper95_excess_top1_regret_r2_maximum"])
        checks["linear_control_top3"] = float(linear["upper95_excess_top3_opportunity_regret_r2"]) <= float(linear_gate["upper95_excess_top3_opportunity_regret_r2_maximum"])
        checks["linear_control_pairs"] = float(linear["upper95_material_pair_accuracy_loss"]) <= float(linear_gate["upper95_material_pair_accuracy_loss_maximum"])
        nonlinear = outcomes["nonlinear_opportunity"]
        nonlinear_gate = protocol["success"]["nonlinear_opportunity"]
        checks["nonlinear_top1"] = float(nonlinear["upper95_method_minus_linear_top1_regret_r2"]) <= float(nonlinear_gate["upper95_method_minus_linear_top1_regret_r2_maximum"])
        checks["nonlinear_top3"] = float(nonlinear["upper95_method_minus_linear_top3_opportunity_regret_r2"]) <= float(nonlinear_gate["upper95_method_minus_linear_top3_opportunity_regret_r2_maximum"])
        broad = outcomes["broad_panel"]
        broad_gate = protocol["success"]["broad_panel"]
        checks["broad_top1"] = float(broad["upper95_excess_top1_regret_r2"]) <= float(broad_gate["upper95_excess_top1_regret_r2_maximum"])
        checks["broad_top3"] = float(broad["upper95_excess_top3_opportunity_regret_r2"]) <= float(broad_gate["upper95_excess_top3_opportunity_regret_r2_maximum"])
        runtime = outcomes["runtime"]
        runtime_gate = protocol["success"]["runtime"]
        checks["runtime_geomean"] = float(runtime["geometric_mean_ratio"]) <= float(runtime_gate["geometric_mean_ratio_maximum"])
        checks["runtime_p90"] = float(runtime["p90_ratio"]) <= float(runtime_gate["p90_ratio_maximum"])
        checks["runtime_complete"] = runtime["complete_grid"] is True and runtime["peak_memory_recorded"] is True
    except (KeyError, TypeError, ValueError) as exc:
        return {"passed": False, "checks": checks, "failure": str(exc), "next_action": "stop"}
    passed = bool(checks) and all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "failure": None if passed else "one or more practical-value gates failed",
        "next_action": "promote_candidate" if passed else "stop",
        "preferred_speed": float(outcomes["runtime"]["geometric_mean_ratio"]) <= float(protocol["runtime"]["preferred_geomean_ratio_maximum"]),
    }


__all__ = [
    "PROTOCOL_PATH",
    "PROTOCOL_SHA256_PATH",
    "admission_decision",
    "load_protocol",
    "outcome_decision",
    "sha256_path",
    "validate_benchmark_plan",
    "validate_protocol",
    "validate_reference_headroom",
    "validate_step2_record",
]
