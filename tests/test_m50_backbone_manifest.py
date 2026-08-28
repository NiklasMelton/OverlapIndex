"""Focused, outcome-free tests for M50 planning/provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.m50_backbone_ranking import manifest as m


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(m.canonical_json(payload) + "\n", encoding="utf-8")


def _selected_recipe_binding(
    *,
    selected_candidate: str = "M1-SW",
    selected_chain: str = "M1-SW",
    guardrail_base_id: str | None = None,
) -> dict[str, object]:
    """Build a small valid final-lock recipe binding for provenance tests."""

    raw_candidate = "A" if selected_candidate == "M0-SW" else "B"
    if selected_chain in {"M0-SW", "M1-SW"}:
        components = [selected_candidate]
        derived: dict[str, object] = {}
    elif selected_chain == "F":
        components = sorted({raw_candidate, selected_candidate})
        derived = {
            "F": {
                "candidate_id": "F",
                "recipe": "equal_fractional_midrank_fusion",
                "component_ids": [raw_candidate, selected_candidate],
                "component_weights": [0.5, 0.5],
                "rank_transform": "complete_panel_fractional_midranks_[0,1]_exact_ties_averaged",
                "selection": "frozen_backbone_order_argmax_on_exact_fused_ties",
            }
        }
    else:
        components = sorted({selected_candidate, "B", "M1-SW", "LP-CAPPED-2048"})
        derived = {}
        if guardrail_base_id == "F":
            components = sorted(set(components) | {raw_candidate})
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
    component_hashes = {
        component: {
            "food_recipe_family_sha256": m.sha256_bytes(
                f"food:{component}".encode("utf-8")
            ),
            "runtime_recipe_family_sha256": m.sha256_bytes(
                f"runtime:{component}".encode("utf-8")
            ),
        }
        for component in components
    }
    derived_hashes = {
        candidate_id: m.sha256_bytes(m.canonical_json(identity).encode("utf-8"))
        for candidate_id, identity in sorted(derived.items())
    }
    return {
        "schema_version": 1,
        "selected_candidate": selected_candidate,
        "selected_chain": selected_chain,
        "guardrail_base_id": guardrail_base_id,
        "primitive_component_ids": components,
        "component_recipe_family_sha256": component_hashes,
        "derived_recipe_identity": derived,
        "derived_recipe_sha256": derived_hashes,
        "deterministic_seed_rule": {
            "food": {
                "selector_panel_seed": "SEED + replicate",
                "oi_fold_seed": "selector_panel_seed + fold",
                "probe_split_seed": "selector_panel_seed",
                "probe_model_random_state": "selector_panel_seed",
                "capped_subset_seed": "selector_panel_seed",
            },
            "runtime": {
                "runtime_call_seed": "SEED + budget",
                "oi_fold_seed": "runtime_call_seed + fold",
                "probe_split_seed": "runtime_call_seed",
                "probe_model_random_state": "runtime_call_seed",
                "capped_subset_seed": "runtime_call_seed",
                "repeat_effect": "order_and_timing_only",
            },
        },
    }


def _write_final_lock_artifact(root: Path, binding: dict[str, object]) -> tuple[str, str]:
    """Write a complete sibling-bound final lock for closed-schema tests."""

    root.mkdir(parents=True, exist_ok=True)
    protocol_hash = "p" * 64
    code_hash = "c" * 64
    selected_candidate = str(binding["selected_candidate"])
    selected_chain = str(binding["selected_chain"])
    chain_kind = (
        "pure_oi"
        if selected_chain in {"M0-SW", "M1-SW"}
        else ("oi_fusion" if selected_chain == "F" else "product_guardrail")
    )
    resources = {"candidate_id": selected_chain, "status": "pass"}
    binding_hash = m.sha256_bytes(m.canonical_json(binding).encode("utf-8"))
    lock = {
        "stage": "development",
        "status": "locked_for_future_confirmation",
        "protocol_sha256": protocol_hash,
        "code_identity_sha256": code_hash,
        "selected_candidate": selected_candidate,
        "selected_chain": selected_chain,
        "chain_kind": chain_kind,
        "algorithmic_oi_status": "pass" if selected_chain != "G" else "fail_linear_claim",
        "product_policy_status": "pass" if selected_chain == "G" else "not_applicable",
        "resource_candidate_id": selected_chain,
        "resource_status": "pass",
        "resource_summary_sha256": m.sha256_bytes(
            m.canonical_json(resources).encode("utf-8")
        ),
        "selected_chain_recipe_bindings": binding,
        "selected_chain_recipe_bindings_sha256": binding_hash,
    }
    _write_json(root / "summary.json", {"resources": resources})
    _write_json(root / "selected_chain_recipe_bindings.json", binding)
    _write_json(root / "development_lock.json", lock)
    file_hashes = {
        name: {"sha256": m.sha256_path(root / name)}
        for name in (
            "development_lock.json",
            "summary.json",
            "selected_chain_recipe_bindings.json",
        )
    }
    _write_json(
        root / "analysis_manifest.json",
        {
            "schema_version": 1,
            "stage": "development_final",
            "protocol_sha256": protocol_hash,
            "code_identity_sha256": code_hash,
            "development_lock_sha256": file_hashes["development_lock.json"]["sha256"],
            "selected_chain_recipe_bindings_sha256": binding_hash,
            "files": file_hashes,
        },
    )
    return protocol_hash, code_hash


def _refresh_final_lock_artifact(
    root: Path,
    *,
    binding: dict[str, object] | None = None,
    binding_hash_override: str | None = None,
) -> None:
    """Rewrite all sibling identities after a deliberate test corruption."""

    lock = json.loads((root / "development_lock.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "analysis_manifest.json").read_text(encoding="utf-8"))
    if binding is not None:
        binding_hash = m.sha256_bytes(m.canonical_json(binding).encode("utf-8"))
        lock["selected_chain_recipe_bindings"] = binding
        lock["selected_chain_recipe_bindings_sha256"] = binding_hash
        manifest["selected_chain_recipe_bindings_sha256"] = binding_hash
        _write_json(root / "selected_chain_recipe_bindings.json", binding)
    elif binding_hash_override is not None:
        lock["selected_chain_recipe_bindings_sha256"] = binding_hash_override
    _write_json(root / "development_lock.json", lock)
    for name in (
        "development_lock.json",
        "summary.json",
        "selected_chain_recipe_bindings.json",
    ):
        manifest["files"][name] = {"sha256": m.sha256_path(root / name)}
    _write_json(root / "analysis_manifest.json", manifest)


def _write_provisional_artifact(
    root: Path,
    *,
    protocol_hash: str = "p" * 64,
    code_hash: str = "c" * 64,
    runtime_ids: tuple[str, ...] = m.RUNTIME_PRIMITIVE_METHODS,
    selected_chain: object = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    decision = {
        "stage": "development_provisional",
        "status": "provisional_runtime_pending",
        "protocol_sha256": protocol_hash,
        "code_identity_sha256": code_hash,
        "selected_candidate": "M1-SW",
        "selected_chain": selected_chain,
        "runtime_candidate_ids": list(runtime_ids),
        "runtime_pending": True,
        "runner_up_allowed": False,
    }
    decision_path = root / "development_lock.json"
    _write_json(decision_path, decision)
    decision_hash = m.sha256_path(decision_path)
    _write_json(
        root / "analysis_manifest.json",
        {
            "schema_version": 1,
            "stage": "development_provisional",
            "protocol_sha256": protocol_hash,
            "code_identity_sha256": code_hash,
            "development_lock_sha256": decision_hash,
            "files": {"development_lock.json": {"sha256": decision_hash}},
        },
    )
    return root


def test_exact_food_and_runtime_grid_counts_and_canonical_ids() -> None:
    panels = m.development_panels()
    assert len(panels) == 600
    assert m.EXPECTED_DEVELOPMENT_ROW_COUNT == 4200
    assert m.SELECTOR_METHODS == (
        "A", "B", "M0-SW", "M1-SW", "M0-CB", "M1-CB", "LP-FULL"
    )
    assert not any(name.startswith("RAW") or name.startswith("M50-") for name in m.SELECTOR_METHODS)
    assert len(m.runtime_panels()) == 750
    assert m.EXPECTED_RUNTIME_PANEL_COUNT == 750
    assert m.EXPECTED_RUNTIME_ROW_COUNT == 4500
    assert len({panel.panel_id for panel in panels}) == 600
    assert m.validate_development_panels(panels)


def test_schedule_is_cyclic_near_counterbalanced_and_order_independent() -> None:
    panels = m.development_panels()[:7]
    first = m.planned_execution_order(panels)
    second = m.planned_execution_order(tuple(reversed(panels)))
    assert first == second
    assert m.validate_execution_order(first, panels)
    for panel in panels:
        rows = [row for row in first if row["panel_id"] == panel.panel_id]
        assert [row["execution_position"] for row in rows] == list(range(7))


def test_full_development_schedule_has_85_86_assignments_per_method_position() -> None:
    schedule = m.planned_execution_order(m.development_panels())
    assert len(schedule) == 600 * 7
    counts: dict[tuple[str, int], int] = {}
    for row in schedule:
        key = (str(row["method_id"]), int(row["execution_position"]))
        counts[key] = counts.get(key, 0) + 1
    assert set(counts) == {
        (method, position)
        for method in m.SELECTOR_METHODS
        for position in range(len(m.SELECTOR_METHODS))
    }
    assert set(counts.values()) <= {85, 86}
    assert set(counts.values()) >= {85, 86}


def test_full_runtime_schedule_is_exactly_counterbalanced() -> None:
    schedule = m.runtime_execution_order(m.RUNTIME_PRIMITIVE_METHODS)
    assert len(schedule) == 750 * 6
    counts: dict[tuple[str, int], int] = {}
    for row in schedule:
        key = (str(row["method_id"]), int(row["execution_position"]))
        counts[key] = counts.get(key, 0) + 1
    assert set(counts) == {
        (method, position)
        for method in m.RUNTIME_PRIMITIVE_METHODS
        for position in range(len(m.RUNTIME_PRIMITIVE_METHODS))
    }
    assert set(counts.values()) == {125}


def test_code_identity_does_not_change_when_output_is_created(tmp_path: Path) -> None:
    for relative in m.DECLARED_SOURCE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    before = m.code_identity_sha256(tmp_path)
    output = tmp_path / "artifacts/m50_backbone_ranking/development"
    output.mkdir(parents=True)
    (output / "raw_results.json").write_text("generated", encoding="utf-8")
    (output / "checkpoint.json").write_text("generated", encoding="utf-8")
    assert m.code_identity_sha256(tmp_path) == before
    with pytest.raises(ValueError, match="generated output root"):
        m.source_hashes(tmp_path, ("artifacts/m50_backbone_ranking/raw.json",))


def test_code_identity_changes_when_local_overlapindex_source_changes(
    tmp_path: Path,
) -> None:
    for relative in m.DECLARED_SOURCE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    before = m.code_identity_sha256(tmp_path, require_existing=True)
    source = tmp_path / "overlapindex/OverlapIndex.py"
    source.write_text("changed\n", encoding="utf-8")
    after = m.code_identity_sha256(tmp_path, require_existing=True)
    assert after != before


def test_environment_preserves_per_package_versions_and_local_project_version() -> None:
    observed = m.environment(
        {
            "experiment_commit": "e" * 40,
            "git_dirty": True,
        }
    )
    assert set(observed["packages"]) == {
        "numpy", "scipy", "scikit-learn", "overlapindex"
    }
    assert observed["packages"]["numpy"] is not None
    assert observed["packages"]["scipy"] is not None
    assert observed["packages"]["scikit-learn"] is not None
    assert observed["project"] == {"name": "overlapindex", "version": "0.1.4"}


def test_provisional_runtime_decision_requires_bound_artifact_and_exact_chain(
    tmp_path: Path,
) -> None:
    protocol_hash = "p" * 64
    code_hash = "c" * 64
    artifact = _write_provisional_artifact(
        tmp_path / "analysis",
        protocol_hash=protocol_hash,
        code_hash=code_hash,
    )
    observed = m.validate_provisional_runtime_decision(
        artifact, protocol_hash=protocol_hash, code_identity_hash=code_hash
    )
    assert observed["selected_chain"] is None
    assert tuple(observed["runtime_candidate_ids"]) == m.RUNTIME_PRIMITIVE_METHODS
    with pytest.raises(PermissionError, match="hand-authored mapping"):
        m.validate_provisional_runtime_decision(
            dict(observed), protocol_hash=protocol_hash, code_identity_hash=code_hash
        )
    invalid = _write_provisional_artifact(
        tmp_path / "invalid",
        protocol_hash=protocol_hash,
        code_hash=code_hash,
        runtime_ids=("M1-SW",),
    )
    with pytest.raises(PermissionError, match="exact primitive schedule"):
        m.validate_provisional_runtime_decision(
            invalid, protocol_hash=protocol_hash, code_identity_hash=code_hash
        )


def test_protocol_sidecar_is_required_well_formed_and_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = tmp_path / "protocol.json"
    sidecar = tmp_path / "protocol.sha256"
    protocol.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(m, "PROTOCOL_SHA256_PATH", sidecar)
    with pytest.raises(RuntimeError, match="missing"):
        m.verify_protocol_hash(protocol)
    sidecar.write_text("A" * 64 + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="lower-case"):
        m.verify_protocol_hash(protocol)
    sidecar.write_text(m.protocol_sha256(protocol) + "\n", encoding="utf-8")
    assert m.verify_protocol_hash(protocol) == m.protocol_sha256(protocol)


def test_confirmation_registry_blocks_placeholder_and_known_panels() -> None:
    with pytest.raises(RuntimeError, match="blocked_missing_untouched_registry"):
        from experiments.m50_backbone_ranking.confirmation import load_registry

        load_registry()

    rows = []
    for index, dataset_id in enumerate(m.CONFIRMATION_DATASET_IDS):
        rows.append(
            {
                "id": dataset_id,
                **m.CONFIRMATION_DATASET_SPECS[dataset_id],
                "minimum_training_examples_per_class": 64,
                "minimum_evaluation_examples_per_class": 20,
                "new": True,
                "audited": True,
                "embedding_manifest": {
                    "sha256": f"{index:064x}",
                    "matrix_sha256": f"{index + 1:064x}",
                },
                "source": "pinned",
                "provenance": "pinned",
            }
        )
    payload = {
        "registry_id": "m50_backbone_ranking_confirmation_v1",
        "status": "audited",
        "ready": True,
        "outcomes_inspected": False,
        "backbones": list(m.MODELS),
        "budgets_per_class": [32, 64],
        "replicate_seeds": [2026082710 + index for index in range(5)],
        "datasets": rows[:4] + [dict(rows[0])],
    }
    # Duplicate identities are rejected before any result/outcome access.
    with pytest.raises(PermissionError, match="duplicated"):
        m.validate_untouched_registry(payload)


def test_confirmation_registry_accepts_exactly_five_valid_new_panels() -> None:
    rows = []
    for index, dataset_id in enumerate(m.CONFIRMATION_DATASET_IDS):
        rows.append(
            {
                "id": dataset_id,
                **m.CONFIRMATION_DATASET_SPECS[dataset_id],
                "minimum_training_examples_per_class": 64,
                "minimum_evaluation_examples_per_class": 20,
                "new": True,
                "audited": True,
                "embedding_manifest": {
                    "sha256": f"{index:064x}",
                    "matrix_sha256": f"{index + 1:064x}",
                },
                "source": "pinned",
                "provenance": "pinned",
            }
        )
    payload = {
        "registry_id": "m50_backbone_ranking_confirmation_v1",
        "status": "audited",
        "ready": True,
        "outcomes_inspected": False,
        "backbones": list(m.MODELS),
        "budgets_per_class": [32, 64],
        "replicate_seeds": [2026082710 + index for index in range(5)],
        "datasets": rows,
    }
    assert len(m.validate_untouched_registry(payload)) == 5
    with pytest.raises(PermissionError, match="exactly five"):
        m.validate_untouched_registry(payload, minimum_panels=4)


def test_confirmation_with_valid_registry_still_blocks_without_embedding_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_hash = "p" * 64
    code_hash = "c" * 64
    monkeypatch.setattr(m, "verify_protocol_hash", lambda: protocol_hash)
    monkeypatch.setattr(m, "code_identity_sha256", lambda *args, **kwargs: code_hash)

    resources = {"candidate_id": "M1-SW", "status": "pass"}
    recipe_binding = _selected_recipe_binding()
    recipe_binding_hash = m.sha256_bytes(
        m.canonical_json(recipe_binding).encode("utf-8")
    )
    lock = {
        "stage": "development",
        "status": "locked_for_future_confirmation",
        "protocol_sha256": protocol_hash,
        "code_identity_sha256": code_hash,
        "selected_candidate": "M1-SW",
        "selected_chain": "M1-SW",
        "chain_kind": "pure_oi",
        "algorithmic_oi_status": "pass",
        "product_policy_status": "not_applicable",
        "resource_candidate_id": "M1-SW",
        "resource_status": "pass",
        "resource_summary_sha256": m.sha256_bytes(
            m.canonical_json(resources).encode("utf-8")
        ),
        "selected_chain_recipe_bindings": recipe_binding,
        "selected_chain_recipe_bindings_sha256": recipe_binding_hash,
    }
    root = tmp_path / "final_analysis"
    root.mkdir()
    _write_json(root / "summary.json", {"resources": resources})
    summary_hash = m.sha256_path(root / "summary.json")
    _write_json(root / "selected_chain_recipe_bindings.json", recipe_binding)
    recipe_file_hash = m.sha256_path(root / "selected_chain_recipe_bindings.json")
    _write_json(root / "development_lock.json", lock)
    lock_hash = m.sha256_path(root / "development_lock.json")
    _write_json(
        root / "analysis_manifest.json",
        {
            "stage": "development_final",
            "protocol_sha256": protocol_hash,
            "code_identity_sha256": code_hash,
            "development_lock_sha256": lock_hash,
            "files": {
                "development_lock.json": {"sha256": lock_hash},
                "summary.json": {"sha256": summary_hash},
                "selected_chain_recipe_bindings.json": {"sha256": recipe_file_hash},
            },
            "selected_chain_recipe_bindings_sha256": recipe_binding_hash,
        },
    )
    registry_rows = [
        {
            "id": dataset_id,
            **m.CONFIRMATION_DATASET_SPECS[dataset_id],
            "minimum_training_examples_per_class": 64,
            "minimum_evaluation_examples_per_class": 20,
            "new": True,
            "audited": True,
            "embedding_manifest": {
                "sha256": f"{index:064x}",
                "matrix_sha256": f"{index + 1:064x}",
            },
            "source": "pinned",
            "provenance": "pinned",
        }
        for index, dataset_id in enumerate(m.CONFIRMATION_DATASET_IDS)
    ]
    registry = tmp_path / "registry.json"
    _write_json(
        registry,
        {
            "registry_id": "m50_backbone_ranking_confirmation_v1",
            "status": "audited",
            "ready": True,
            "outcomes_inspected": False,
            "backbones": list(m.MODELS),
            "budgets_per_class": [32, 64],
            "replicate_seeds": [2026082710 + index for index in range(5)],
            "datasets": registry_rows,
        },
    )
    from experiments.m50_backbone_ranking import confirmation

    output = tmp_path / "confirmation"
    with pytest.raises(RuntimeError, match="blocked_unavailable_embeddings"):
        confirmation.run_confirmation(
            output=output,
            development_decision=root,
            registry=registry,
            outcome_factory=lambda _identity: {"status": "completed"},
        )
    assert not output.exists()
    # v1 is a terminal proof of non-authorization.  Even an exact ready
    # registry and a valid v1 lock must not expose an authorization token.
    assert not hasattr(m, "confirmation_authorization")
    assert not hasattr(m, "require_confirmation_authorization")


def test_final_lock_recipe_binding_is_recomputed_and_sibling_bound(tmp_path: Path) -> None:
    protocol_hash = "p" * 64
    code_hash = "c" * 64
    resources = {"candidate_id": "M1-SW", "status": "pass"}
    binding = _selected_recipe_binding()
    binding_hash = m.sha256_bytes(m.canonical_json(binding).encode("utf-8"))
    root = tmp_path / "final"
    root.mkdir()
    _write_json(root / "summary.json", {"resources": resources})
    _write_json(root / "selected_chain_recipe_bindings.json", binding)
    lock = {
        "stage": "development",
        "status": "locked_for_future_confirmation",
        "protocol_sha256": protocol_hash,
        "code_identity_sha256": code_hash,
        "selected_candidate": "M1-SW",
        "selected_chain": "M1-SW",
        "chain_kind": "pure_oi",
        "algorithmic_oi_status": "pass",
        "product_policy_status": "not_applicable",
        "resource_candidate_id": "M1-SW",
        "resource_status": "pass",
        "resource_summary_sha256": m.sha256_bytes(
            m.canonical_json(resources).encode("utf-8")
        ),
        "selected_chain_recipe_bindings": binding,
        "selected_chain_recipe_bindings_sha256": binding_hash,
    }
    _write_json(root / "development_lock.json", lock)
    file_hashes = {
        name: {"sha256": m.sha256_path(root / name)}
        for name in (
            "development_lock.json",
            "summary.json",
            "selected_chain_recipe_bindings.json",
        )
    }
    _write_json(
        root / "analysis_manifest.json",
        {
            "stage": "development_final",
            "protocol_sha256": protocol_hash,
            "code_identity_sha256": code_hash,
            "development_lock_sha256": file_hashes["development_lock.json"]["sha256"],
            "selected_chain_recipe_bindings_sha256": binding_hash,
            "files": file_hashes,
        },
    )
    assert m.validate_development_lock(
        root, protocol_hash=protocol_hash, code_identity_hash=code_hash
    )["selected_chain"] == "M1-SW"

    # Updating the sibling file identity cannot authorize a changed binding:
    # the lock and sidecar must still carry the same recomputed semantic object.
    changed = dict(binding)
    changed["derived_recipe_sha256"] = {"unexpected": "0" * 64}
    _write_json(root / "selected_chain_recipe_bindings.json", changed)
    file_hashes["selected_chain_recipe_bindings.json"] = {
        "sha256": m.sha256_path(root / "selected_chain_recipe_bindings.json")
    }
    _write_json(root / "analysis_manifest.json", {
        "stage": "development_final",
        "protocol_sha256": protocol_hash,
        "code_identity_sha256": code_hash,
        "development_lock_sha256": file_hashes["development_lock.json"]["sha256"],
        "selected_chain_recipe_bindings_sha256": binding_hash,
        "files": file_hashes,
    })
    with pytest.raises(PermissionError, match="sidecar does not match|derived hash"):
        m.validate_development_lock(
            root, protocol_hash=protocol_hash, code_identity_hash=code_hash
        )


@pytest.mark.parametrize(
    ("selected_candidate", "selected_chain", "guardrail_base_id"),
    [
        ("M0-SW", "M0-SW", None),
        ("M1-SW", "M1-SW", None),
        ("M0-SW", "F", None),
        ("M1-SW", "F", None),
        ("M0-SW", "G", "M0-SW"),
        ("M1-SW", "G", "F"),
    ],
)
def test_final_lock_recipe_binding_accepts_only_closed_chain_compositions(
    tmp_path: Path,
    selected_candidate: str,
    selected_chain: str,
    guardrail_base_id: str | None,
) -> None:
    binding = _selected_recipe_binding(
        selected_candidate=selected_candidate,
        selected_chain=selected_chain,
        guardrail_base_id=guardrail_base_id,
    )
    root = tmp_path / f"{selected_candidate}-{selected_chain}-{guardrail_base_id}"
    protocol_hash, code_hash = _write_final_lock_artifact(root, binding)
    observed = m.validate_development_lock(
        root, protocol_hash=protocol_hash, code_identity_hash=code_hash
    )
    assert observed["selected_chain"] == selected_chain
    assert observed["selected_chain_recipe_bindings"] == binding


@pytest.mark.parametrize("corruption", ["M0", "M1", "F", "G", "seed", "component", "derived"])
def test_final_lock_recipe_binding_rejects_independent_chain_and_hash_corruption(
    tmp_path: Path, corruption: str
) -> None:
    selected_candidate = "M1-SW"
    binding = _selected_recipe_binding()
    root = tmp_path / corruption
    protocol_hash, code_hash = _write_final_lock_artifact(root, binding)
    changed = json.loads(json.dumps(binding))
    if corruption == "M0":
        changed["selected_candidate"] = "M0-SW"
    elif corruption == "M1":
        changed["selected_candidate"] = "M0-SW"
    elif corruption == "F":
        changed["selected_chain"] = "F"
    elif corruption == "G":
        changed["selected_chain"] = "G"
    elif corruption == "seed":
        changed["deterministic_seed_rule"]["food"]["selector_panel_seed"] = "SEED + fold"
    elif corruption == "component":
        changed["component_recipe_family_sha256"][selected_candidate][
            "food_recipe_family_sha256"
        ] = "Z" * 64
    elif corruption == "derived":
        changed["derived_recipe_sha256"] = {"F": "0" * 64}
    _refresh_final_lock_artifact(root, binding=changed)
    with pytest.raises(PermissionError):
        m.validate_development_lock(
            root, protocol_hash=protocol_hash, code_identity_hash=code_hash
        )


def test_final_lock_recipe_binding_rejects_outer_hash_corruption(tmp_path: Path) -> None:
    binding = _selected_recipe_binding()
    root = tmp_path / "outer-hash"
    protocol_hash, code_hash = _write_final_lock_artifact(root, binding)
    _refresh_final_lock_artifact(root, binding_hash_override="0" * 64)
    with pytest.raises(PermissionError):
        m.validate_development_lock(
            root, protocol_hash=protocol_hash, code_identity_hash=code_hash
        )
