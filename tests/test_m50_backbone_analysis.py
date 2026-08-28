"""Focused fail-closed orchestration tests for M50 analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.m50_backbone_ranking import analysis
from experiments.m50_backbone_ranking import manifest as experiment_manifest
from experiments.m50_backbone_ranking import reporting
from experiments.m50_backbone_ranking import statistics


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _artifact(root: Path) -> None:
    root.mkdir()
    rows = [{"candidate_id": "A", "value": 1.0}]
    (root / "rows.jsonl").write_text(
        "".join(experiment_manifest.canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    tables = {
        "rows": {
            "row_count": 1,
            "sha256": analysis.canonical_table_sha256(rows),
            "encoding": "canonical_jsonl_utf8",
        }
    }
    raw = {
        "artifact_status": "completed",
        "study": "tiny-study",
        "protocol": {"sha256": "a" * 64},
        "repository_provenance": {"code_identity_sha256": "b" * 64},
        "tables": tables,
        "rows": rows,
    }
    _write_json(root / "raw_results.json", raw)
    raw_payload = (root / "raw_results.json").read_bytes()
    artifact_manifest = {
        "artifact_status": "completed",
        "study": "tiny-study",
        "protocol": {"sha256": "a" * 64},
        "repository_provenance": {"code_identity_sha256": "b" * 64},
        "raw_results": {
            "sha256": __import__("hashlib").sha256(raw_payload).hexdigest(),
            "size_bytes": len(raw_payload),
        },
        "tables": tables,
    }
    _write_json(root / "manifest.json", artifact_manifest)


def test_completed_artifact_verification_rejects_table_tampering(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _artifact(root)
    loaded = analysis.verify_completed_artifact(
        root,
        table_names=("rows",),
        expected_study="tiny-study",
        verify_current_identity=False,
    )
    assert loaded["tables"]["rows"] == [{"candidate_id": "A", "value": 1.0}]
    raw = json.loads((root / "raw_results.json").read_text())
    raw["rows"][0]["value"] = 2.0
    _write_json(root / "raw_results.json", raw)
    with pytest.raises(ValueError, match="raw_results identity"):
        analysis.verify_completed_artifact(
            root,
            table_names=("rows",),
            expected_study="tiny-study",
            verify_current_identity=False,
        )


def test_completed_artifact_requires_exact_tables_and_completed_status(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _artifact(root)
    with pytest.raises(ValueError, match="exactly the required"):
        analysis.verify_completed_artifact(
            root,
            table_names=("rows", "missing"),
            verify_current_identity=False,
        )
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["artifact_status"] = "stopped"
    _write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="does not exactly match"):
        analysis.verify_completed_artifact(
            root, table_names=("rows",), verify_current_identity=False
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("protocol", {"sha256": "c" * 64}),
        ("repository_provenance", {"code_identity_sha256": "d" * 64}),
    ],
)
def test_completed_artifact_rejects_manifest_only_identity_tampering(
    tmp_path: Path, field: str, replacement: object
) -> None:
    root = tmp_path / field
    _artifact(root)
    artifact_manifest = json.loads((root / "manifest.json").read_text())
    artifact_manifest[field] = replacement
    _write_json(root / "manifest.json", artifact_manifest)
    with pytest.raises(ValueError, match="does not exactly match hash-bound raw_results"):
        analysis.verify_completed_artifact(
            root, table_names=("rows",), verify_current_identity=False
        )


def test_cli_dispatches_only_v1_development_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        analysis,
        "analyze_development",
        lambda *args: calls.append(("development", args)),
    )
    monkeypatch.setattr(
        analysis,
        "finalize_development",
        lambda *args: calls.append(("finalize", args)),
    )
    assert analysis.main(["development", "--input", "in", "--output", "out"]) == 0
    assert analysis.main(
        ["finalize", "--provisional", "provisional", "--runtime", "runtime", "--output", "out"]
    ) == 0
    assert [name for name, _args in calls] == ["development", "finalize"]
    with pytest.raises(SystemExit):
        analysis.main(
            [
                "confirmation",
                "--input",
                "in",
                "--lock",
                "lock",
                "--registry",
                "registry",
                "--output",
                "out",
            ]
        )
    assert not hasattr(analysis, "analyze_confirmation")


def test_provisional_decision_hash_is_bound_to_analysis_manifest(tmp_path: Path) -> None:
    root = tmp_path / "provisional"
    decision = {
        "stage": "development_provisional",
        "status": "provisional_runtime_pending",
    }
    reporting.write_report_bundle(
        root,
        summary={},
        development_lock=decision,
        panel_metrics=(),
        rank_auc=(),
        lineage={
            "stage": "development_provisional",
            "status": "provisional_runtime_pending",
            "protocol_sha256": "a" * 64,
            "code_identity_sha256": "b" * 64,
            "input_manifest_sha256": "c" * 64,
        },
    )
    digest = analysis._sha256(root / "development_lock.json")
    observed, _manifest, observed_hash = analysis._load_provisional(root)
    assert observed == decision
    assert observed_hash == digest
    _write_json(root / "development_lock.json", {**decision, "selected_candidate": "M0-SW"})
    with pytest.raises(ValueError, match="SHA-256 mismatch|development-lock hash mismatch"):
        analysis._load_provisional(root)


def _complete_food_loaded() -> dict[str, object]:
    panels = [
        (backbone, replicate, arm, budget)
        for backbone in statistics.FOOD_BACKBONES
        for replicate in statistics.FOOD_REPLICATES
        for arm in statistics.FOOD_ARMS
        for budget in statistics.FOOD_BUDGETS
    ]
    refined = {"B", "M1-SW", "M1-CB"}

    def outcome_row(
        backbone: str,
        replicate: int,
        arm: str,
        budget: int,
        candidate: str,
    ) -> dict[str, object]:
        folds = []
        for fold in range(5):
            config = {
                "candidate_id": candidate,
                "fold": fold,
                "frozen_test_recipe": True,
            }
            folds.append(
                {
                    "fold": fold,
                    "candidate_config_identity": config,
                    "candidate_config_sha256": hashlib.sha256(
                        experiment_manifest.canonical_json(config).encode("utf-8")
                    ).hexdigest(),
                }
            )
        return {
            "stage": "development",
            "model": backbone,
            "backbone": backbone,
            "replicate": replicate,
            "arm": arm,
            "budget": budget,
            "candidate_id": candidate,
            "prototype_refinement_enabled": candidate in refined,
            "prototype_refinement": {},
            "conditioning_diagnostics": {},
            "folds": folds,
            "score": 0.5,
            "fit_wall_seconds": 0.1,
            "fit_cpu_seconds": 0.1,
            "score_fixed_wall_seconds": 0.1,
            "score_fixed_cpu_seconds": 0.1,
            "total_wall_seconds": 0.2,
            "total_cpu_seconds": 0.2,
            "warmup_excluded": True,
            "status": "ok",
            "error": None,
        }

    selector_rows = [
        outcome_row(backbone, replicate, arm, budget, candidate)
        for backbone, replicate, arm, budget in panels
        for candidate in experiment_manifest.SELECTOR_METHODS
    ]
    reference_rows = [
        {
            "model": backbone,
            "backbone": backbone,
            "replicate": replicate,
            "arm": arm,
            "head": head,
            "test_accuracy": 0.5,
        }
        for backbone in statistics.FOOD_BACKBONES
        for replicate in statistics.FOOD_REPLICATES
        for arm in statistics.FOOD_ARMS
        for head in statistics.REFERENCE_HEADS
    ]
    capped_rows = [
        outcome_row(
            backbone,
            replicate,
            arm,
            budget,
            "LP-CAPPED-2048",
        )
        for backbone, replicate, arm, budget in panels
    ]
    recipe_hashes: dict[str, set[str]] = {}
    for row in [*selector_rows, *capped_rows]:
        candidate = str(row["candidate_id"])
        for fold in row["folds"]:  # type: ignore[union-attr]
            recipe_hashes.setdefault(candidate, set()).add(
                str(fold["candidate_config_sha256"])
            )
    ordered_recipe_hashes = {
        candidate: sorted(values) for candidate, values in sorted(recipe_hashes.items())
    }
    recipe_family_hashes = {
        candidate: hashlib.sha256(
            experiment_manifest.canonical_json(
                {
                    "candidate_id": candidate,
                    "candidate_config_sha256": values,
                }
            ).encode("utf-8")
        ).hexdigest()
        for candidate, values in ordered_recipe_hashes.items()
    }
    return {
        "tables": {
            "selector_rows": selector_rows,
            "reference_rows": reference_rows,
            "probe_rows": [
                dict(row)
                for row in selector_rows
                if row["candidate_id"] == "LP-FULL"
            ],
            "capped_probe_rows": capped_rows,
            "baseline_parity_rows": [
                {
                    "backbone": backbone,
                    "replicate": replicate,
                    "arm": arm,
                    "budget": budget,
                    "candidate_id": candidate,
                    "exact": True,
                    "delta": 0.0,
                }
                for backbone, replicate, arm, budget in panels
                for candidate in ("A", "B", "LP-FULL")
            ],
        },
        "raw": {
            "recipe_bindings": {
                "candidate_config_sha256": ordered_recipe_hashes,
                "recipe_family_sha256": recipe_family_hashes,
                "deterministic_seed_rule": dict(analysis.FOOD_RECIPE_SEED_RULE),
            },
            "determinism_verification": {
                candidate: {
                    "exact": True,
                    "first_signature_sha256": "a" * 64,
                    "second_signature_sha256": "a" * 64,
                    "runtime_fields_excluded": True,
                    "panel_identity": {
                        "model": statistics.FOOD_BACKBONES[0],
                        "replicate": statistics.FOOD_REPLICATES[0],
                        "arm": statistics.FOOD_ARMS[0],
                        "budget": statistics.FOOD_BUDGETS[0],
                        "candidate_id": candidate,
                    },
                }
                for candidate in (
                    *experiment_manifest.SELECTOR_METHODS,
                    "LP-CAPPED-2048",
                )
            }
        },
    }


def test_exact_food_completeness_path_fails_closed() -> None:
    loaded = _complete_food_loaded()
    analysis._validate_food_completed_artifact(loaded)
    deterministic = loaded["raw"]["determinism_verification"]  # type: ignore[index]
    capped_identity = deterministic.pop("LP-CAPPED-2048")
    with pytest.raises(ValueError, match="deterministic-repeat gate"):
        analysis._validate_food_completed_artifact(loaded)
    deterministic["LP-CAPPED-2048"] = capped_identity
    loaded["tables"]["selector_rows"].pop()  # type: ignore[index]
    with pytest.raises(ValueError, match="exact frozen 4,200-row grid"):
        analysis._validate_food_completed_artifact(loaded)


@pytest.mark.parametrize(
    ("table", "field", "value", "message"),
    [
        ("probe_rows", "status", "error", "non-ok"),
        ("probe_rows", "prototype_refinement_enabled", 0, "refinement flag"),
        ("probe_rows", "score", float("nan"), "score/timing"),
        ("probe_rows", "total_wall_seconds", None, "score/timing"),
        ("capped_probe_rows", "candidate_id", "LP-FULL", "unknown candidate"),
        ("capped_probe_rows", "prototype_refinement_enabled", True, "refinement flag"),
        ("capped_probe_rows", "fit_cpu_seconds", float("inf"), "score/timing"),
    ],
)
def test_food_probe_surfaces_reject_malformed_completed_rows(
    table: str, field: str, value: object, message: str
) -> None:
    loaded = _complete_food_loaded()
    loaded["tables"][table][0][field] = value  # type: ignore[index]
    with pytest.raises(ValueError, match=message):
        analysis._validate_food_completed_artifact(loaded)


@pytest.mark.parametrize("field", ["score", "fit_wall_seconds", "prototype_refinement"])
def test_food_lp_full_duplicate_surface_requires_exact_row_equality(field: str) -> None:
    loaded = _complete_food_loaded()
    row = loaded["tables"]["probe_rows"][0]  # type: ignore[index]
    row[field] = 0.6 if field == "score" else (0.3 if field == "fit_wall_seconds" else {"changed": 1})
    with pytest.raises(ValueError, match="do not agree exactly"):
        analysis._validate_food_completed_artifact(loaded)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_hash", "deterministic-repeat gate"),
        ("bad_hash", "deterministic-repeat gate"),
        ("mismatched_hash", "deterministic-repeat gate"),
        ("wrong_panel_candidate", "deterministic-repeat gate"),
        ("extra_panel_alias", "deterministic-repeat gate"),
    ],
)
def test_food_determinism_descriptor_is_closed_and_hash_bound(
    mutation: str, message: str
) -> None:
    loaded = _complete_food_loaded()
    descriptor = loaded["raw"]["determinism_verification"]["A"]  # type: ignore[index]
    if mutation == "missing_hash":
        descriptor.pop("first_signature_sha256")
    elif mutation == "bad_hash":
        descriptor["first_signature_sha256"] = "not-a-hash"
    elif mutation == "mismatched_hash":
        descriptor["second_signature_sha256"] = "b" * 64
    elif mutation == "wrong_panel_candidate":
        descriptor["panel_identity"]["candidate_id"] = "B"
    else:
        descriptor["panel_identity"]["backbone"] = statistics.FOOD_BACKBONES[0]
    with pytest.raises(ValueError, match=message):
        analysis._validate_food_completed_artifact(loaded)


def test_food_determinism_rejects_valid_but_later_frozen_panel() -> None:
    loaded = _complete_food_loaded()
    later = experiment_manifest.development_panels()[1]
    descriptor = loaded["raw"]["determinism_verification"]["A"]  # type: ignore[index]
    descriptor["panel_identity"] = {
        **later.identity(),
        "candidate_id": "A",
    }
    with pytest.raises(ValueError, match="deterministic-repeat gate"):
        analysis._validate_food_completed_artifact(loaded)


@pytest.mark.parametrize("mutation", ["fold_hash", "family_hash", "seed_rule"])
def test_food_recipe_bindings_reject_mixed_or_hand_authored_hashes(
    mutation: str,
) -> None:
    loaded = _complete_food_loaded()
    if mutation == "fold_hash":
        loaded["tables"]["selector_rows"][0]["folds"][0][  # type: ignore[index]
            "candidate_config_sha256"
        ] = "f" * 64
    elif mutation == "family_hash":
        loaded["raw"]["recipe_bindings"]["recipe_family_sha256"]["A"] = "f" * 64  # type: ignore[index]
    else:
        loaded["raw"]["recipe_bindings"]["deterministic_seed_rule"] = {  # type: ignore[index]
            "selector_panel_seed": "outcome_dependent"
        }
    with pytest.raises(ValueError, match="recipe"):
        analysis._validate_food_completed_artifact(loaded)


def test_runtime_recipe_bindings_require_exact_resolved_folds_and_seed_rule() -> None:
    rows = _runtime_rows()
    bindings = _test_recipe_bindings(rows, analysis.RUNTIME_RECIPE_SEED_RULE)
    observed = analysis._validate_recipe_bindings(
        bindings,
        rows,
        expected_candidates=set(statistics.RUNTIME_PRIMITIVE_IDS),
        expected_seed_rule=analysis.RUNTIME_RECIPE_SEED_RULE,
    )
    assert observed == bindings
    bad = json.loads(json.dumps(bindings))
    bad["deterministic_seed_rule"]["repeat_effect"] = "changes_dataset"
    with pytest.raises(ValueError, match="seed rule mismatch"):
        analysis._validate_recipe_bindings(
            bad,
            rows,
            expected_candidates=set(statistics.RUNTIME_PRIMITIVE_IDS),
            expected_seed_rule=analysis.RUNTIME_RECIPE_SEED_RULE,
        )


def test_selected_guardrail_recipe_binding_closes_base_trigger_and_cap_components() -> None:
    components = set(statistics.RUNTIME_PRIMITIVE_IDS)
    food = {
        "recipe_family_sha256": {
            candidate: hashlib.sha256(f"food:{candidate}".encode()).hexdigest()
            for candidate in components
        },
        "deterministic_seed_rule": dict(analysis.FOOD_RECIPE_SEED_RULE),
    }
    runtime = {
        "recipe_family_sha256": {
            candidate: hashlib.sha256(f"runtime:{candidate}".encode()).hexdigest()
            for candidate in components
        },
        "deterministic_seed_rule": dict(analysis.RUNTIME_RECIPE_SEED_RULE),
    }
    binding = analysis._selected_chain_recipe_binding(
        selected_candidate="M1-SW",
        selected_chain="G",
        guardrail_base_id="F",
        food_bindings=food,
        runtime_bindings=runtime,
    )
    assert binding["primitive_component_ids"] == [
        "B",
        "LP-CAPPED-2048",
        "M1-SW",
    ]
    assert set(binding["derived_recipe_sha256"]) == {"F", "G"}
    assert binding["derived_recipe_identity"]["G"]["trigger"] == {
        "scope": "panel_wide_any_backbone",
        "B_refinement_activity_gte": 0.50,
        "absolute_B_minus_M1_score_gte": 0.05,
        "outcome_fields_allowed": False,
    }
    missing = json.loads(json.dumps(runtime))
    missing["recipe_family_sha256"].pop("LP-CAPPED-2048")
    with pytest.raises(ValueError, match="missing component"):
        analysis._selected_chain_recipe_binding(
            selected_candidate="M1-SW",
            selected_chain="G",
            guardrail_base_id="F",
            food_bindings=food,
            runtime_bindings=missing,
        )


@pytest.mark.parametrize("name", ["report.md", "panel_metrics.csv", "selector_regret.svg"])
def test_provisional_loader_verifies_every_analysis_file_hash(
    tmp_path: Path, name: str
) -> None:
    root = tmp_path / name.replace(".", "_")
    reporting.write_report_bundle(
        root,
        summary={},
        development_lock={
            "stage": "development_provisional",
            "status": "provisional_runtime_pending",
        },
        panel_metrics=({"candidate_id": "A"},),
        rank_auc=(),
        lineage={
            "stage": "development_provisional",
            "status": "provisional_runtime_pending",
            "protocol_sha256": "a" * 64,
            "code_identity_sha256": "b" * 64,
            "input_manifest_sha256": "c" * 64,
        },
    )
    (root / name).write_bytes((root / name).read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        analysis._load_provisional(root)


def _runtime_rows() -> list[dict[str, object]]:
    rows = []
    for method in statistics.RUNTIME_PRIMITIVE_IDS:
        for backbone in statistics.FOOD_BACKBONES:
            for arm in statistics.FOOD_ARMS:
                for budget in statistics.FOOD_RUNTIME_BUDGETS:
                    for repeat in statistics.FOOD_REPLICATES:
                        wall = 2.0 if method == "M0-SW" else 1.0
                        block = f"{backbone}:{arm}:{budget}".encode()
                        folds = []
                        for fold in range(5):
                            config = {
                                "candidate_id": method,
                                "fold": fold,
                                "budget": budget,
                                "frozen_test_recipe": True,
                            }
                            folds.append(
                                {
                                    "fold": fold,
                                    "candidate_config_identity": config,
                                    "candidate_config_sha256": hashlib.sha256(
                                        experiment_manifest.canonical_json(config).encode(
                                            "utf-8"
                                        )
                                    ).hexdigest(),
                                }
                            )
                        rows.append(
                            {
                                "candidate_id": method,
                                "backbone": backbone,
                                "arm": arm,
                                "budget": budget,
                                "repeat": repeat,
                                "fit_wall_seconds": wall,
                                "fit_cpu_seconds": wall,
                                "conditioning_fit_wall_seconds": 0.0,
                                "conditioning_fit_cpu_seconds": 0.0,
                                "oi_fit_wall_seconds": 0.0,
                                "oi_fit_cpu_seconds": 0.0,
                                "score_fixed_wall_seconds": 0.0,
                                "score_fixed_cpu_seconds": 0.0,
                                "total_wall_seconds": wall,
                                "total_cpu_seconds": wall,
                                "peak_memory_mb": 100.0,
                                "candidate_score": 0.1,
                                "prototype_refinement": {"applied_rate": 0.0},
                                "conditioning_diagnostics": {},
                                "folds": folds,
                                "dataset_identity": {
                                    "values_sha256": hashlib.sha256(block).hexdigest(),
                                    "labels_sha256": hashlib.sha256(
                                        block + b":labels"
                                    ).hexdigest(),
                                    "row_count": 40 * budget,
                                    "feature_count": 128,
                                    "metadata": {
                                        "bridge_arm": arm,
                                        "bridge_lambda": 0.0,
                                        "bridge_nuisance_strength": 0.0,
                                    },
                                },
                                "warmup_excluded": True,
                            }
                        )
    return rows


def _test_recipe_rows(candidate_ids: tuple[str, ...]) -> list[dict[str, object]]:
    rows = []
    for candidate_id in candidate_ids:
        folds = []
        for fold in range(5):
            config = {
                "candidate_id": candidate_id,
                "fold": fold,
                "frozen_test_recipe": True,
            }
            folds.append(
                {
                    "fold": fold,
                    "candidate_config_identity": config,
                    "candidate_config_sha256": hashlib.sha256(
                        experiment_manifest.canonical_json(config).encode("utf-8")
                    ).hexdigest(),
                }
            )
        rows.append({"candidate_id": candidate_id, "folds": folds})
    return rows


def _test_recipe_bindings(
    rows: list[dict[str, object]], seed_rule: dict[str, str]
) -> dict[str, object]:
    hashes: dict[str, set[str]] = {}
    for row in rows:
        candidate_id = str(row["candidate_id"])
        for fold in row["folds"]:  # type: ignore[union-attr]
            hashes.setdefault(candidate_id, set()).add(
                str(fold["candidate_config_sha256"])
            )
    ordered = {
        candidate_id: sorted(values) for candidate_id, values in sorted(hashes.items())
    }
    families = {
        candidate_id: hashlib.sha256(
            experiment_manifest.canonical_json(
                {
                    "candidate_id": candidate_id,
                    "candidate_config_sha256": values,
                }
            ).encode("utf-8")
        ).hexdigest()
        for candidate_id, values in ordered.items()
    }
    return {
        "candidate_config_sha256": ordered,
        "recipe_family_sha256": families,
        "deterministic_seed_rule": dict(seed_rule),
    }


def _pass_derived_gate(candidate_id: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "status": "pass",
        "gates": {
            f"{arm}:{head}": {
                "candidate_id": candidate_id,
                "comparator_id": "LP-FULL",
                "status": "pass",
            }
            for arm, head in statistics.PRIMARY_CELLS
        },
    }


def test_runtime_grid_and_guardrail_are_derived_from_unique_primitives() -> None:
    rows = _runtime_rows()
    assert len(rows) == 4_500
    lookup = analysis._validate_runtime_primitives(rows)
    base = [row for row in rows if row["candidate_id"] == "M1-SW"]
    untriggered = analysis._derive_g_runtime_rows(
        lookup, base_candidate_id="M1-SW", base_rows=base
    )
    assert len(untriggered) == 750
    assert all(row["triggered"] is False for row in untriggered)
    assert all(set(row["primitive_components"]) == {"B", "M1-SW"} for row in untriggered)

    for row in rows:
        if (
            row["candidate_id"] == "B"
            and row["backbone"] == statistics.FOOD_BACKBONES[0]
            and row["arm"] == "baseline"
            and row["budget"] == 64
        ):
            row["prototype_refinement"] = {"applied_rate": 0.50}
    triggered_lookup = analysis._validate_runtime_primitives(rows)
    triggered = analysis._derive_g_runtime_rows(
        triggered_lookup, base_candidate_id="M1-SW", base_rows=base
    )
    panel = [
        row
        for row in triggered
        if row["arm"] == "baseline" and row["budget"] == 64 and row["repeat"] == 0
    ]
    assert all(row["triggered"] is True for row in panel)
    assert all(
        set(row["primitive_components"]) == {"B", "M1-SW", "LP-CAPPED-2048"}
        for row in panel
    )
    with pytest.raises(ValueError, match="duplicate runtime primitive"):
        analysis._validate_runtime_primitives([*rows, dict(rows[0])])
    mixed = [dict(row) for row in rows]
    first = mixed[0]
    conflicting = next(
        row
        for row in mixed
        if row["candidate_id"] != first["candidate_id"]
        and all(
            row[field] == first[field]
            for field in ("backbone", "arm", "budget", "repeat")
        )
    )
    conflicting["dataset_identity"] = {
        **dict(conflicting["dataset_identity"]),  # type: ignore[arg-type]
        "values_sha256": "f" * 64,
    }
    with pytest.raises(ValueError, match="mixed dataset identities"):
        analysis._validate_runtime_primitives(mixed)
    changed_repeat = [dict(row) for row in rows]
    repeated_panel = {
        field: changed_repeat[0][field] for field in ("backbone", "arm", "budget")
    }
    for row in changed_repeat:
        if row["repeat"] == 1 and all(
            row[field] == value for field, value in repeated_panel.items()
        ):
            row["dataset_identity"] = {
                **dict(row["dataset_identity"]),  # type: ignore[arg-type]
                "values_sha256": "e" * 64,
            }
    with pytest.raises(ValueError, match="changed across timing repeats"):
        analysis._validate_runtime_primitives(changed_repeat)
    nondeterministic = [dict(row) for row in rows]
    target = next(
        row
        for row in nondeterministic
        if row["candidate_id"] == "M1-SW"
        and row["backbone"] == statistics.FOOD_BACKBONES[0]
        and row["arm"] == "baseline"
        and row["budget"] == 64
        and row["repeat"] == 1
    )
    target["candidate_score"] = 0.1000000000000001
    with pytest.raises(ValueError, match="structural diagnostics changed across repeats"):
        analysis._validate_runtime_primitives(nondeterministic)
    malformed = [dict(row) for row in rows]
    malformed[0]["dataset_identity"] = {
        "values_sha256": "a" * 64,
        "labels_sha256": "b" * 64,
    }
    with pytest.raises(ValueError, match="exact closed schema"):
        analysis._validate_runtime_primitives(malformed)


def test_finalize_settles_refinement_before_deriving_bound_fusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_rows = _runtime_rows()
    food_recipe_rows = _test_recipe_rows(
        (*experiment_manifest.SELECTOR_METHODS, "LP-CAPPED-2048")
    )
    provisional = {
        "stage": "development_provisional",
        "status": "provisional_runtime_pending",
        "selected_candidate": "M0-SW",
        "selected_chain": None,
        "runtime_pending": True,
        "runner_up_allowed": False,
        "candidate_gates": {
            "M0-SW": {"status": "fusion_required", "requires_fusion": True},
            "M1-SW": {"status": "fusion_required", "requires_fusion": True},
        },
        "unrefined_requires_speed_ratio": True,
        "protocol_sha256": "p",
        "code_identity_sha256": "c",
        "input_manifest_sha256": "food-hash",
        "development_input_root": str(tmp_path / "food"),
    }
    monkeypatch.setattr(
        analysis,
        "_load_provisional",
        lambda _root: (
            provisional,
            {"analysis_manifest_sha256": "analysis-hash"},
            "decision-hash",
        ),
    )
    development = {
        "manifest_sha256": "food-hash",
        "tables": {
            "selector_rows": food_recipe_rows[:-1],
            "reference_rows": [],
            "capped_probe_rows": food_recipe_rows[-1:],
        },
        "raw": {
            "recipe_bindings": _test_recipe_bindings(
                food_recipe_rows, analysis.FOOD_RECIPE_SEED_RULE
            )
        },
    }
    runtime = {
        "manifest": {
            "identity": {
                "provisional_decision_sha256": "decision-hash",
                "provisional_analysis_manifest_sha256": "analysis-hash",
                "methods": list(statistics.RUNTIME_PRIMITIVE_IDS),
                "arms": list(statistics.FOOD_ARMS),
                "deterministic_seed_rule": dict(analysis.RUNTIME_RECIPE_SEED_RULE),
            }
        },
        "tables": {"runtime_rows": runtime_rows},
        "raw": {
            "recipe_bindings": _test_recipe_bindings(
                runtime_rows, analysis.RUNTIME_RECIPE_SEED_RULE
            )
        },
        "protocol_sha256": "p",
        "code_identity_sha256": "c",
        "manifest_sha256": "runtime-hash",
    }
    monkeypatch.setattr(
        analysis,
        "verify_completed_artifact",
        lambda root, **_kwargs: runtime if str(root).endswith("runtime") else development,
    )
    monkeypatch.setattr(analysis, "_validate_food_completed_artifact", lambda _value: None)
    base_metrics = [{"candidate_id": "LP-FULL"}]
    monkeypatch.setattr(statistics, "food_panel_metrics", lambda *args, candidate_ids, **kwargs: [{"candidate_id": "F"}] if tuple(candidate_ids) == ("F",) else base_metrics)
    monkeypatch.setattr(statistics, "rank_auc_rows", lambda _rows: [])
    monkeypatch.setattr(statistics, "selection_rate_summary", lambda _rows: [])
    monkeypatch.setattr(statistics, "rank_auc_summary", lambda _rows: [])
    observed: dict[str, object] = {}

    def derive(_rows: object, *, raw_candidate_id: str, m50_candidate_id: str, backbones: object) -> list[dict[str, object]]:
        observed["fusion_recipe"] = (raw_candidate_id, m50_candidate_id)
        return [{"candidate_id": "F"}]

    monkeypatch.setattr(analysis.fusion, "derive_fusion_rows", derive)
    monkeypatch.setattr(statistics, "candidate_product_gates", lambda *_args, candidate_id, **_kwargs: _pass_derived_gate(candidate_id))

    def resources(rows: object, *, candidate_id: str, **_kwargs: object) -> dict[str, object]:
        materialized = list(rows)  # type: ignore[arg-type]
        observed["resource_candidate"] = candidate_id
        observed["derived_runtime"] = [
            row for row in materialized if row.get("candidate_id") == "F"
        ]
        return {
            "candidate_id": candidate_id,
            "status": "pass",
            "ratio_rows": [],
            "full_panel_ratio_rows": [],
            "arm_gates": {},
            "full_panel_arm_gates": {},
            "pooled_summary": {},
            "scaling_by_budget": [],
            "secondary_runtime": {
                "inferential": False,
                "promotion_gate": False,
                "arm_summary": {},
                "scaling_by_arm_budget": [],
            },
        }

    monkeypatch.setattr(statistics, "resource_gates", resources)
    monkeypatch.setattr(analysis, "_read_object", lambda _path: {"factorial": {"effects": {}}})
    monkeypatch.setattr(
        reporting,
        "write_report_bundle",
        lambda _output, **kwargs: observed.update({"final_lock": kwargs["development_lock"]}) or Path(_output),
    )
    result = analysis.finalize_development(
        tmp_path / "provisional",
        tmp_path / "runtime",
        tmp_path / "final",
        n_resamples=20,
        verify_current_identity=False,
    )
    assert result == tmp_path / "final"
    assert observed["fusion_recipe"] == ("B", "M1-SW")
    assert observed["resource_candidate"] == "F"
    assert len(observed["derived_runtime"]) == 750  # type: ignore[arg-type]
    final_lock = observed["final_lock"]
    assert final_lock["selected_candidate"] == "M1-SW"  # type: ignore[index]
    assert final_lock["selected_chain"] == "F"  # type: ignore[index]
    assert final_lock["chain_kind"] == "oi_fusion"  # type: ignore[index]
    assert final_lock["refinement_runtime_ratio_by_arm"] == {
        arm: 2.0 for arm in statistics.FOOD_ARMS
    }  # type: ignore[index]
    assert final_lock["resource_candidate_id"] == "F"  # type: ignore[index]
    assert isinstance(final_lock["resource_summary_sha256"], str)  # type: ignore[index]
    assert final_lock["status"] == "locked_for_future_confirmation"  # type: ignore[index]
    recipe = final_lock["selected_chain_recipe_bindings"]  # type: ignore[index]
    assert recipe["primitive_component_ids"] == ["B", "M1-SW"]
    assert set(recipe["derived_recipe_sha256"]) == {"F"}
    assert isinstance(final_lock["selected_chain_recipe_bindings_sha256"], str)  # type: ignore[index]
    assert "locked_candidate" not in final_lock  # type: ignore[operator]


def test_finalize_pure_chain_uses_each_runtime_primitive_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_rows = _runtime_rows()
    food_recipe_rows = _test_recipe_rows(
        (*experiment_manifest.SELECTOR_METHODS, "LP-CAPPED-2048")
    )
    provisional = {
        "stage": "development_provisional",
        "status": "provisional_runtime_pending",
        "selected_candidate": "M1-SW",
        "selected_chain": None,
        "runtime_pending": True,
        "runner_up_allowed": False,
        "candidate_gates": {
            "M1-SW": {
                "status": "pass",
                "requires_fusion": False,
                "guardrail_eligible": False,
            }
        },
        "unrefined_requires_speed_ratio": False,
        "protocol_sha256": "p",
        "code_identity_sha256": "c",
        "input_manifest_sha256": "food-hash",
        "development_input_root": str(tmp_path / "food"),
    }
    monkeypatch.setattr(
        analysis,
        "_load_provisional",
        lambda _root: (
            provisional,
            {"analysis_manifest_sha256": "analysis-hash"},
            "decision-hash",
        ),
    )
    development = {
        "manifest_sha256": "food-hash",
        "tables": {
            "selector_rows": food_recipe_rows[:-1],
            "reference_rows": [],
            "capped_probe_rows": food_recipe_rows[-1:],
        },
        "raw": {
            "recipe_bindings": _test_recipe_bindings(
                food_recipe_rows, analysis.FOOD_RECIPE_SEED_RULE
            )
        },
    }
    runtime = {
        "manifest": {
            "identity": {
                "provisional_decision_sha256": "decision-hash",
                "provisional_analysis_manifest_sha256": "analysis-hash",
                "methods": list(statistics.RUNTIME_PRIMITIVE_IDS),
                "arms": list(statistics.FOOD_ARMS),
                "deterministic_seed_rule": dict(analysis.RUNTIME_RECIPE_SEED_RULE),
            }
        },
        "tables": {"runtime_rows": runtime_rows},
        "raw": {
            "recipe_bindings": _test_recipe_bindings(
                runtime_rows, analysis.RUNTIME_RECIPE_SEED_RULE
            )
        },
        "protocol_sha256": "p",
        "code_identity_sha256": "c",
        "manifest_sha256": "runtime-hash",
    }
    monkeypatch.setattr(
        analysis,
        "verify_completed_artifact",
        lambda root, **_kwargs: runtime if str(root).endswith("runtime") else development,
    )
    monkeypatch.setattr(analysis, "_validate_food_completed_artifact", lambda _value: None)
    monkeypatch.setattr(
        statistics,
        "food_panel_metrics",
        lambda *_args, **_kwargs: [{"candidate_id": "LP-FULL"}],
    )
    monkeypatch.setattr(statistics, "rank_auc_rows", lambda _rows: [])
    monkeypatch.setattr(statistics, "selection_rate_summary", lambda _rows: [])
    monkeypatch.setattr(statistics, "rank_auc_summary", lambda _rows: [])
    monkeypatch.setattr(analysis, "_read_object", lambda _path: {"factorial": {"effects": {}}})
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        reporting,
        "write_report_bundle",
        lambda output, **kwargs: observed.update({"lock": kwargs["development_lock"]}) or Path(output),
    )
    analysis.finalize_development(
        tmp_path / "provisional",
        tmp_path / "runtime",
        tmp_path / "final",
        n_resamples=20,
        verify_current_identity=False,
    )
    lock = observed["lock"]
    assert lock["selected_chain"] == "M1-SW"  # type: ignore[index]
    assert lock["chain_kind"] == "pure_oi"  # type: ignore[index]
    assert lock["resource_status"] == "pass"  # type: ignore[index]
    assert lock["status"] == "locked_for_future_confirmation"  # type: ignore[index]
    assert lock["selected_chain_recipe_bindings"]["primitive_component_ids"] == [  # type: ignore[index]
        "M1-SW"
    ]
