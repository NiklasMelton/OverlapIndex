from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.fused3_confirmation_v2 import analysis, lineage, recipes, statistics


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.fixture(autouse=True)
def _sealed_local_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    identity_root = tmp_path / "sealed-identity"
    identity_root.mkdir()
    protocol_path = identity_root / "protocol.json"
    protocol_path.write_text(
        analysis.canonical_json(
            {
                "schema_version": 1,
                "study": statistics.STUDY,
                "status": "frozen_before_confirmation_outcomes",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    sidecar_path = identity_root / "protocol.sha256"
    sidecar_path.write_text(protocol_sha + "\n", encoding="utf-8")
    authority_path = identity_root / "candidate_authority.json"
    authority_path.write_text(
        analysis.canonical_json(
            {
                "status": "locked_for_untouched_confirmation",
                "selected_candidate": "FUSED3",
                "selected_candidate_recipe_sha256": recipes.payload_sha256(
                    recipes.candidate_recipe("FUSED3")
                ),
                "runner_up_reselection_allowed": False,
                "fused3_full": {
                    "decision_status": "pass_full_retrospective",
                    "failed_required_gate_count": 0,
                    "resource_status": "passed_all_per_call_and_full_panel_gates_in_all_three_food_arms",
                },
                "confirmation_constraints": {
                    "candidate_ids": list(statistics.METHODS),
                    "promotable_candidate_ids": ["FUSED3"],
                    "no_reselection": True,
                    "no_recipe_change": True,
                    "no_gate_change": True,
                    "runtime_inheritance": "fused3_full",
                    "confirmation_runtime": "descriptive_only",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(analysis, "PROTOCOL_PATH", protocol_path)
    monkeypatch.setattr(analysis, "PROTOCOL_SIDECAR", sidecar_path)
    monkeypatch.setattr(analysis, "CANDIDATE_AUTHORITY_PATH", authority_path)
    monkeypatch.setattr(
        analysis.runner,
        "_source_paths",
        lambda: (
            Path(analysis.__file__).resolve(),
            Path(statistics.__file__).resolve(),
            Path(recipes.__file__).resolve(),
            Path(lineage.__file__).resolve(),
            protocol_path.resolve(),
            authority_path.resolve(),
        ),
    )

    def fake_build_lineage_from_files(**kwargs):
        return lineage.make_lineage(
            protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            code_identity_sha256=str(kwargs["code_identity_sha256"]),
            v1_protocol_sha256="5" * 64,
            v1_final_lock_sha256="6" * 64,
            v1_confirmation_registry_sha256="7" * 64,
        )

    monkeypatch.setattr(lineage, "build_lineage_from_files", fake_build_lineage_from_files)

    def fake_registry_loader(
        path: Path,
        *,
        expected_protocol_sha: str,
        expected_lock_sha: str,
        expected_parent_registry_sha: str,
    ) -> dict[str, object]:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        assert value["protocol_sha256"] == expected_protocol_sha
        assert value["lineage_lock_sha256"] == expected_lock_sha
        assert expected_parent_registry_sha == "7" * 64
        return value

    monkeypatch.setattr(
        analysis.datasets, "load_and_validate_registry", fake_registry_loader
    )


def _rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selectors: list[dict[str, object]] = []
    references: list[dict[str, object]] = []
    panel_index = 0
    for dataset in statistics.DATASET_IDS:
        class_count = analysis.CLASS_COUNTS[dataset]
        for backbone_index, backbone in enumerate(statistics.BACKBONES):
            for replicate, replicate_seed in enumerate(statistics.REPLICATE_SEEDS):
                for budget in statistics.BUDGETS:
                    order = recipes.method_order(panel_index)
                    training_hash = _digest(
                        f"training:{dataset}:{replicate_seed}:{budget}"
                    )
                    evaluation_hash = _digest(f"evaluation:{dataset}")
                    for candidate in statistics.METHODS:
                        score = (
                            float(backbone_index)
                            if candidate == "FUSED3"
                            else float(-backbone_index)
                        )
                        total_rows = class_count * budget
                        folds = []
                        for fold in range(5):
                            holdout = total_rows // 5 + (fold < total_rows % 5)
                            descriptor: dict[str, object] = {
                                "fold": fold,
                                "seed": replicate_seed + fold
                                if candidate == "FUSED3"
                                else replicate_seed,
                                "train_size": total_rows - holdout,
                                "holdout_size": holdout,
                                "score": 0.5,
                                "fit_wall_seconds": 0.1,
                                "fit_cpu_seconds": 0.1,
                                "candidate_config_sha256": _digest(
                                    f"config:{candidate}:{replicate_seed}:{fold}"
                                ),
                            }
                            if candidate == "FUSED3":
                                descriptor.update(
                                    {
                                        "score_fixed_wall_seconds": 0.1,
                                        "score_fixed_cpu_seconds": 0.1,
                                        "numerical_state_sha256": _digest(
                                            f"state:{dataset}:{backbone}:{replicate_seed}:{budget}:{fold}"
                                        ),
                                        "state_unchanged_after_score_fixed": True,
                                    }
                                )
                            else:
                                descriptor.update(
                                    {
                                        "split_seed": replicate_seed,
                                        "model_random_state": replicate_seed,
                                        "predict_wall_seconds": 0.1,
                                        "predict_cpu_seconds": 0.1,
                                    }
                                )
                            folds.append(descriptor)
                        selectors.append(
                            {
                                "panel_id": f"{dataset}__{backbone}__r{replicate}__b{budget}",
                                "dataset_id": dataset,
                                "backbone": backbone,
                                "replicate": replicate,
                                "replicate_seed": replicate_seed,
                                "budget": budget,
                                "candidate_id": candidate,
                                "score": score / 20.0 + 0.5,
                                "total_wall_seconds": 1.0 if candidate == "FUSED3" else 2.0,
                                "total_cpu_seconds": 0.9 if candidate == "FUSED3" else 1.8,
                                "fit_wall_seconds": 0.5,
                                "fit_cpu_seconds": 0.4,
                                "score_fixed_wall_seconds": 0.25,
                                "score_fixed_cpu_seconds": 0.2,
                                "execution_position": order.index(candidate),
                                "execution_order": list(order),
                                "warmup_excluded": True,
                                "status": "ok",
                                "error": None,
                                "candidate_recipe_sha256": recipes.payload_sha256(
                                    recipes.candidate_recipe(candidate)
                                ),
                                "training_cohort_sha256": training_hash,
                                "evaluation_cohort_sha256": evaluation_hash,
                                "folds": folds,
                            }
                        )
                    for head in statistics.HEADS:
                        step = 0.0005 if head == "linear" else 0.01
                        references.append(
                            {
                                "panel_id": f"{dataset}__{backbone}__r{replicate}__b{budget}",
                                "dataset_id": dataset,
                                "backbone": backbone,
                                "replicate": replicate,
                                "replicate_seed": replicate_seed,
                                "budget": budget,
                                "head": head,
                                "test_accuracy": 0.5 + step * backbone_index,
                                "total_wall_seconds": 3.0,
                                "total_cpu_seconds": 2.8,
                                "training_row_count": class_count * budget,
                                "evaluation_row_count": class_count * 20,
                                "status": "ok",
                                "error": None,
                                "recipe_sha256": recipes.payload_sha256(
                                    recipes.head_recipe(head)
                                ),
                                "training_cohort_sha256": training_hash,
                                "evaluation_cohort_sha256": evaluation_hash,
                            }
                        )
                    panel_index += 1
    return selectors, references


def _table_payload(rows: list[dict[str, object]]) -> bytes:
    return "".join(analysis.canonical_json(row) + "\n" for row in rows).encode()


def _write_artifact(
    root: Path,
    selectors: list[dict[str, object]] | None = None,
    references: list[dict[str, object]] | None = None,
    *,
    manifest_mutation: tuple[str, object] | None = None,
) -> Path:
    root.mkdir()
    selector_rows, reference_rows = (
        _rows() if selectors is None or references is None else (selectors, references)
    )
    selector_payload = _table_payload(selector_rows)
    reference_payload = _table_payload(reference_rows)
    (root / "selector_rows.jsonl").write_bytes(selector_payload)
    (root / "reference_rows.jsonl").write_bytes(reference_payload)
    table_manifest = {
        "selector_rows.jsonl": {
            "row_count": len(selector_rows),
            "sha256": hashlib.sha256(selector_payload).hexdigest(),
        },
        "reference_rows.jsonl": {
            "row_count": len(reference_rows),
            "sha256": hashlib.sha256(reference_payload).hexdigest(),
        },
    }
    first_panel_id = (
        f"{statistics.DATASET_IDS[0]}__{statistics.BACKBONES[0]}__r0__b{statistics.BUDGETS[0]}"
    )
    determinism = {
        identity: {
            "exact": True,
            "runtime_fields_excluded": True,
            "outcome_fields_excluded": True,
            "first_signature_sha256": _digest(f"determinism:{identity}"),
            "second_signature_sha256": _digest(f"determinism:{identity}"),
            "panel_id": first_panel_id,
        }
        for identity in (
            *statistics.METHODS,
            *(f"HEAD:{head}" for head in statistics.HEADS),
        )
    }
    protocol_sha = hashlib.sha256(analysis.PROTOCOL_PATH.read_bytes()).hexdigest()
    authority_sha = hashlib.sha256(
        analysis.CANDIDATE_AUTHORITY_PATH.read_bytes()
    ).hexdigest()
    sealed_lineage = lineage.make_lineage(
        protocol_sha256=protocol_sha,
        code_identity_sha256="2" * 64,
        v1_protocol_sha256="5" * 64,
        v1_final_lock_sha256="6" * 64,
        v1_confirmation_registry_sha256="7" * 64,
    )
    lock = lineage.make_lock_envelope(
        lineage=sealed_lineage,
        candidate_authority_sha256=authority_sha,
    )
    lock_payload = (analysis.canonical_json(lock) + "\n").encode()
    (root / analysis.LINEAGE_LOCK_FILENAME).write_bytes(lock_payload)
    lock_hash = hashlib.sha256(lock_payload).hexdigest()
    registry = {
        "registry_id": "fused3_confirmation_v2_inputs",
        "protocol_sha256": protocol_sha,
        "lineage_lock_sha256": lock_hash,
    }
    registry_payload = (analysis.canonical_json(registry) + "\n").encode()
    (root / analysis.AUDITED_REGISTRY_FILENAME).write_bytes(registry_payload)
    registry_hash = hashlib.sha256(registry_payload).hexdigest()
    source_paths = (
        Path(analysis.__file__).resolve(),
        Path(statistics.__file__).resolve(),
        Path(recipes.__file__).resolve(),
        Path(lineage.__file__).resolve(),
        analysis.PROTOCOL_PATH.resolve(),
        analysis.CANDIDATE_AUTHORITY_PATH.resolve(),
    )
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths
    }
    code_identity_sha = recipes.payload_sha256(source_hashes)
    score_environment = dict(analysis.EXPECTED_SCORE_ENVIRONMENT)
    thread_environment = dict(analysis.EXPECTED_THREAD_ENVIRONMENT)
    run_identity_sha = recipes.payload_sha256(
        {
            "study": statistics.STUDY,
            "mode": "full",
            "protocol_sha256": protocol_sha,
            "code_identity_sha256": code_identity_sha,
            "lineage_lock_sha256": lock_hash,
            "audited_registry_sha256": registry_hash,
            "environment": score_environment,
            "thread_environment": thread_environment,
        }
    )
    # Seal the same code identity into the input lock after source identity is known.
    sealed_lineage["code_identity_sha256"] = code_identity_sha
    lock = lineage.make_lock_envelope(
        lineage=sealed_lineage,
        candidate_authority_sha256=authority_sha,
    )
    lock_payload = (analysis.canonical_json(lock) + "\n").encode()
    (root / analysis.LINEAGE_LOCK_FILENAME).write_bytes(lock_payload)
    lock_hash = hashlib.sha256(lock_payload).hexdigest()
    registry["lineage_lock_sha256"] = lock_hash
    registry_payload = (analysis.canonical_json(registry) + "\n").encode()
    (root / analysis.AUDITED_REGISTRY_FILENAME).write_bytes(registry_payload)
    registry_hash = hashlib.sha256(registry_payload).hexdigest()
    run_identity_sha = recipes.payload_sha256(
        {
            "study": statistics.STUDY,
            "mode": "full",
            "protocol_sha256": protocol_sha,
            "code_identity_sha256": code_identity_sha,
            "lineage_lock_sha256": lock_hash,
            "audited_registry_sha256": registry_hash,
            "environment": score_environment,
            "thread_environment": thread_environment,
        }
    )
    raw = {
        "schema_version": 1,
        "study": statistics.STUDY,
        "artifact_status": "completed",
        "mode": "full",
        "protocol_sha256": protocol_sha,
        "code_identity_sha256": code_identity_sha,
        "run_identity_sha256": run_identity_sha,
        "audited_registry_sha256": registry_hash,
        "lineage_lock_sha256": lock_hash,
        "environment": {**score_environment, "thread_environment": thread_environment},
        "thread_environment": thread_environment,
        "repository_provenance": {"source": "fixture"},
        "source_hashes": source_hashes,
        "audited_registry": {
            "path": analysis.AUDITED_REGISTRY_FILENAME,
            "sha256": registry_hash,
            "registry_id": "fused3_confirmation_v2_inputs",
        },
        "grid": analysis._expected_grid(),
        "table_manifest": table_manifest,
        "checkpoint_count": 500,
        "determinism_verification": determinism,
        "runtime_gate": {
            "status": "inherited_pass",
            "source": "hash_bound_fused3_full_food_resource_result",
            "confirmation_clocks": "descriptive_only",
        },
        "protocol": json.loads(analysis.PROTOCOL_PATH.read_text(encoding="utf-8")),
    }
    raw_payload = (analysis.canonical_json(raw) + "\n").encode()
    (root / "raw_results.json").write_bytes(raw_payload)
    manifest = copy.deepcopy(raw)
    manifest["raw_results_sha256"] = hashlib.sha256(raw_payload).hexdigest()
    if manifest_mutation is not None:
        manifest[manifest_mutation[0]] = manifest_mutation[1]
    (root / "manifest.json").write_text(
        analysis.canonical_json(manifest) + "\n", encoding="utf-8"
    )
    return root


def _rewrite_raw_and_manifest(root: Path, raw: dict[str, object]) -> None:
    raw_payload = (analysis.canonical_json(raw) + "\n").encode()
    (root / "raw_results.json").write_bytes(raw_payload)
    manifest = copy.deepcopy(raw)
    manifest["raw_results_sha256"] = hashlib.sha256(raw_payload).hexdigest()
    (root / "manifest.json").write_text(
        analysis.canonical_json(manifest) + "\n", encoding="utf-8"
    )


def test_verify_completed_artifact_checks_hashes_grid_and_rows(tmp_path: Path) -> None:
    source = _write_artifact(tmp_path / "source")
    loaded = analysis.verify_completed_artifact(source)
    assert len(loaded["selector_rows"]) == 1000
    assert len(loaded["reference_rows"]) == 2000
    assert loaded["raw"]["lineage_lock_sha256"] == hashlib.sha256(
        (source / analysis.LINEAGE_LOCK_FILENAME).read_bytes()
    ).hexdigest()

    tampered = tmp_path / "tampered"
    _write_artifact(tampered)
    with (tampered / "selector_rows.jsonl").open("ab") as stream:
        stream.write(b"{}\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        analysis.verify_completed_artifact(tampered)


def test_manifest_only_lineage_tampering_is_rejected(tmp_path: Path) -> None:
    source = _write_artifact(
        tmp_path / "source",
        manifest_mutation=("code_identity_sha256", "f" * 64),
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        analysis.verify_completed_artifact(source)


def test_local_lock_registry_determinism_and_cohort_pairing_fail_closed(
    tmp_path: Path,
) -> None:
    registry_tamper = _write_artifact(tmp_path / "registry-tamper")
    with (registry_tamper / analysis.AUDITED_REGISTRY_FILENAME).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="registry SHA-256 mismatch"):
        analysis.verify_completed_artifact(registry_tamper)

    lock_tamper = _write_artifact(tmp_path / "lock-tamper")
    with (lock_tamper / analysis.LINEAGE_LOCK_FILENAME).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="lineage lock SHA-256 mismatch"):
        analysis.verify_completed_artifact(lock_tamper)

    determinism_tamper = _write_artifact(tmp_path / "determinism-tamper")
    raw = json.loads((determinism_tamper / "raw_results.json").read_text())
    raw["determinism_verification"]["FUSED3"]["panel_id"] = "later-panel"
    _rewrite_raw_and_manifest(determinism_tamper, raw)
    with pytest.raises(ValueError, match="first frozen panel"):
        analysis.verify_completed_artifact(determinism_tamper)

    selectors, references = _rows()
    selectors[0]["training_cohort_sha256"] = "f" * 64
    cohort_tamper = _write_artifact(
        tmp_path / "cohort-tamper", selectors, references
    )
    with pytest.raises(ValueError, match="cohort hashes differ"):
        analysis.verify_completed_artifact(cohort_tamper)


def test_rehashed_incomplete_and_bad_recipe_artifacts_fail_closed(tmp_path: Path) -> None:
    selectors, references = _rows()
    incomplete = _write_artifact(
        tmp_path / "incomplete", selectors[:-1], references
    )
    with pytest.raises(ValueError, match="exact confirmation grid"):
        analysis.verify_completed_artifact(incomplete)

    selectors, references = _rows()
    selectors[0]["candidate_recipe_sha256"] = "f" * 64
    wrong_recipe = _write_artifact(tmp_path / "wrong-recipe", selectors, references)
    with pytest.raises(ValueError, match="candidate recipe hash mismatch"):
        analysis.verify_completed_artifact(wrong_recipe)


def test_replicate_pair_schedule_and_reference_counts_are_independently_checked(
    tmp_path: Path,
) -> None:
    selectors, references = _rows()
    selectors[0]["replicate_seed"] = statistics.REPLICATE_SEEDS[1]
    bad_pair = _write_artifact(tmp_path / "bad-pair", selectors, references)
    with pytest.raises(ValueError, match="frozen pairing"):
        analysis.verify_completed_artifact(bad_pair)

    selectors, references = _rows()
    references[0]["evaluation_row_count"] = 1
    bad_count = _write_artifact(tmp_path / "bad-count", selectors, references)
    with pytest.raises(ValueError, match="20 rows per class"):
        analysis.verify_completed_artifact(bad_count)


def test_analysis_bundle_is_atomic_nonoverwriting_and_byte_stable(tmp_path: Path) -> None:
    source = _write_artifact(tmp_path / "source")
    first = tmp_path / "analysis-a"
    second = tmp_path / "analysis-b"
    summary = analysis.write_analysis_bundle(source, first)
    analysis.write_analysis_bundle(source, second)
    expected = {
        "summary.json",
        "decision.json",
        "panel_metrics.csv",
        "regret_contrasts.csv",
        "rank_auc.csv",
        "runtime_descriptive.csv",
        "report.md",
        "analysis_manifest.json",
    }
    assert {path.name for path in first.iterdir()} == expected
    assert {name: (first / name).read_bytes() for name in expected} == {
        name: (second / name).read_bytes() for name in expected
    }
    assert summary["decision"]["status"] == "pass_confirmed"
    assert summary["decision"]["runner_up_allowed"] is False
    assert summary["runtime"]["promotion_veto"] is False
    report = (first / "report.md").read_text(encoding="utf-8")
    assert "same-distribution five-dataset panel" in report
    assert "No candidate reselection" in report
    assert "descriptive only" in report
    manifest = json.loads((first / "analysis_manifest.json").read_text())
    assert set(manifest["files"]) == expected - {"analysis_manifest.json"}
    for name, descriptor in manifest["files"].items():
        assert descriptor["sha256"] == hashlib.sha256((first / name).read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        analysis.write_analysis_bundle(source, first)


def test_report_rejects_locked_candidate_without_substitution() -> None:
    selectors, references = _rows()
    for row in references:
        if row["head"] == "quadratic":
            backbone_index = statistics.BACKBONES.index(str(row["backbone"]))
            row["test_accuracy"] = 0.5 - 0.01 * backbone_index
    summary = statistics.summarize(selectors, references)
    summary["artifact_status"] = "completed"
    summary["lineage"] = {"lineage_lock_sha256": "a" * 64}
    report = analysis.render_report(summary)
    assert summary["decision"]["status"] == "fail_locked_candidate"
    assert "no runner-up is substituted" in report
