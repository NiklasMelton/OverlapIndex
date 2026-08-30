from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.fused3_confirmation_v2 import recipes, runner


def _identity() -> dict[str, object]:
    return {
        "panel_index": 0,
        "panel_id": "torchvision_cifar10__dinov2-small__r0__b32",
        "dataset_id": "torchvision_cifar10",
        "backbone": "dinov2-small",
        "replicate": 0,
        "replicate_seed": recipes.REPLICATE_SEEDS[0],
        "budget": 32,
        "execution_order": list(recipes.SELECTORS),
    }


def _panel() -> dict[str, object]:
    labels = np.repeat(np.arange(2), 32)
    test_labels = np.repeat(np.arange(2), 20)
    values = np.arange(len(labels) * 4, dtype=np.float32).reshape(len(labels), 4)
    test = np.arange(len(test_labels) * 4, dtype=np.float32).reshape(
        len(test_labels), 4
    )
    return {
        "training_values": values,
        "training_labels": labels,
        "evaluation_values": test,
        "evaluation_labels": test_labels,
        "training_cohort_sha256": "1" * 64,
        "evaluation_cohort_sha256": "2" * 64,
    }


def _selector(candidate: str, values: object, labels: object, *, seed: int):
    del values, labels, seed
    return {
        "candidate_id": candidate,
        "score": 0.25 if candidate == "FUSED3" else 0.5,
        "candidate_recipe_sha256": recipes.payload_sha256(
            recipes.candidate_recipe(candidate)
        ),
        "folds": [],
    }


def _reference(
    head: str,
    training_values: object,
    training_labels: object,
    evaluation_values: object,
    evaluation_labels: object,
    *,
    seed: int,
):
    del training_labels, evaluation_labels, seed
    recipe = recipes.head_recipe(head)
    return {
        "head": head,
        "test_accuracy": 0.75,
        "training_row_count": len(training_values),
        "evaluation_row_count": len(evaluation_values),
        "recipe_sha256": recipes.payload_sha256(recipe),
    }


def test_frozen_panel_grid_and_counterbalance() -> None:
    identities = runner.panel_identities()
    assert len(identities) == 500
    assert len({row["panel_id"] for row in identities}) == 500
    assert identities[0]["dataset_id"] == recipes.DATASET_IDS[0]
    assert identities[-1]["dataset_id"] == recipes.DATASET_IDS[-1]
    counts = {candidate: [0, 0] for candidate in recipes.SELECTORS}
    for row in identities:
        for position, candidate in enumerate(row["execution_order"]):
            counts[candidate][position] += 1
    assert all(value == [250, 250] for value in counts.values())


def test_execute_panel_uses_identical_cohort_for_selectors_and_heads() -> None:
    result = runner.execute_panel(
        _identity(),
        _panel(),
        selector_executor=_selector,
        reference_executor=_reference,
    )
    assert {row["candidate_id"] for row in result["selector_rows"]} == set(
        recipes.SELECTORS
    )
    assert {row["head"] for row in result["reference_rows"]} == set(recipes.HEADS)
    assert {row["training_cohort_sha256"] for row in result["selector_rows"]} == {
        "1" * 64
    }
    assert {row["training_cohort_sha256"] for row in result["reference_rows"]} == {
        "1" * 64
    }
    assert all(row["training_row_count"] == 64 for row in result["reference_rows"])
    assert all(row["evaluation_row_count"] == 40 for row in result["reference_rows"])


def test_structural_smoke_redacts_every_numeric_outcome() -> None:
    result = runner.execute_panel(
        _identity(),
        _panel(),
        selector_executor=_selector,
        reference_executor=_reference,
    )
    redacted = runner.redact_smoke_panel(result)
    text = json.dumps(redacted, sort_keys=True)
    assert '"score":' not in text
    assert '"test_accuracy":' not in text
    assert all(row["outcomes_redacted"] for row in redacted["selector_rows"])
    assert all(row["score_structure"]["finite"] for row in redacted["selector_rows"])
    assert all(row["outcomes_redacted"] for row in redacted["reference_rows"])


def test_determinism_signatures_exclude_all_numeric_outcomes() -> None:
    result = runner.execute_panel(
        _identity(),
        _panel(),
        selector_executor=_selector,
        reference_executor=_reference,
    )
    selector = result["selector_rows"][0]
    changed_selector = {
        **selector,
        "score": float(selector["score"]) + 0.125,
        "folds": [{"fold": 0, "score": 0.999}],
        "total_wall_seconds": 999.0,
    }
    assert runner._structural_signature(
        selector, row_kind="selector"
    ) == runner._structural_signature(changed_selector, row_kind="selector")
    assert runner._repeat_exact(selector, changed_selector) is False

    reference = result["reference_rows"][0]
    changed_reference = {
        **reference,
        "test_accuracy": float(reference["test_accuracy"]) - 0.25,
        "total_cpu_seconds": 999.0,
    }
    assert runner._structural_signature(
        reference, row_kind="reference"
    ) == runner._structural_signature(changed_reference, row_kind="reference")
    assert runner._repeat_exact(reference, changed_reference) is False


def test_panel_rejects_misalignment_and_nonbalanced_budget() -> None:
    panel = _panel()
    panel["training_labels"] = np.asarray(panel["training_labels"])[:-1]
    with pytest.raises(RuntimeError, match="misaligned"):
        runner.execute_panel(
            _identity(),
            panel,
            selector_executor=_selector,
            reference_executor=_reference,
        )
    panel = _panel()
    panel["training_values"] = np.asarray(panel["training_values"])[:-2]
    panel["training_labels"] = np.asarray(panel["training_labels"])[:-2]
    with pytest.raises(RuntimeError, match="class-balanced budget"):
        runner.execute_panel(
            _identity(),
            panel,
            selector_executor=_selector,
            reference_executor=_reference,
        )


def test_protocol_and_sidecar_are_exactly_frozen() -> None:
    assert runner.PROTOCOL_SIDECAR.is_file()
    observed, protocol = runner.validate_protocol(require_frozen=True)
    assert runner.PROTOCOL_SIDECAR.read_text(encoding="utf-8") == observed + "\n"
    assert len(observed) == 64
    assert protocol["status"] == "frozen_before_confirmation_outcomes"


def test_output_refuses_cross_identity_resume(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "running_manifest.json").write_text(
        json.dumps({"run_identity_sha256": "a" * 64}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        runner._prepare_output(output, run_identity="b" * 64, resume=True)


def test_smoke_checkpoint_policy_never_persists_numeric_outcomes() -> None:
    result = runner.execute_panel(
        _identity(),
        _panel(),
        selector_executor=_selector,
        reference_executor=_reference,
    )
    redacted = runner.redact_smoke_panel(result)
    serialized = runner.canonical_json(redacted)
    assert '"score":' not in serialized
    assert '"test_accuracy":' not in serialized
    # Smoke checkpoints are intentionally forbidden: run_confirmation keeps
    # the numerical result only in memory and writes this descriptor surface.
    assert redacted["artifact_status"] == "completed_structural_smoke_panel"


def test_run_smoke_persists_no_numeric_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = {"status": "frozen_before_confirmation_outcomes"}
    monkeypatch.setattr(
        runner, "validate_protocol", lambda require_frozen=True: ("a" * 64, protocol)
    )
    monkeypatch.setattr(runner, "_thread_environment", lambda: dict(runner.THREAD_ENVIRONMENT))
    monkeypatch.setattr(
        runner, "_score_environment", lambda: dict(runner.EXPECTED_SCORE_ENVIRONMENT)
    )
    monkeypatch.setattr(
        runner,
        "source_identity",
        lambda: {
            "code_identity_sha256": "b" * 64,
            "source_hashes": {"source.py": "c" * 64},
            "repository": {"branch": "test"},
        },
    )
    from experiments.fused3_confirmation_v2 import lineage

    monkeypatch.setattr(lineage, "validate_candidate_authority_file", lambda _path: {})
    monkeypatch.setattr(lineage, "build_lineage_from_files", lambda **_kwargs: {"status": "frozen_v2_design"})
    monkeypatch.setattr(lineage, "validate_lock_envelope", lambda value, **_kwargs: value)
    monkeypatch.setattr(lineage, "authority_sha256", lambda _path: "d" * 64)
    registry = {"registry_id": "fused3_confirmation_v2_inputs"}
    monkeypatch.setattr(
        runner,
        "_load_registry",
        lambda *_args: (registry, lambda *_a, **_k: _panel()),
    )
    panel_result = runner.execute_panel(
        _identity(), _panel(), selector_executor=_selector, reference_executor=_reference
    )
    monkeypatch.setattr(runner, "execute_panel", lambda *_args, **_kwargs: panel_result)
    registry_path = tmp_path / "registry.json"
    lock_path = tmp_path / "lock.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    lock_path.write_text(
        json.dumps(
            {
                "lineage": {"v1_confirmation_registry_sha256": "e" * 64}
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "smoke"
    runner.run_confirmation(
        registry_path=registry_path,
        lineage_lock_path=lock_path,
        output=output,
        smoke=True,
    )
    assert not list((output / "checkpoints").glob("*.json"))
    for path in output.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert '"score":' not in text
        assert '"test_accuracy":' not in text
        assert '"folds":' not in text
    raw = json.loads((output / "raw_results.json").read_text(encoding="utf-8"))
    for descriptor in raw["determinism_verification"].values():
        assert descriptor["exact"] is True
        assert descriptor["runtime_fields_excluded"] is True
        assert descriptor["outcome_fields_excluded"] is True
