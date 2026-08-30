from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.fused3_confirmation_v2 import recipes, resource_preflight, runner


PROTOCOL_SHA = "a" * 64
CODE_SHA = "b" * 64
SOURCE_SHA = "c" * 64


def _protocol() -> dict[str, object]:
    return {"status": "frozen_before_confirmation_outcomes"}


def _sources() -> dict[str, object]:
    return {
        "code_identity_sha256": CODE_SHA,
        "source_hashes": {"source.py": SOURCE_SHA},
        "repository": {"branch": "codex/test", "experiment_commit": "e" * 40},
    }


def _run_identity(*, registry_sha: str, lock_sha: str) -> str:
    return runner.payload_sha256(
        {
            "study": resource_preflight.STUDY,
            "mode": "smoke",
            "protocol_sha256": PROTOCOL_SHA,
            "code_identity_sha256": CODE_SHA,
            "lineage_lock_sha256": lock_sha,
            "audited_registry_sha256": registry_sha,
            "environment": dict(runner.EXPECTED_SCORE_ENVIRONMENT),
            "thread_environment": dict(runner.THREAD_ENVIRONMENT),
        }
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(runner.canonical_json(value) + "\n", encoding="utf-8")


def _full_panel() -> dict[str, object]:
    identity = runner.panel_identities()[0]
    common = {
        key: identity[key]
        for key in (
            "panel_id",
            "dataset_id",
            "backbone",
            "replicate",
            "replicate_seed",
            "budget",
        )
    }
    selectors = []
    for position, candidate_id in enumerate(identity["execution_order"]):
        selectors.append(
            {
                **common,
                "candidate_id": candidate_id,
                "execution_position": position,
                "execution_order": list(identity["execution_order"]),
                "warmup_excluded": True,
                "candidate_recipe_sha256": recipes.payload_sha256(
                    recipes.candidate_recipe(candidate_id)
                ),
                "training_cohort_sha256": "1" * 64,
                "evaluation_cohort_sha256": "2" * 64,
                "status": "ok",
                "error": None,
                "score": 0.25 + position * 0.25,
            }
        )
    references = []
    for index, head in enumerate(recipes.HEADS):
        references.append(
            {
                **common,
                "head": head,
                "training_row_count": 320,
                "evaluation_row_count": 200,
                "recipe_sha256": recipes.payload_sha256(recipes.head_recipe(head)),
                "training_cohort_sha256": "1" * 64,
                "evaluation_cohort_sha256": "2" * 64,
                "status": "ok",
                "error": None,
                "test_accuracy": 0.5 + index * 0.05,
            }
        )
    return {
        "schema_version": 1,
        "artifact_status": "completed_panel",
        "panel_identity": identity,
        "selector_rows": selectors,
        "reference_rows": references,
    }


def _write_smoke(
    root: Path,
    *,
    registry_path: Path,
    lock_path: Path,
    extra_raw: dict[str, object] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_sha = runner.sha256_path(registry_path)
    lock_sha = runner.sha256_path(lock_path)
    (root / "audited_registry.json").write_bytes(registry_path.read_bytes())
    (root / "lineage_lock.json").write_bytes(lock_path.read_bytes())
    panel = runner.redact_smoke_panel(_full_panel())
    determinism: dict[str, object] = {}
    for row in panel["selector_rows"]:
        identity = str(row["candidate_id"])
        signature = runner._structural_signature(row, row_kind="selector")
        determinism[identity] = {
            "first_signature_sha256": signature,
            "second_signature_sha256": signature,
            "exact": True,
            "runtime_fields_excluded": True,
            "outcome_fields_excluded": True,
            "panel_id": panel["panel_identity"]["panel_id"],
        }
    for row in panel["reference_rows"]:
        identity = f"HEAD:{row['head']}"
        signature = runner._structural_signature(row, row_kind="reference")
        determinism[identity] = {
            "first_signature_sha256": signature,
            "second_signature_sha256": signature,
            "exact": True,
            "runtime_fields_excluded": True,
            "outcome_fields_excluded": True,
            "panel_id": panel["panel_identity"]["panel_id"],
        }
    shared = {
        "schema_version": 1,
        "study": resource_preflight.STUDY,
        "artifact_status": "completed_structural_smoke",
        "mode": "smoke",
        "protocol_sha256": PROTOCOL_SHA,
        "code_identity_sha256": CODE_SHA,
        "lineage_lock_sha256": lock_sha,
        "audited_registry_sha256": registry_sha,
        "audited_registry": {
            "path": "audited_registry.json",
            "sha256": registry_sha,
            "registry_id": registry.get("registry_id"),
        },
        "source_hashes": _sources()["source_hashes"],
        "repository_provenance": _sources()["repository"],
        "thread_environment": dict(runner.THREAD_ENVIRONMENT),
        "environment": dict(runner.EXPECTED_SCORE_ENVIRONMENT),
        "protocol": _protocol(),
        "outcomes_redacted": True,
        "run_identity_sha256": _run_identity(
            registry_sha=registry_sha, lock_sha=lock_sha
        ),
    }
    raw = {
        **shared,
        "ranking_available": False,
        "panel": panel,
        "determinism_verification": determinism,
        **(extra_raw or {}),
    }
    _write_json(root / "raw_results.json", raw)
    manifest = {
        **shared,
        "raw_results_sha256": runner.sha256_path(root / "raw_results.json"),
    }
    _write_json(root / "manifest.json", manifest)


@pytest.fixture
def frozen_surfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path]:
    monkeypatch.setattr(
        runner,
        "validate_protocol",
        lambda require_frozen=True: (PROTOCOL_SHA, _protocol()),
    )
    monkeypatch.setattr(runner, "source_identity", _sources)
    registry = tmp_path / "registry.json"
    lock = tmp_path / "lock.json"
    _write_json(
        registry,
        {
            "registry_id": "fused3_confirmation_v2_inputs",
            "status": "audited_complete",
            "datasets": [],
        },
    )
    _write_json(lock, {"status": "frozen_v2_design"})
    smoke = tmp_path / "smoke"
    _write_smoke(smoke, registry_path=registry, lock_path=lock)
    return {"registry": registry, "lock": lock, "smoke": smoke}


def _resource_values() -> dict[str, object]:
    return {
        "scope": "fresh_child_full_structural_smoke_including_imports_validation_excluded_warmup_and_measured_call",
        "wall_seconds": 3.5,
        "user_cpu_seconds": 2.5,
        "system_cpu_seconds": 0.5,
        "peak_rss_bytes": 123456789,
        "peak_rss_recipe": "macos_usr_bin_time_l_maximum_resident_set_size_bytes",
        "warmup_included": True,
        "outcomes_persisted": False,
        "descriptive_only": True,
        "promotion_veto": False,
    }


def _rewrite_raw(smoke: Path, mutate: object) -> None:
    raw_path = smoke / "raw_results.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    mutate(raw)
    _write_json(raw_path, raw)
    manifest_path = smoke / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["raw_results_sha256"] = runner.sha256_path(raw_path)
    _write_json(manifest_path, manifest)


def test_validator_accepts_the_canonical_runner_smoke_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "validate_protocol",
        lambda require_frozen=True: (PROTOCOL_SHA, _protocol()),
    )
    monkeypatch.setattr(runner, "source_identity", _sources)
    monkeypatch.setattr(
        runner, "_thread_environment", lambda: dict(runner.THREAD_ENVIRONMENT)
    )
    monkeypatch.setattr(
        runner, "_score_environment", lambda: dict(runner.EXPECTED_SCORE_ENVIRONMENT)
    )
    from experiments.fused3_confirmation_v2 import lineage

    monkeypatch.setattr(lineage, "validate_candidate_authority_file", lambda _path: {})
    monkeypatch.setattr(
        lineage,
        "build_lineage_from_files",
        lambda **_kwargs: {"status": "frozen_v2_design"},
    )
    monkeypatch.setattr(lineage, "validate_lock_envelope", lambda value, **_kwargs: value)
    monkeypatch.setattr(lineage, "authority_sha256", lambda _path: "f" * 64)
    registry = {
        "registry_id": "fused3_confirmation_v2_inputs",
        "status": "audited_complete",
    }
    monkeypatch.setattr(
        runner,
        "_load_registry",
        lambda *_args: (registry, lambda *_a, **_kwargs: {"loaded": True}),
    )
    monkeypatch.setattr(
        runner, "execute_panel", lambda *_args, **_kwargs: _full_panel()
    )
    registry_path = tmp_path / "registry.json"
    lock_path = tmp_path / "lock.json"
    _write_json(registry_path, registry)
    _write_json(
        lock_path,
        {
            "status": "frozen_v2_design",
            "lineage": {"v1_confirmation_registry_sha256": "0" * 64},
        },
    )
    smoke = tmp_path / "smoke"
    runner.run_confirmation(
        registry_path=registry_path,
        lineage_lock_path=lock_path,
        output=smoke,
        smoke=True,
    )
    observed = resource_preflight._validate_redacted_smoke(
        smoke, registry_path=registry_path, lineage_lock_path=lock_path
    )
    assert observed["run_identity_sha256"] == _run_identity(
        registry_sha=runner.sha256_path(registry_path),
        lock_sha=runner.sha256_path(lock_path),
    )
    raw = json.loads((smoke / "raw_results.json").read_text(encoding="utf-8"))
    assert raw["audited_registry"] == {
        "path": "audited_registry.json",
        "sha256": runner.sha256_path(registry_path),
        "registry_id": registry["registry_id"],
    }


def test_resource_preflight_writes_closed_outcome_blind_bundle(
    frozen_surfaces: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def execute_child(**kwargs: object) -> dict[str, object]:
        _write_smoke(
            Path(kwargs["worker_output"]),
            registry_path=frozen_surfaces["registry"],
            lock_path=frozen_surfaces["lock"],
        )
        return _resource_values()

    monkeypatch.setattr(resource_preflight, "_execute_fresh_child", execute_child)
    output = tmp_path / "resource"
    result = resource_preflight.run_resource_preflight(
        smoke_path=frozen_surfaces["smoke"],
        registry_path=frozen_surfaces["registry"],
        lineage_lock_path=frozen_surfaces["lock"],
        output=output,
    )
    assert result["artifact_status"] == resource_preflight.ARTIFACT_STATUS
    assert result["status"] == "pass"
    assert result["resources"]["descriptive_only"] is True
    assert result["resources"]["promotion_veto"] is False
    assert result["resources"]["outcomes_persisted"] is False
    expected = {
        "manifest.json",
        "resource_preflight.json",
        *(f"worker_smoke/{name}" for name in resource_preflight.SMOKE_FILES),
    }
    assert {
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file()
    } == expected
    for path in output.rglob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if path.name not in {"resource_preflight.json", "manifest.json"}:
            continue
        for key, item in value.items():
            if key != "protocol":
                resource_preflight._scan_outcome_keys(item, path=(key,))


def test_resource_preflight_refuses_overwrite(
    frozen_surfaces: dict[str, Path], tmp_path: Path
) -> None:
    output = tmp_path / "resource"
    output.mkdir()
    (output / "sentinel").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        resource_preflight.run_resource_preflight(
            smoke_path=frozen_surfaces["smoke"],
            registry_path=frozen_surfaces["registry"],
            lineage_lock_path=frozen_surfaces["lock"],
            output=output,
        )
    assert (output / "sentinel").read_text(encoding="utf-8") == "preserve"


def test_resource_preflight_rejects_extra_or_outcome_smoke_fields(
    frozen_surfaces: dict[str, Path], tmp_path: Path
) -> None:
    malformed = tmp_path / "malformed"
    _write_smoke(
        malformed,
        registry_path=frozen_surfaces["registry"],
        lock_path=frozen_surfaces["lock"],
        extra_raw={"test_accuracy": 0.75},
    )
    with pytest.raises(RuntimeError, match="top-level schema mismatch"):
        resource_preflight._validate_redacted_smoke(
            malformed,
            registry_path=frozen_surfaces["registry"],
            lineage_lock_path=frozen_surfaces["lock"],
        )


def test_resource_preflight_rejects_raw_hash_corruption(
    frozen_surfaces: dict[str, Path]
) -> None:
    raw_path = frozen_surfaces["smoke"] / "raw_results.json"
    raw_path.write_bytes(raw_path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="raw-results hash mismatch"):
        resource_preflight._validate_redacted_smoke(
            frozen_surfaces["smoke"],
            registry_path=frozen_surfaces["registry"],
            lineage_lock_path=frozen_surfaces["lock"],
        )


@pytest.mark.parametrize("copy_name", ["audited_registry.json", "lineage_lock.json"])
def test_resource_preflight_rejects_local_identity_copy_corruption(
    frozen_surfaces: dict[str, Path], copy_name: str
) -> None:
    copy_path = frozen_surfaces["smoke"] / copy_name
    copy_path.write_bytes(copy_path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="copy differs"):
        resource_preflight._validate_redacted_smoke(
            frozen_surfaces["smoke"],
            registry_path=frozen_surfaces["registry"],
            lineage_lock_path=frozen_surfaces["lock"],
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("protocol", {"status": "changed"}, "protocol differs"),
        ("source_hashes", {"source.py": "9" * 64}, "source hashes differ"),
        ("environment", {"python": "0"}, "score environment mismatch"),
        ("run_identity_sha256", "9" * 64, "run identity mismatch"),
    ],
)
def test_resource_preflight_rejects_identity_drift(
    frozen_surfaces: dict[str, Path],
    field: str,
    replacement: object,
    message: str,
) -> None:
    _rewrite_raw(
        frozen_surfaces["smoke"],
        lambda raw: raw.__setitem__(field, replacement),
    )
    with pytest.raises(RuntimeError, match=message):
        resource_preflight._validate_redacted_smoke(
            frozen_surfaces["smoke"],
            registry_path=frozen_surfaces["registry"],
            lineage_lock_path=frozen_surfaces["lock"],
        )


@pytest.mark.parametrize("malformation", ["missing_manifest", "extra_file"])
def test_resource_preflight_rejects_malformed_worker_bundle(
    frozen_surfaces: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
) -> None:
    def malformed_child(**kwargs: object) -> dict[str, object]:
        worker = Path(kwargs["worker_output"])
        _write_smoke(
            worker,
            registry_path=frozen_surfaces["registry"],
            lock_path=frozen_surfaces["lock"],
        )
        if malformation == "missing_manifest":
            (worker / "manifest.json").unlink()
        else:
            (worker / "unexpected.json").write_text("{}\n", encoding="utf-8")
        return _resource_values()

    monkeypatch.setattr(
        resource_preflight, "_execute_fresh_child", malformed_child
    )
    output = tmp_path / f"resource-{malformation}"
    with pytest.raises(RuntimeError, match="artifact file set mismatch"):
        resource_preflight.run_resource_preflight(
            smoke_path=frozen_surfaces["smoke"],
            registry_path=frozen_surfaces["registry"],
            lineage_lock_path=frozen_surfaces["lock"],
            output=output,
        )
    assert not (output / "manifest.json").exists()
    assert not (output / "resource_preflight.json").exists()


def test_resource_preflight_child_failure_has_no_terminal_manifest(
    frozen_surfaces: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_child(**_kwargs: object) -> dict[str, object]:
        raise RuntimeError("fresh structural-smoke resource child failed")

    monkeypatch.setattr(resource_preflight, "_execute_fresh_child", fail_child)
    output = tmp_path / "resource"
    with pytest.raises(RuntimeError, match="resource child failed"):
        resource_preflight.run_resource_preflight(
            smoke_path=frozen_surfaces["smoke"],
            registry_path=frozen_surfaces["registry"],
            lineage_lock_path=frozen_surfaces["lock"],
            output=output,
        )
    assert not (output / "manifest.json").exists()
    assert not (output / "resource_preflight.json").exists()


def test_macos_time_peak_rss_parser_is_strict() -> None:
    assert (
        resource_preflight._parse_peak_rss(
            "  987654 maximum resident set size\n  0 page reclaims\n"
        )
        == 987654
    )
    with pytest.raises(RuntimeError, match="parse one"):
        resource_preflight._parse_peak_rss("no resource record")
    with pytest.raises(RuntimeError, match="parse one"):
        resource_preflight._parse_peak_rss(
            "1 maximum resident set size\n2 maximum resident set size\n"
        )
