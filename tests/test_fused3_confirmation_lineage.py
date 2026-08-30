from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.fused3_confirmation_v2 import lineage
from experiments.fused3_confirmation_v2 import extractors


def _hash(value: str) -> str:
    return lineage.sha256_bytes(value.encode("utf-8"))


def _lineage() -> dict[str, object]:
    return lineage.make_lineage(
        protocol_sha256=_hash("v2-protocol"),
        code_identity_sha256=_hash("v2-code"),
        v1_protocol_sha256=_hash("v1-protocol"),
        v1_final_lock_sha256=_hash("v1-lock"),
        v1_confirmation_registry_sha256=_hash("v1-registry"),
    )


def test_lineage_is_exact_dual_history_envelope() -> None:
    observed = _lineage()
    assert tuple(observed) == lineage.LINEAGE_KEYS
    assert observed["authority"] == "FUSED3"
    assert observed["status"] == "frozen_v2_design"
    assert lineage.validate_lineage(observed) == observed


def test_lineage_rejects_extra_keys_and_mutated_v1_ancestry() -> None:
    observed = _lineage()
    with pytest.raises(ValueError, match="exactly"):
        lineage.validate_lineage({**observed, "candidate_authority_sha256": _hash("x")})
    with pytest.raises(ValueError, match="sealed"):
        lineage.validate_lineage(
            {**observed, "v1_protocol_sha256": _hash("changed")},
            expected=observed,
        )


def test_lock_is_fused3_only_and_never_allows_reselection() -> None:
    observed = _lineage()
    lock = lineage.make_lock_envelope(
        lineage=observed,
        candidate_authority_sha256=_hash("candidate-authority"),
    )
    assert tuple(lock) == lineage.LOCK_KEYS
    assert lock["selected_candidate"] == "FUSED3"
    assert lock["runner_up_reselection_allowed"] is False
    assert lineage.validate_lock_envelope(
        lock,
        expected_lineage=observed,
        expected_candidate_authority_sha256=_hash("candidate-authority"),
    ) == lock
    with pytest.raises(ValueError, match="reselection"):
        lineage.validate_lock_envelope(
            {**lock, "runner_up_reselection_allowed": True}
        )
    with pytest.raises(ValueError, match="FUSED3"):
        lineage.validate_lock_envelope({**lock, "selected_candidate": "LP-FULL"})


def test_build_lineage_from_files_requires_protocol_sidecar_and_exact_v1_bytes(
    tmp_path: Path,
) -> None:
    protocol = tmp_path / "protocol.json"
    sidecar = tmp_path / "protocol.sha256"
    protocol.write_text(json.dumps({"status": "draft"}), encoding="utf-8")
    sidecar.write_text(lineage.sha256_path(protocol) + "\n", encoding="utf-8")
    observed = lineage.build_lineage_from_files(
        protocol_path=protocol,
        protocol_sidecar_path=sidecar,
        code_identity_sha256=_hash("code"),
    )
    assert observed["protocol_sha256"] == lineage.sha256_path(protocol)
    assert observed["v1_final_lock_sha256"] == lineage.V1_FINAL_LOCK_SHA256
    sidecar.write_text(_hash("wrong") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar"):
        lineage.build_lineage_from_files(
            protocol_path=protocol,
            protocol_sidecar_path=sidecar,
            code_identity_sha256=_hash("code"),
        )


def test_lock_file_hash_matches_canonical_serialized_bytes(tmp_path: Path) -> None:
    lock = lineage.make_lock_envelope(
        lineage=_lineage(), candidate_authority_sha256=_hash("authority")
    )
    path = tmp_path / "lineage_lock.json"
    observed = lineage.write_lock(path, lock)
    assert observed == lineage.sha256_path(path) == lineage.lock_sha256(lock)
    with pytest.raises(FileExistsError, match="already exists"):
        lineage.write_lock(path, lock)


def test_candidate_authority_hash_is_byte_bound(tmp_path: Path) -> None:
    authority = tmp_path / "candidate_authority.json"
    authority.write_text('{"selected_candidate":"FUSED3"}\n', encoding="utf-8")
    first = lineage.authority_sha256(authority)
    authority.write_text('{"selected_candidate":"LP-FULL"}\n', encoding="utf-8")
    assert lineage.authority_sha256(authority) != first


def test_food_driver_is_hashed_before_import_and_factory_is_closed(tmp_path: Path) -> None:
    driver = tmp_path / "driver.py"
    driver.write_text(
        "_DEFAULT_MODELS = " + repr(extractors.FOOD_BACKBONE_IDS) + "\n"
        "def _build_final_output_extractors(models, *, batch_size, device):\n"
        "    return list(models)\n",
        encoding="utf-8",
    )
    expected = lineage.sha256_path(driver)
    factory = extractors.verified_food_extractor_factory(
        driver, expected_sha256=expected
    )
    assert factory(("dinov2-small",), batch_size=2, device="cpu") == ["dinov2-small"]
    with pytest.raises(ValueError, match="frozen Food backbone order"):
        factory(("resnet50", "dinov2-small"), batch_size=2, device="cpu")
    driver.write_text(driver.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        extractors.import_verified_food_driver(driver, expected_sha256=expected)
