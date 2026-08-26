"""Focused tests for immutable planning and authorization primitives."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.heteroscedastic_distance_conditioning import fixtures as f
from experiments.heteroscedastic_distance_conditioning import manifest as m


def test_protocol_and_rank_tables_are_frozen() -> None:
    assert m.verify_protocol_hash() == f.PROTOCOL_SHA256
    protocol = json.loads(f.PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["repository"]["starting_commit"] == m.EXPECTED_FOLLOWUP_STARTING_COMMIT
    assert protocol["repository"]["original_develop_base"] == m.EXPECTED_ORIGINAL_DEVELOP_BASE
    assert m.validate_rank_permutations(stage="development")
    assert m.validate_rank_permutations(stage="confirmation")
    with pytest.raises(ValueError):
        bad = list(f.DEVELOPMENT_SCALE_RANKS)
        bad[0] = (0, 0, 2, 3)
        m.validate_rank_permutations(scale_ranks=bad)


def test_case_grid_hashes_and_exact_counts() -> None:
    smoke = f.smoke_cases()
    development = f.development_cases()
    confirmation = f.confirmation_cases()
    assert m.validate_case_grid(smoke, stage="smoke")
    assert m.validate_case_grid(development, stage="development")
    assert m.validate_case_grid(confirmation, stage="confirmation")
    assert len({case.case_id for case in development}) == 5184
    assert len({case.case_id for case in confirmation}) == 10368
    assert len({m.case_grid_sha256(development[:100]), m.case_grid_sha256(development[:100])}) == 1
    with pytest.raises(ValueError, match="case count"):
        m.validate_case_grid(development[:-1], stage="development")
    mutated = list(development)
    original = mutated[0]
    mutated[0] = f.CaseSpec(
        original.stage,
        original.seed,
        original.scenario,
        original.balance,
        original.count_level,
        original.nuisance_dim,
        original.k,
        original.signal_state,
        (1, 0, 2, 3),
        original.size_ranks,
    )
    with pytest.raises(ValueError, match="rank assignment"):
        m.validate_case_grid(mutated, stage="development")


def test_schedule_is_hash_seeded_cyclic_and_max_difference_one() -> None:
    cases = f.smoke_cases()
    first = m.planned_execution_order(cases, f.SMOKE_METHODS)
    second = m.planned_execution_order(tuple(reversed(cases)), f.SMOKE_METHODS)
    assert first == second
    assert m.validate_execution_order(first, [case.case_id for case in cases], f.SMOKE_METHODS)
    # Every complete case cycle contributes one row to each position.
    position_counts = {}
    for row in first:
        position_counts[row["position"]] = position_counts.get(row["position"], 0) + 1
    assert max(position_counts.values()) - min(position_counts.values()) <= 1
    assert m.method_position_counts(first)


def test_schedule_does_not_depend_on_python_hash_seed() -> None:
    script = (
        "from experiments.heteroscedastic_distance_conditioning import fixtures as f; "
        "from experiments.heteroscedastic_distance_conditioning import manifest as m; "
        "print(m.sha256_bytes(m.canonical_json(m.planned_execution_order(f.smoke_cases(), f.SMOKE_METHODS)).encode()))"
    )
    outputs = []
    for hash_seed in ("1", "2", "random"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hash_seed
        result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True, env=env)
        outputs.append(result.stdout.strip())
    assert len(set(outputs)) == 1


def test_source_identity_excludes_output_artifacts(tmp_path: Path) -> None:
    root = tmp_path
    source = root / "experiments/heteroscedastic_distance_conditioning"
    source.mkdir(parents=True)
    for relative in m.DECLARED_SOURCE_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
    before = m.code_identity_sha256(root)
    output = root / "artifacts/heteroscedastic_distance_conditioning/smoke"
    output.mkdir(parents=True)
    (output / "raw_results.json").write_text("generated", encoding="utf-8")
    after = m.code_identity_sha256(root)
    assert before == after
    with pytest.raises(ValueError, match="generated output root"):
        m.source_hashes(root, ("artifacts/heteroscedastic_distance_conditioning/raw.json",))


def test_declared_sources_cover_execution_and_have_no_stale_analysis_module() -> None:
    declared = set(m.DECLARED_SOURCE_PATHS)
    assert "experiments/heteroscedastic_distance_conditioning/analysis.py" not in declared
    for required in (
        "experiments/heteroscedastic_distance_conditioning/runner.py",
        "experiments/heteroscedastic_distance_conditioning/resource_worker.py",
        "experiments/heteroscedastic_distance_conditioning/resource_benchmark.py",
        "experiments/heteroscedastic_distance_conditioning/prior_regression.py",
        "experiments/heteroscedastic_distance_conditioning/statistics.py",
        "experiments/heteroscedastic_distance_conditioning/reporting.py",
    ):
        assert required in declared


def test_decision_validation_and_confirmation_authorization() -> None:
    code_hash = "code-identity"
    review = {"status": "go", "protocol_sha256": f.PROTOCOL_SHA256, "code_identity_sha256": code_hash}
    assert m.validate_implementation_review(review, code_identity_hash=code_hash)
    assert m.validate_pre_screen_review(review, code_identity_hash=code_hash)
    smoke = {"status": "pass", "structural_pass": True, "protocol_sha256": f.PROTOCOL_SHA256, "code_identity_sha256": code_hash}
    assert m.validate_smoke_decision(smoke, code_identity_hash=code_hash)
    promotion = {"status": "locked", "locked_candidate": "W50-CB", "protocol_sha256": f.PROTOCOL_SHA256, "code_identity_sha256": code_hash}
    promotion_hash = hashlib.sha256((m.canonical_json(promotion) + "\n").encode()).hexdigest()
    prior = {
        "status": "pass",
        "locked_candidate": "W50-CB",
        "promotion_decision_sha256": promotion_hash,
        "protocol_sha256": f.PROTOCOL_SHA256,
        "code_identity_sha256": code_hash,
    }
    assert m.validate_promotion_decision(promotion, code_identity_hash=code_hash) == "W50-CB"
    token = m.authorize_confirmation(promotion, prior, code_identity_hash=code_hash)
    assert m.validate_confirmation_authorization(token)
    confirmed = f.confirmation_cases(token)
    assert len(confirmed) == 10368
    dataset = f.generate_confirmation_dataset(confirmed[0], authorization=token)
    assert dataset.case.stage == "confirmation"
    with pytest.raises(PermissionError):
        m.validate_promotion_decision({**promotion, "locked_candidate": "L"}, code_identity_hash=code_hash)
