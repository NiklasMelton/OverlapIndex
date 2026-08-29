import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.scalable_relevance_kmeans import benchmark


def _factory(_model: str, _arm: str, budget: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(900 + int(budget))
    labels = np.repeat(np.arange(3), 30)
    values = rng.normal(size=(labels.size, 12)).astype(np.float32)
    values[:, :3] += np.eye(3, dtype=np.float32)[labels] * np.float32(2.0)
    return values, labels


def test_anchor_writes_hash_bound_complete_bundle(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "BUDGETS", (128, 640))
    monkeypatch.setattr(
        benchmark,
        "_validate_frozen_protocol",
        lambda: "a" * 64,
    )
    monkeypatch.setattr(
        benchmark,
        "_validate_thread_environment",
        lambda: dict(benchmark.THREAD_ENVIRONMENT),
    )
    output = tmp_path / "anchor"
    result = benchmark.run_anchor(output, dataset_factory=_factory)
    assert result["artifact_status"] == "completed"
    assert len(result["rows"]) == 6
    assert all(row["warmup_score_exact"] for row in result["rows"])
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    observed = hashlib.sha256((output / "raw_results.json").read_bytes()).hexdigest()
    assert manifest["raw_results_sha256"] == observed
    assert manifest["row_count"] == 6


def test_anchor_refuses_overwrite(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "_validate_frozen_protocol", lambda: "b" * 64)
    monkeypatch.setattr(
        benchmark,
        "_validate_thread_environment",
        lambda: dict(benchmark.THREAD_ENVIRONMENT),
    )
    output = tmp_path / "anchor"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")
    try:
        benchmark.run_anchor(output, dataset_factory=_factory)
    except FileExistsError:
        pass
    else:
        raise AssertionError("benchmark must refuse a non-empty output directory")
