from experiments.scalable_relevance_kmeans import synthetic_check


def test_frozen_synthetic_check_is_complete_and_finite(monkeypatch) -> None:
    candidates = synthetic_check.base.PANEL_CANDIDATES[:2]
    monkeypatch.setattr(synthetic_check.base, "PANEL_CANDIDATES", candidates)
    monkeypatch.setattr(
        synthetic_check,
        "METHODS",
        ("raw_unrefined", "scalable_one_update"),
    )
    original_loads = synthetic_check.json.loads

    def loads_with_small_grid(value):
        payload = original_loads(value)
        payload["synthetic_regression"]["candidate_ids"] = [
            row["candidate_id"] for row in candidates
        ]
        payload["synthetic_regression"]["methods"] = list(synthetic_check.METHODS)
        return payload

    monkeypatch.setattr(synthetic_check.json, "loads", loads_with_small_grid)
    payload = synthetic_check.run_check()
    assert payload["artifact_status"] == "completed"
    assert len(payload["rows"]) == 4
    assert {row["method"] for row in payload["summary"]} == set(synthetic_check.METHODS)
