import json

from experiments.continuous_cached_embedding_smooth_boundary_v1 import run


def test_boundary_protocol_is_narrow_reference_only_resolution():
    protocol = json.loads(run.PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["scope"]["overlapindex_evaluated"] is False
    assert protocol["scope"]["reserved_backbones_evaluated"] is False
    assert len(protocol["targets"]) == 1
    assert len(protocol["backbones"]) == 3
    assert len(protocol["evaluated_views"]) == 1
    assert len(run.smooth._grid(protocol)) == 504

