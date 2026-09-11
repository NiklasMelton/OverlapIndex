"""Fail-loud import shim for the archived RFF convergence diagnostic."""


def _unavailable(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("the archived RFF convergence source is unavailable")


_rff_mean_geometry = _unavailable
_map_seed = _unavailable
ranking_metrics = _unavailable
_load_json_evidence = _unavailable
_load_gtsrb = _unavailable
_panel_candidates = _unavailable

