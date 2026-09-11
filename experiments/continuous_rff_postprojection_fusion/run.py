"""Fail-loud import shim for the archived postprojection-fusion runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


# These values only permit import of the byte-for-byte restored broad-RFF
# runner.  No recovered experiment invokes that runner through this shim.
MAP_OFFSETS = tuple(range(8))
REPLICATE_SEEDS = (4103, 4201, 4303)
SMALL_TASK = "archived_small_task_unavailable"
LINEAR_TASK = "archived_linear_task_unavailable"
TIE_ATOL = 1.0e-12


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_payload(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _unavailable(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("the archived postprojection-fusion source is unavailable")


evaluate_postprojection_candidate_geometry = _unavailable
_global_candidate_ordinal = _unavailable

