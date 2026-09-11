"""Import-only compatibility shim for a cleaned experiment-local module.

Frozen downstream runners retain this module in their provenance graphs.  The
information-ladder diagnostic does not execute the broad-RFF candidate path;
calling it through this recovery shim therefore fails loudly instead of
silently changing archived semantics.
"""


def evaluate_broad_only_candidate_geometry(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        "the archived broad-RFF evaluator source is unavailable; this read-only "
        "compatibility shim cannot recompute its outcomes"
    )


__all__ = ["evaluate_broad_only_candidate_geometry"]
