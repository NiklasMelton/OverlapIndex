"""Outcome-neutral performance probes for the frozen M50 conditioner.

This follow-up package is intentionally separate from
``experiments.m50_backbone_ranking``.  The completed M50 experiment remains
byte-identifiable at its frozen commit; nothing here changes its candidate
recipes, artifacts, or conclusions.
"""

# Keep package import inert.  Importing ``benchmark`` here would cause
# ``python -m experiments.m50_fast_conditioning.benchmark`` to load the module
# twice and emit a runpy warning before the timed process begins.

__all__: list[str] = []
