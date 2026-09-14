# Continuous reference historical screen

This development-only screen applies the qualified V3 four-head library to all
eight cached historical continuous targets. It uses five scout backbones and
the existing 200-row evaluation cohorts, so its nonlinear tags are leads for a
larger benchmark rather than confirmation claims.

Targets with reusable pools receive ten unique 600-row training draws. Targets
whose archived pools contain only one complete budget receive one draw and are
reported with that limitation. Every transformation is fitted inside each
inner fold, and the exact V3 head grids are consumed without tuning.

The screen uses paired resampling and Bonferroni-adjusted utility intervals to
classify each target as a nonlinear-challenge lead, linear control, mixed, or
unresolved. No OverlapIndex method or reserved backbone is evaluated.

