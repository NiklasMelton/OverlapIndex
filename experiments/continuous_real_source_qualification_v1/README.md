# Continuous real-source qualification V1

This development-only screen asks whether four stronger real continuous targets actually create the backbone-ranking problem that Continuous OI is meant to solve. It evaluates only the already-qualified linear Ridge, quadratic Ridge, KNN, and RBF-SVR reference heads on five scout backbones. It does **not** evaluate OverlapIndex, reserved backbones, or select an OI method.

The four discovery targets are KonIQ-10k MOS, log-transformed JHU-CROWD++ head count, log median SUN RGB-D scene depth, and RSNA bone age in months. Each target uses a 2,000-image training pool, ten unique 600-image draws, and a fixed 1,000-image evaluation cohort. Published train/test partitions are retained for JHU and SUN; KonIQ and RSNA use deterministic, disjoint target-decile-stratified partitions. These are same-target-distribution comparisons, not distribution-shift tests.

The head family, search grids, fold-local preprocessing, and inner validation rules are inherited exactly from the passing V3 reference-library qualification. The source gate requires at least three independently sourced nonlinear challenge leads. A failure stops before any OI evaluation; thresholds may not be relaxed after outcomes.

SPAQ was replaced by KonIQ-10k before any predictive outcome was inspected because the official SPAQ Google Drive object returned 404 and the maintained mirror was 34.8 GB. SPAQ remains an untouched replication source.

Run the stages in order:

```bash
python -m experiments.continuous_real_source_qualification_v1.prepare
python -m experiments.continuous_real_source_qualification_v1.extract
python -m experiments.continuous_real_source_qualification_v1.run
```
