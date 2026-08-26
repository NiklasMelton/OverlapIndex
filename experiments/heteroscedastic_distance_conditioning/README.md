# Heteroscedastic distance-conditioning experiment

This private research package tests whether OI's remaining nuisance failure comes
from transferring one overconfident sample-weighted pooled precision metric across
heterogeneous class scale. It evaluates partial whitening, robust diagonal scale, and
class-balanced pooling without changing the public `OverlapIndex` API.

The human-readable design is in [`PLAN.md`](PLAN.md). The machine-readable candidate,
data, statistics, gate, lock, and artifact contract is in
[`protocol.json`](protocol.json). `protocol.json` wins if prose and code ever disagree.

## Current authorization

The protocol may be implemented and structurally verified. Candidate outcomes remain
unauthorized until all focused tests pass and an independent pre-outcome code review
returns GO. Smoke may expose invariant/structural checks only; it may not emit rankings
or change thresholds. Development, prior-regression, and confirmation stages must
respect the immutable lock and stop semantics in the protocol.

Food-101 is not authorized by this package. It requires a separate protocol after a
successful synthetic confirmation.

## Planned commands

Commands become active only after their corresponding modules and tests exist.

```console
python3 -m pytest -q \
  tests/test_heteroscedastic_conditioning.py \
  tests/test_heteroscedastic_fixtures.py \
  tests/test_heteroscedastic_manifest.py \
  tests/test_heteroscedastic_diagnostics.py \
  tests/test_heteroscedastic_runner.py \
  tests/test_heteroscedastic_statistics.py \
  tests/test_heteroscedastic_resources.py \
  tests/test_heteroscedastic_reporting.py

python3 -m experiments.heteroscedastic_distance_conditioning.runner \
  smoke --output artifacts/heteroscedastic_distance_conditioning/smoke

python3 -m experiments.heteroscedastic_distance_conditioning.runner \
  development --output artifacts/heteroscedastic_distance_conditioning/development

python3 -m experiments.heteroscedastic_distance_conditioning.statistics \
  --input artifacts/heteroscedastic_distance_conditioning/development \
  --output artifacts/heteroscedastic_distance_conditioning/development/analysis
```

Prior-regression and confirmation commands require a hashed passing development lock.
They will be documented when implemented. Long research runs stay outside the default
unit suite.
