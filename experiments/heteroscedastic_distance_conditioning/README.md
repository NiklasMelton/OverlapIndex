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

## Reproduction commands

Run the focused, outcome-free verification suite first. The prior-regression test is
included because its immutable lock and historical-gate handoff are part of this
experiment, even though the long regression run stays outside the default unit suite.

```console
PYTHONPATH=. python3 -m pytest -q \
  tests/test_heteroscedastic_conditioning.py \
  tests/test_heteroscedastic_fixtures.py \
  tests/test_heteroscedastic_manifest.py \
  tests/test_heteroscedastic_diagnostics.py \
  tests/test_heteroscedastic_runner.py \
  tests/test_heteroscedastic_statistics.py \
  tests/test_heteroscedastic_resources.py \
  tests/test_heteroscedastic_reporting.py \
  tests/test_heteroscedastic_prior_regression.py

# Requires a separately produced, hashed, outcome-blind GO review.
PYTHONPATH=. python3 -m experiments.heteroscedastic_distance_conditioning.runner \
  smoke \
  --output artifacts/heteroscedastic_distance_conditioning/smoke \
  --implementation-review \
    artifacts/heteroscedastic_distance_conditioning/reviews/implementation_review.json

PYTHONPATH=. python3 -m experiments.heteroscedastic_distance_conditioning.statistics \
  --stage smoke \
  --input artifacts/heteroscedastic_distance_conditioning/smoke \
  --output artifacts/heteroscedastic_distance_conditioning/smoke/analysis

# Development additionally requires an independent pre-screen GO review that
# identifies the exact smoke decision and smoke manifest hashes.
PYTHONPATH=. python3 -m experiments.heteroscedastic_distance_conditioning.runner \
  development \
  --output artifacts/heteroscedastic_distance_conditioning/development \
  --smoke-decision \
    artifacts/heteroscedastic_distance_conditioning/smoke/analysis/smoke_decision.json \
  --pre-screen-review \
    artifacts/heteroscedastic_distance_conditioning/reviews/pre_screen_review.json

PYTHONPATH=. python3 -m experiments.heteroscedastic_distance_conditioning.statistics \
  --stage development \
  --input artifacts/heteroscedastic_distance_conditioning/development \
  --output artifacts/heteroscedastic_distance_conditioning/development/analysis

# Run only if development emitted one immutable locked candidate.
PYTHONPATH=. python3 -m experiments.heteroscedastic_distance_conditioning.prior_regression \
  --promotion-decision \
    artifacts/heteroscedastic_distance_conditioning/development/analysis/promotion_decision.json \
  --output artifacts/heteroscedastic_distance_conditioning/prior_regression

# Run only if the prior regression passed for that same lock and decision hash.
PYTHONPATH=. python3 -m experiments.heteroscedastic_distance_conditioning.runner \
  confirmation \
  --output artifacts/heteroscedastic_distance_conditioning/confirmation \
  --promotion-decision \
    artifacts/heteroscedastic_distance_conditioning/development/analysis/promotion_decision.json \
  --prior-regression-decision \
    artifacts/heteroscedastic_distance_conditioning/prior_regression/prior_regression_decision.json

PYTHONPATH=. python3 -m experiments.heteroscedastic_distance_conditioning.statistics \
  --stage confirmation \
  --input artifacts/heteroscedastic_distance_conditioning/confirmation \
  --output artifacts/heteroscedastic_distance_conditioning/confirmation/analysis \
  --promotion-decision \
    artifacts/heteroscedastic_distance_conditioning/development/analysis/promotion_decision.json \
  --prior-regression-decision \
    artifacts/heteroscedastic_distance_conditioning/prior_regression/prior_regression_decision.json
```

The development and confirmation runners include mandatory fresh-process R0-R3
runtime/memory measurements. Warm-up rows are excluded, measured candidate order is
the frozen counterbalanced schedule, and stopped or completed artifacts are terminal.
The confirmation command never re-ranks candidates: it consumes the exact development
lock and prior-regression decision hashes. Long research runs stay outside the default
unit suite.
