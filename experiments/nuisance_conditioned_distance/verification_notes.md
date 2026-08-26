# Verification notes

## Pre-screen integration

- The recorded `develop` start, current `develop`, and branch/develop merge-base
  were independently verified as
  `165fa344653a72dd2abc04a20387d90b891b9acc`.
- The archived Stage-2 generator and config hashes were independently verified
  as `277af6d6a6fd8fb8d613faeb4bf0bce845145b57c6a0862de05167efd5f0efb3`
  and `66719e8f0a053a618e0c946dd34a346d34d31ef639287fba578dfbed599b6586`.
  The recovered grid contains 2,160 unique cases; screen contains 720 and smoke
  contains 24.
- The recovered fixture was also compared directly with the archived
  `confirm_generators` output across all 54 family/condition/strength
  combinations at seed 1000. Training and evaluation feature arrays, labels,
  signal/nuisance components, and generator truth matched bit-for-bit.
- Before the final statistical review, the runner owner executed one bounded
  correctness-only smoke case and reported A--F status `ok`. No candidate
  score, regret, or ranking outcome from that case was inspected or used to
  amend candidates, thresholds, or promotion rules. The final robustness-AUC
  amendment was driven by a row-order/mixed-estimand code review, not by that
  smoke result.
- The final pre-outcome protocol SHA-256 is
  `1ec47e521b67ce39db5c887004a68f1f2319a0ed5ee075067d0818b0823df0ea`.
  Earlier amendments removed an independently reviewed ambiguity in the
  secondary robustness AUC by excluding genuine-overlap cases and fixing
  complete matched candidate-minus-B nuisance curves. The last amendment,
  commit `bd30eae`, restored the recovered 64/128/256/512/640 Food-101
  large-budget resource panel after the independent pre-screen review found
  that the 64/68/72/80 selector replay was not the frozen runtime estimand.
  Follow-up commit `b7595c9` bound that panel to A, B, the one immutable
  screen-locked C/D/E candidate, the full probe, and the capped G component;
  it does not spend full-run resources evaluating unpromoted runner-ups.
  Commit `1bf882a` then made the G accounting explicit: E remains a timed
  diagnostic primitive when needed, the capped probe remains a separate
  component, and the reported policy time follows the frozen triggered versus
  untriggered B/E/locked-candidate aggregation rather than relabeling the
  capped component as the complete guardrail.
  No smoke, screen, full, or candidate outcome was viewed before either change.

An intermediate Python 3.9 verification run produced one joblib warning:
joblib could not parse the host's reported physical-core count and fell back to
the logical-core count. This was the known environment/core-count warning, not a
model, fit, or numerical warning. The experiment runner now establishes
`LOKY_MAX_CPU_COUNT=1` together with the BLAS/OpenMP one-thread controls before
numeric imports. A subsequent combined adapter, runner, Food, analysis, and
reporting run under the archived Python 3.9 environment completed with 67
passing tests and no warning. The warning is therefore
classified as benign for results, while the explicit serial controls remain
recorded in every run manifest.

The archived Python 3.9 byte-compilation command initially attempted to create
its interpreter cache outside the writable worktree. Verification was rerun with
`PYTHONPYCACHEPREFIX=/tmp/overlapindex_pycache`; this changes only bytecode cache
placement, not imported source or experiment behavior.

The pre-screen repository-wide Python 3.9 test run completed with 309 passing
and 14 skipped tests. Its 34 warnings were the repository's expected
degenerate-prototype and unevaluable-multilabel warnings; no experiment,
joblib, numerical, or fit warning was emitted.
