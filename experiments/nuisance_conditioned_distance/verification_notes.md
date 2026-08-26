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

The final pre-outcome repository-wide Python 3.9 test run completed with 336
passing and 14 skipped tests. Its 34 warnings were the repository's expected
degenerate-prototype and unevaluable-multilabel warnings; no experiment,
joblib, numerical, or fit warning was emitted.

## Frozen execution and stop decision

- The independently reviewed pre-outcome implementation was committed as
  `eb298a845e3c7ac4b482fa7344cbc9d097e7364a`. The runner recorded that exact
  clean commit, the verified starting commit `165fa344653a72dd2abc04a20387d90b891b9acc`,
  and per-source hashes in every raw artifact.
- The 24-case smoke completed all 144 A--F rows with no candidate error or
  early stop. The excluded warm-up and first measured case had bit-exact A--E
  structural signatures. Two warnings on deliberately tiny cells were the
  existing under-prototyped-label warning, not numerical or fit failures.
- The 720-case screen completed all 4,320 paired A--F rows with no errors.
  Artifact completeness passed (720/720 cases and 4,320/4,320 rows), the
  designated-pair robustness grid passed (120/120 strata for each B--E arm),
  and the excluded warm-up repeat was bit-exact for A--E.
- The immutable screen raw and manifest SHA-256 values are
  `9e88c48369ff3dea28d119391cf66120d740b5e3a19c6963be5ec334238c23e3`
  and `9a289e261236e82bd94d188f44833971737267928240806be8ec98e06341137a`.
- No C--E candidate passed the recovered historical reliability gates, so the
  screen decision is `inconclusive` with no selected or locked candidate.
  C failed separated-FPR, AUROC/AUPRC, clean-MAE, calibration, family-drift,
  and stable-shift gates. D failed the family-drift gates. E failed the
  family-drift and stable-shift AUROC gates. The full synthetic, Food-101,
  large-budget runtime, G policy, and fresh-process memory panels were therefore
  not run; running them without a valid screen lock would violate the frozen
  stop rule.
- A post-outcome audit found two report-only extraction errors (the configured
  condition-number cap was shown as the measured value, and applied/eligible
  was mislabeled as prototype activity). Commit
  `f21fe5a67042822edd97740934719de3d0008c67` corrected only those diagnostics.
  Commit `20b06afcec6c98ef8afeeab93362df6a2cc6e0cd` then sorted robustness
  accumulation to remove last-bit `PYTHONHASHSEED` dependence without changing
  its estimand, bootstrap, threshold, gate status, or decision.
- The final analysis was generated twice from the unchanged raw screen and all
  ten output files were byte-identical. Its summary, promotion decision, and
  report SHA-256 values are
  `86d0238d292ebc6125ec7e06ba6fda12a192cc0ab5a8d9981db96a836b85c9c4`,
  `d3ed3d0b69e9cc51dd8c1fd32c52b3169fc8947a48f29c085743c7950fb46e3a`,
  and `3dbfd2da109821841d671d96d70ca5bd1c444a581f18bae49b33f7a774f42c8c`.
  Original analysis artifacts remain preserved beside `analysis_final_v2/`.
- After the two post-outcome analysis/reporting corrections, the six focused
  Python 3.9 suites passed 96 tests, and the final repository-wide suite passed
  338 tests with 14 skips and the same 34 expected upstream warnings.
