# Continuous OI reliability-gate result

## Decision

**FAIL (narrowly).** The frozen leave-dataset-out reliability gate did not
establish lower top-5 mean utility regret than broad RFF OI. Its paired
contrast against broad OI was `-0.005803`, with a 95% interval of
`[-0.013052, +0.000810]`; negative values favor the gate.

This is a post-outcome developmental diagnostic over two independent dataset
groups, not confirmation evidence. No threshold, feature, or estimator was
retuned after inspecting the result.

## Aggregate result

| Method | Spearman | Top-1 regret | Top-1 exact | Top-5 regret | Top-5 precision |
|---|---:|---:|---:|---:|---:|
| broad RFF OI | 0.7810 | 0.02518 | 0.667 | 0.01739 | 0.893 |
| bounded 2k OI | 0.7602 | 0.02518 | 0.667 | 0.02758 | 0.827 |
| fixed 50/50 Borda blend | 0.7988 | 0.01830 | 0.600 | 0.01425 | 0.893 |
| LODO reliability gate | 0.8035 | 0.01330 | 0.700 | 0.01159 | 0.913 |
| oracle panel gate | 0.7974 | 0.01770 | 0.667 | 0.00980 | 0.927 |
| linear probe | 0.9827 | 0.00000 | 1.000 | 0.00000 | 1.000 |

The reliability gate improved the point estimate of top-5 regret by about one
third relative to broad OI and retained zero top-5 opportunity regret. It also
improved top-1 opportunity regret by `-0.011875`, with a 95% interval of
`[-0.027089, -0.000288]`. It nevertheless remained materially behind the
linear probe: the top-5 regret contrast was `+0.011586`, 95% interval
`[+0.005938, +0.017235]`.

## What happened

The OI-only diagnostic features were informative on UTKFace. When trained on
Oxford Pet and evaluated on the two UTKFace target families, the gate made the
same broad-versus-blend decision as the per-panel oracle in all 10 panels. It
selected the blend in 3 of those 10 panels.

The reverse transfer failed. When trained on UTKFace, the gate selected the
blend in all 5 Oxford Pet panels but agreed with the oracle in only 2. Oxford
Pet's broad-score scale was outside the UTKFace training range, so the ridge
model extrapolated rather than recognizing that the panel was unsupported.
Overall oracle-decision agreement was 12/15.

The failure is therefore more specific than "OI diagnostics do not predict
aggregation reliability." They predicted the UTKFace regimes accurately, but
two dataset groups are insufficient to learn a stable cross-dataset mapping,
and the current gate has no support-aware abstention rule.

## Interpretation and next step

Do not promote or retune this gate on the same panels. The result supports two
next steps:

1. Add at least two or three genuinely new natural continuous dataset groups,
   then evaluate the unchanged diagnostic schema with group-held-out folds.
2. Before fitting another adaptive gate, freeze a conservative support-aware
   version that falls back to broad OI when the held-out panel lies outside the
   training diagnostic envelope. Evaluate that rule only on new groups.

The first is the stronger scientific next experiment because it tests whether
the promising UTKFace transfer generalizes. The second is a useful safety
mechanism, but should not be tuned to repair the observed Oxford Pet errors.

## Reproducibility

- Frozen implementation commit: `4c033726b4977e944b1a1a7a58df4af1e66cf3c0`
- Protocol SHA-256: `52a41f040d2c5a7f97e24cc44ad3950dec0a8c0da05ca9c66371707eaef1eaa5`
- Summary SHA-256: `e1ac14ce0aa1bc9e6ec47cf7214ca41f0e5f8ff2828a8036f3112c64b6616336`
- Report SHA-256: `8b7d7f411689dcab217037a2897b8d61dd38cc88e21037aaf93d009ed398bd28`
- Manifest SHA-256: `555be3f6b3db3478be6f3ad10e0100247334222d2a287387aa21eaaa9affa153`
- Artifact reconstruction: byte-exact against the frozen source panels
- Verification: `231 passed` in the complete `tests/test_continuous_*.py` suite
