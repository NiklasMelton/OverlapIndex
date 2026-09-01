# Shared bounded-evaluator result

The shared evaluator passed every frozen exactness and speed criterion.

Across six target-family panels and 96 candidate representations, both consumed
component scores were bit-exact to the legacy two-evaluator path. Broad RFF,
SAVE+RFF, and bounded-top-five rank groups were exactly identical in every
panel. The shared path fit one SAVE+RFF selector per fold, reused its already
computed RFF-mean block and feature-independent target scaling, and omitted the
unconsumed broad prototype fit and SAVE+RFF kernel calculation.

Median shared/legacy panel wall time was `0.5285743`, corresponding to a
`47.14%` reduction. Per-family ratios ranged from `0.5247` to `0.5402`; the
largest component-score difference was exactly zero.

This validates an implementation optimization only. It does not change the
confirmed bounded shortlist formula or the public API.

## Frozen identities

- implementation commit: `33ec50dd2cfdda6dcfada832e52ecf2cdc7a805c`
- protocol SHA-256: `c4cbf2ba593e50634b8a2076a801839423ccb1c1d1b9e3f783445044858faed4`
- raw-results SHA-256: `7547500b85517b9441dcf78cff975bfb40e48978ccf4761a88e87e870f6bba30`
- summary SHA-256: `a5332c0d4f88603bcf1fa8527204dc27be52115b8dbe36009f8f168fb19862a8`
- report SHA-256: `949393e2a18c1c3223a195cd4f71a59164414d04a57264a0d71da676da5e2baf`
- manifest SHA-256: `a7cc41ce06eff331ef915ea45ade7389bb8cbf20eb1eb7ec9bad807453ee42ad`
