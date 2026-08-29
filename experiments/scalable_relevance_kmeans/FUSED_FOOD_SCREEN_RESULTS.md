# Fixed-Lloyd Food-101 development screen

Recorded 2026-08-29 on branch
`codex/experiment-m50-fast-conditioning` from executed commit
`91a633184acd11cfb6b89f77bc5cf32eb08b01b6`.

This is retrospective Food-101 development evidence. It is not an untouched
confirmation and it does not change public `OverlapIndex` behavior.

## Frozen identity

- Protocol SHA-256:
  `2b4289636157fcdcd552c688cd3212c6311a8fddf128766d48fe126ab03d1069`
- Executed code identity SHA-256:
  `2660bd81b113427e4c1da1cbef37830a5f35722887f6968463ef74a76b450216`
- Valid raw results SHA-256:
  `aa34175fa1d06ea4721561a0b1011c5a97267e5838b0ce044749922f3524f2a1`
- Valid terminal run manifest SHA-256:
  `e3f6d06c4094aa05c5ff0fefd4def96ecebcc32e63b39551aee715daa809a86e`
- Corrected analysis summary SHA-256:
  `19fff8bb9874a871d0c1c14b483e718dd50d90408b6ace93e699c874c0c451ba`
- Corrected analysis manifest SHA-256:
  `a2476dcc2b112321be8ca76d6569bd4b25091ca7249a931d0364402833511a02`

The valid run used Python 3.12.2, NumPy 2.1.3, sklearn 1.6.1, and the six
frozen one-thread controls. The preserved `fused_food_screen_v1` partial run
used Python 3.9 / NumPy 2.0 and stopped after a one-event `NOLLOYD` parity
difference on `convnext-tiny / nuisance_full`. The valid v2 rerun used the
exact environment of the prior speed artifact; the protocol, candidates,
outcomes, gates, and thresholds were not changed.

## Methods

- `NOLLOYD`: current capped relevance selector with one sklearn
  MiniBatchKMeans estimator per class.
- `LOOP3`: deterministic sampled initialization plus three full-batch Lloyd
  updates, executed class by class.
- `FUSED3`: the exact same initialization and three Lloyd updates in one
  padded class-batched kernel.
- `LP-FULL`: freshly measured five-fold full linear probe.

`LOOP3` and `FUSED3` matched exactly on all 30 panel scores, 150 fold scores,
fitted numerical-state hashes, and held-out score diagnostics. `NOLLOYD` and
`LP-FULL` matched all 60 hash-frozen prior scores exactly in the valid run.

## Ranking result

At replicate zero and 80 rows per class:

| Geometry / downstream head | `FUSED3` regret pp | `FUSED3` Spearman | `LP-FULL` regret pp | `LP-FULL` Spearman |
|---|---:|---:|---:|---:|
| baseline / linear | 0.000 | 0.964 | 0.000 | 0.976 |
| nuisance / linear | 0.000 | 0.855 | 0.000 | 0.939 |
| nonlinear / quadratic | 0.000 | 0.979 | 3.462 | 0.758 |
| nonlinear / kNN | 0.000 | 0.948 | 4.760 | 0.806 |
| nonlinear / RBF | 0.000 | 0.973 | 3.750 | 0.754 |

The fixed-Lloyd candidate chose the exact best backbone in every primary
panel. It preserved the current selector's zero nonlinear regret while the
linear probe selected `convnext-tiny` instead of the best
`openclip-vit-b-32` on all three nonlinear heads.

## Runtime result

Paired `FUSED3 / LP-FULL` wall-time ratios were:

| Geometry | Median | Upper 95% |
|---|---:|---:|
| baseline | 0.211 | 0.229 |
| nonlinearity_full | 0.370 | 0.432 |
| nuisance_full | 0.250 | 0.282 |

Median outer wall time across the 30 panels was 0.3750 seconds for `FUSED3`,
0.3608 seconds for `LOOP3`, 0.9832 seconds for `NOLLOYD`, and 1.3946 seconds
for `LP-FULL`. The loop was only 3.8% faster than the fused kernel, inside the
predeclared five-percent tie band, so the immutable screen decision selected
`FUSED3`.

## Decision and limitation

All 32 required candidate gates passed. The terminal decision is:

```text
status: pass_for_full_food_design
selected_candidate: FUSED3
eligible_candidates: [FUSED3, LOOP3]
full_replay_status: not_run_requires_separate_frozen_protocol
```

This screen has only one replicate and one sampling budget. It supports a
separately frozen 600-panel Food replay of `FUSED3`; it does not by itself
establish general reliability or a product claim.

## Analysis correction

The raw run artifact is unchanged. The original verifier incorrectly required
OI fold timing names on `LP-FULL`, whose frozen schema uses `predict_*`. The
versioned `fused_food_analysis_erratum.py` aliases those two clock fields in
memory only. Independent recomputation confirmed that scores, regrets,
correlations, runtime ratios, all 32 gates, and the selected candidate are
unchanged.
