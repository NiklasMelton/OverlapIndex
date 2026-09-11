# Real opportunity-stratified discovery result

## Outcome

The frozen no-refit join completed with status
`single_dataset_rank_changing_opportunity_stratum_only`.

Across 78 archived panels, 540 backbone rows, and 22 source-task-roster
cells, only two cells passed the recoverability, utility-spread, nonlinear-
prevalence, and rank-change requirements. Both are the same semantic target
on FGVC Aircraft:

- bbox occupancy over the ten-backbone roster;
- box area fraction over the five scout backbones.

They are therefore one dataset-level opportunity, not two independent positive
examples. Twelve cells form recoverable LP-dominated controls and eight are
unusable because target recovery or between-backbone spread is inadequate.
No method is promoted.

## The critical distinction

SmallNORB elevation confirms why the stricter definition was necessary. RBF
or another nonlinear head materially improved 73.3% of scout-backbone rows and
86.7% of reserved-backbone rows. Nevertheless, the held-out linear head chose
the same envelope-best backbone in every seed on both rosters, with zero top-1
regret and no top-three miss. This is real nonlinear *head* opportunity, but it
is not nonlinear *backbone-ranking* opportunity.

SmallNORB azimuth sine shows the converse failure. It has abundant rank change,
including a 66.7% winner-change rate and 23.3% material pair reversals, but its
median best-head R2 is negative. Rankings among unrecoverable models are not a
useful transferability benchmark.

Aircraft box area is the only archived target that satisfies both sides:
recoverable utility and a nonlinear correction large enough to alter the
backbone decision.

## Ranker behavior

Equal weight is given to the two Aircraft task/roster cells within the positive
stratum.

| Training-only ranker | Mean Spearman | Top-1 regret R2 | Exact-best rate | Top-3 opportunity regret R2 | Top-3 recall |
|---|---:|---:|---:|---:|---:|
| Continuous broad RFF | 0.3627 | 0.03609 | 0.200 | **0.00796** | **0.656** |
| Continuous SAVE RFF | 0.0097 | 0.06780 | 0.100 | 0.02442 | 0.478 |
| Linear CV | **0.5079** | **0.03597** | **0.200** | 0.00995 | 0.622 |
| Raw four-head CV maximum | 0.4733 | **0.03597** | **0.200** | 0.00995 | 0.622 |

Broad Continuous OI is nearly tied with Linear CV in top-1 regret and is
slightly better for top-three shortlisting, but it is substantially worse in
full-ranking Spearman. SAVE is clearly worse. The raw four-head training
envelope does not improve regret here, so simply evaluating more head families
is not enough on this real target.

The LP-dominated control stratum remains an important safety test. Across its
12 cells, Linear CV achieved 0.7276 Spearman and 0.01942 regret, versus 0.5951
and 0.03688 for broad Continuous OI. The current continuous variants therefore
do not preserve linear-case ranking quality.

## Interpretation

The discovery pool resolves two recurring ambiguities:

1. A nonlinear head winning many individual backbone cells does not establish
   that nonlinear potential changes the backbone ranking. SmallNORB elevation
   demonstrates this directly.
2. Strong rank disagreement is not useful when no head recovers the target.
   SmallNORB azimuth and several centroid targets demonstrate this failure.

The archived real data supply only one genuine positive dataset. That is too
narrow to optimize, select, or confirm a general continuous-OI method. It also
explains why repeated aggregation and prototype modifications have not yielded
a stable universal conclusion: most real panels either reward the linear
ordering or contain unrecoverable nonlinear structure.

## Next experiment

Do not tune broad RFF against Aircraft. Acquire at least two fresh structured
continuous datasets selected from task semantics before backbone outcomes are
seen. The most useful targets combine a meaningful held-out signal with a
plausibly non-monotone or periodic readout, such as object/camera pose,
keypoint localization, depth or distance, and counting under instance or scene
splits.

Use the same frozen four-head envelope first as a *dataset admission* screen.
Keep only targets that pass the current recoverability, spread, nonlinear-
prevalence, and rank-change gates; retain matched recoverable LP-dominated
targets as controls. Only after at least two independent datasets qualify
should we compare or modify continuous OI. A separate untouched dataset/roster
must remain reserved for confirmation.

For method development, report two product modes separately:

- full-ranking Spearman, where Linear CV currently wins;
- shortlist regret/recall, where broad Continuous OI shows a small Aircraft-
  specific advantage worth testing prospectively.

## Verification and identity

- Protocol SHA-256:
  `bbd7b6657bb3a8550146cf007bbb58ead94a3f111d825e677362a2b75257cbab`
- Runner SHA-256:
  `3d9ffabfa0f8dd8d04b359c54361f1047ab3d1a8f0fc15bf57473c05fc1f04d0`
- Raw results SHA-256:
  `395e3cf123d36daf93d3b7b430640fd9ba02a140d76d77b662839a21a81ffc03`
- Summary SHA-256:
  `04aafbeb094f0b3b24a4f1cbe91fdc72b9ce74bc6b744826d5f7266eabfeecdf`
- Report SHA-256:
  `55be335658ca2ec4408cbdad196d8bd57113829722f2b28540f657c247576ed7`
- Manifest SHA-256:
  `7a03318255dc72682aab24f7b288a8c044b5183b7991ebb847a5ceca6e71508e`
- Focused and source-chain tests: 35 passed.
- Independent raw-file reconstruction exactly matched stratum membership and
  all four positive-stratum Spearman/regret aggregates.

An initial artifact that aggregated medians at the panel level was preserved
as `artifacts/continuous_real_opportunity_stratified_discovery_noncanonical_panel_median`.
The canonical artifact above pools candidate rows directly, as specified; the
stratum classification was unchanged.
