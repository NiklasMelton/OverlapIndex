# Native OI shortlist evidence

The completed `oi_shortlist_prospective_continuation_v1` study supports carrying
forward the frozen `native_oi_top2` policy within its tested scope. It was the
only eligible geometric policy to pass all eight prespecified joint utility,
tail-risk, and cost gates. The aggregate record is versioned in
[native_oi_shortlist_v1.json](evidence/native_oi_shortlist_v1.json); reading this
page requires no ignored artifacts, restricted datasets, or old worktree.

## What was tested

Six independent sources each contributed three dependent draws, with four
backbones: 18 panels, 72 candidates, and 21,024 logical head fits. The independent
units are the six sources, not the draws or candidates. Native OI used
**KMeans with 16 initial prototypes per class plus refinement**, three-fold
training-only cross-fitting, fold-local centering, and one scalar RMS scale.
The top-two OI shortlist was followed by the existing training-CV finalist
selection. All predictions and training decisions were committed before held-out
target access.

This was a complete shortlist-policy evaluation, not a KMeans-versus-
MiniBatchKMeans trial or an isolated refinement ablation. Refinement alone
cannot be credited with the outcome. The study does not establish these exact
results for Vertebrae's current MiniBatchKMeans settings, adaptive prototype
counts, normalization or fold defaults, all modalities, or regression.
Downstream head training remains owned by the caller. Refinement remains
opt-in, `False` by default in both packages.

## Outcome

Regret is held-out macro-accuracy loss relative to the best validation-selected
head/backbone candidate in a panel. Values below are fractions, not percentage
points. Source means average the three dependent draws first.

| Policy | Mean final regret | Post-embedding cost / full bank | Joint gates |
| --- | ---: | ---: | --- |
| `native_oi_top2` | 0.012249228395061726 | 0.37484333261095054 | 8/8 passed |
| `nested_linear_probe_top2` (control) | 0.019085165895061724 | 0.6219313119439076 | Prespecified comparator |
| `fixed32_top2` | 0.018744510523377982 | 0.5142125882481305 | 7/8 passed |

Native OI's worst-case regret was **0.078125** (7.8125 percentage points),
equal to the control; its 95th-percentile regret was also 0.078125. Its logical
end-to-end cost fraction was 0.45037213210909577. Policy costs replay measured
logical components; they are not fresh deployment timings or cluster makespan.

Native OI improved mean regret on three sources and tied the control on three.
Its source-mean difference from control was -0.0068359375, with a paired
source-bootstrap 95% interval of
[-0.015299479166666666, -0.0007233796296296256]. This interval is **descriptive
with only six independent sources**.

Fixed32 failed `one_pp_miss_noninferior`: its shortlist one-percentage-point
miss rate was 0.38888888888888895 versus control's 0.11111111111111116, exceeding
the permitted excess of 0.05. Native OI's miss rate was 0.05555555555555569.
The direct fixed32-minus-native mean-regret interval includes zero; passing all
gates is the basis for the native-policy decision, not demonstrated direct
superiority to fixed32.

## Provenance and limits

The canonical final review was recorded on 2026-10-16. The versioned aggregate
contains exact SHA-256 hashes for its two source records:
`artifacts/oi_shortlist_prospective_continuation_v1/run-r2/final-review.json`
and `analysis.json`. It retains all eight gate outcomes for both geometric
policies, source-level mean regrets, uncertainty, and limitations. It includes
no individual observations, images, predictions, or identities.

All 72 candidate execution freezes identify package source files matching
[`606c5f0`](https://github.com/NiklasMelton/OverlapIndex/commit/606c5f00e43d419a6a733389c27aa303bd5d31a0)
exactly. Those package hashes are included in the aggregate. The original
independent review checked predictions, training-CV decisions, separation of
training/test groups, source aggregation, bootstrap results, component costs,
and static-input hashes without new model fits.

Earlier failed runs and the original failed pilot gate remain historical
outcomes. The outcome-blind A1 numerical safeguard and final-analysis packaging
repair are disclosed in the source review; they did not change the frozen
objective, convergence thresholds, grids, splits, selection, or analysis rules.
This bounded six-source result supports no universal, benchmark-equivalence,
or clinical-performance claim. No new experiment was run to prepare these
release documents.
