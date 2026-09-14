# Continuous practical-value gate V1

This is a prospective, outcome-free admission and stopping contract for a
future broad continuous ranker benchmark. It does not load embeddings, fit a
head, fit OverlapIndex, evaluate a reserved backbone, or write result
artifacts. Its purpose is to prevent another broad comparison from being
opened when the reference target cannot support the claim.

The deployable comparator is training-only Linear CV. The primary
endpoint is top-one selection regret against the held-out best-head envelope.
Top-three opportunity regret, top-three mean utility regret, and material-pair
ordering are separate endpoints; Spearman and exact-winner rate are secondary.
All differences use paired 95% uncertainty bounds from training-draw and
evaluation-group resampling. A complete interval inside ±0.01 R2 is a practical
tie; unresolved pairs are retained as unresolved.

Entry requires a method whose configuration was frozen before the preceding
mechanism diagnostic outcomes and that survives that diagnostic. Step 2 is
allowed to produce outcome-bearing evidence, but its outcomes cannot tune the
candidate or select a change. Entry also requires at least two fresh source dataset groups,
at least two qualified nonlinear groups and two linear controls, and at least
two reserved backbones. The reference must itself have enough achievable
linear-probe regret: its 95% lower bound must exceed 0.01 R2. This prevents a
nonlinear head improvement from being advertised as a backbone-ranking gain
when the linear selector has too little regret left to recover. The headroom
field is explicitly computed from the training-only Linear CV selector;
held-out linear-head versus envelope disagreement alone cannot admit a
practical nonlinear opportunity.

The planned real-embedding capacity grid is 384, 768, and 1536 dimensions at
300, 600, and 1200 training rows per draw wherever the source supports the
cell. Native dimensions are measured for every backbone and reported without
coercion. A supplementary synthetic scaling grid of 128, 512, and 1024
dimensions may be used to diagnose computational scaling, but cannot replace
the native real-embedding measurements.
Runtime includes the complete deployable selector from loaded embeddings
through scores and required resampling, with one excluded warmup, five timed
repeats, one thread control, geometric-mean and p90 ratios, and peak memory.
One formulation change is allowed after the mechanism result only when its
diagnosis-to-change mapping was declared before Step-2 outcomes, the changed
configuration receives a distinct hash, and the Step-2 outcomes did not tune
it. After that, the independent evaluation is run once. A failed gate stops
the formulation family; rows are not added, ties are not broken post hoc, and
a new method is not selected from the same outcome.

The numerical margins are carried forward from
`experiments/continuous_ranking_benchmark_v2/DESIGN.md` and the reliability,
opportunity, recoverable-ranker, and early-mean holdout result notes. The only
stricter additions are explicit LP-headroom, fresh source-group, and reserved-
backbone requirements.

This package contains only protocol validation. No outcome run is authorized
by this slice.
