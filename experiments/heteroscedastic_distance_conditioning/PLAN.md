# Heteroscedastic distance-conditioning follow-up

Status: **implementation authorized after the protocol commit; no candidate outcomes
are authorized yet**.

Independent design review: **GO for conversion to a hashed machine-readable
protocol**. User authorization permits implementation after that protocol is committed;
smoke/development outcomes remain guarded by the review artifacts in `protocol.json`.

This plan follows the stopped nuisance-conditioned-distance experiment. Before any
candidate outcomes are generated, it must be independently reviewed, converted to a
machine-readable `protocol.json`, hashed, and committed. The completed experiment,
its protocol, and its artifacts remain immutable.

## 1. Motivation and narrower diagnosis

The preceding screen ruled out global full whitening and the tested sample-weighted
pooled-within-class diagonal/full transforms. It nevertheless supplied a sharper
working diagnosis:

1. Within-class conditioning improved pooled false-overlap detection and selector
   regret, so nuisance corruption does occur before prototype refinement and
   adjacency.
2. Diagonal and full pooled conditioning performed similarly, so correcting
   off-diagonal covariance is not the main missing ingredient in the observed panel.
3. Both pooled methods failed the heteroscedastic-family drift gate, and the full
   method also failed a stable-to-shift AUROC gate. A single fully whitened,
   sample-weighted covariance is therefore too rigid when nuisance scale is
   class-conditional, heavy-tailed, imbalanced, or changes between fit and held-out
   rows.
4. The condition-number cap was inactive. The failure is not explained by a binding
   numerical cap.

The follow-up tests this more specific hypothesis:

> OI's remaining nuisance failure is driven primarily by overconfident estimation and
> transfer of one pooled diagonal precision metric under heterogeneous scale. Partial
> whitening should reduce overcorrection; robust scale estimation should reduce
> sensitivity to heavy tails and contaminated rows; class-balanced pooling should
> prevent large classes from dominating the metric under imbalance.

This is still a hypothesis. In particular, the experiment must allow the result that
none of partial whitening, robust scale, or class balancing explains the observed
family drift.

## 2. Questions and falsifiable claims

The experiment has four primary questions.

### Q1 — Does geometry corruption precede the downstream OI failure?

Measure the paired chain

`distance ordering -> neighbor purity -> prototype purity -> adjacency -> refinement -> score -> selector regret`

under controlled changes to class scale, tail behavior, imbalance, and train/held-out
scale assignment. The diagnosis is supported only if changes in downstream errors are
preceded by corresponding changes in held-out distance/neighborhood recovery. A score
movement alone is not mechanistic evidence. This is ordered mechanistic concordance,
not formal causal localization: no stage-specific intervention is included.

### Q2 — Does partial whitening outperform full whitening?

At an otherwise identical estimator and weighting rule, compare covariance exponents
`gamma = 0.25`, `0.50`, and the legacy full-whitening value `1.00`. For diagonal
variance `v_j`, the fitted transform is

`T_jj = v_j ** (-gamma / 2)`.

`gamma = 0` must be implemented as an exact identity test seam, but is not a research
candidate because direct raw OI is already the control.

### Q3 — Does robust pooled scale address tails and contamination?

At fixed `gamma = 0.50`, compare ordinary OAS diagonal scale with two predeclared
robust estimators: MAD scale and MAD-seeded winsorized OAS. Robust methods must improve
the heavy-tail/contamination interaction, not merely an overall pooled average.

### Q4 — Does class-balanced pooling address imbalance?

At fixed estimator and `gamma`, compare sample-weighted residual rows with equal total
weight per class. Class balancing must improve the imbalance interaction without
materially worsening the balanced cells.

## 3. Evidence hierarchy

Evidence is partitioned before execution.

| Evidence | Use | Status |
|---|---|---|
| Prior nuisance-conditioned screen | Design evidence and mandatory regression panel | Already observed |
| New mechanistic development panel | Diagnose effects and lock one candidate | Development only |
| New synthetic confirmation seeds | Evaluate the locked candidate and fixed diagnostic comparators once | Untouched until lock |
| Existing Food-101 replay | Deferred to a separately frozen retrospective protocol | Already observed |
| Future new embedding panel | Required before a public/default algorithm change | Not currently available |

No result from an already observed panel may be described as untouched confirmation.
The confirmation seed list and generator hash must be committed before the development
run, and the confirmation rows must not be generated or inspected until a single
candidate is locked.

## 4. Controls and candidate table

All methods retain balanced-median prototype refinement except control A. A and B must
instantiate the current upstream `OverlapIndex` directly and exactly.

| ID | Scale estimator | Residual weighting | gamma | Role |
|---|---|---|---:|---|
| A | none | not applicable | 0 | exact unrefined OI control |
| B | none | not applicable | 0 | exact refined OI control |
| L | current pooled diagonal OAS | sample-weighted | 1.00 | exact legacy-D conditioned reference |
| P25 | pooled diagonal OAS | sample-weighted | 0.25 | partial-whitening dose |
| P50-SW | pooled diagonal OAS | sample-weighted | 0.50 | ordinary partial whitening |
| P50-CB | pooled diagonal OAS | class-balanced | 0.50 | isolates class weighting |
| W50-SW | MAD-seeded winsorized OAS | sample-weighted | 0.50 | robust partial whitening |
| W50-CB | MAD-seeded winsorized OAS | class-balanced | 0.50 | lead robust/balanced candidate |
| M50-SW | pooled MAD | sample-weighted | 0.50 | robust-estimator check |
| M50-CB | pooled MAD | class-balanced | 0.50 | robust/balanced alternative |

The previous global-isotropy and pooled-full candidates are not repeated as new
candidates. Their results remain prior development evidence. The legacy reference L
must exactly reproduce the previous pooled-diagonal transform and is not promotable.

### 4.1 Common fit and scoring contract

- Fit class centers and all conditioning state on training-fold rows only.
- Apply the frozen transform unchanged to held-out rows.
- Use the same transformed geometry for prototype fitting, refinement, adjacency,
  `predict`, and `score_fixed`.
- Do not refit conditioning or prototypes in `predict` or `score_fixed`.
- Preserve scalar, single-label targets and the strict Python-boolean refinement
  contract.
- Keep the implementation experiment-local; do not add a public constructor option or
  package export.
- Retain the preceding experiment's eigenvalue floor and condition-number cap unless
  the pre-outcome protocol review changes them. Record whether either binds.

### 4.2 Residuals and weighting

For training row `i` in class `c`, define

`r_i = x_i - mean(x | y = c)`.

Sample weighting uses `w_i = 1 / N`. Class-balanced weighting uses
`w_i = 1 / (C * n_c)`, so every class contributes total weight `1 / C`.

For each OAS-based estimator family, obtain the OAS shrinkage coefficient from its
sample-weighted preprocessed residual matrix once. Use that same coefficient for the
sample-weighted and class-balanced diagonal moments, shrinking each moment vector
toward its own weighted mean variance. This prevents the weighting comparison from
also changing OAS effective-sample-size behavior. The sample-weighted ordinary path
must exactly reproduce the legacy diagonal OAS transform. When classes are balanced,
the SW and CB moments and transforms must be bit-identical or within a predeclared
machine-level tolerance if operation order prevents bit identity.

### 4.3 Robust scale definitions

Definitions must be finalized in `protocol.json` and covered by fixed-vector tests.
The proposed definitions are:

- **MAD:** use a deterministic weighted median for each residual feature, then
  `q_j = 1.4826 * weighted_median(abs(r_ij - weighted_median(r_.j)))` and
  `v_j = q_j^2`.
- **Winsorized OAS:** compute `q_j` as above, clip residuals featurewise to
  `[-3 q_j, 3 q_j]`, and compute the selected weighted diagonal moment. For the SW/CB
  comparison, both paths use the same sample-weighted median/MAD clip thresholds and
  the same sample-weighted OAS coefficient; only the final diagonal moment weights
  differ. MAD-only SW/CB candidates use their declared weights in the median itself.

Zero MAD, constant features, non-finite input, unknown methods, and invalid gamma must
fail or floor according to explicit tests. Robust diagnostics must record per-feature
clip counts, the fraction of rows clipped, raw and regularized condition numbers, and
whether the cap or floor changed any scale.

## 5. New mechanistic synthetic panel

### 5.1 Fixed base design

- Four scalar-labeled classes.
- Signal dimension: 2.
- Nuisance dimensions: 4 and 32.
- `k`: 2 and 8.
- Sample-count regimes:
  - small balanced: `[40, 40, 40, 40]`;
  - small imbalanced size-rank values: `[16, 24, 40, 80]`, permuted across classes;
  - large balanced: `[160, 160, 160, 160]`;
  - large imbalanced size-rank values: `[64, 96, 160, 320]`, permuted across classes.
- Signal states: linearly separated, nonlinearly separated, and genuine overlap at the
  previously frozen half-overlap severity.
- Development seeds: `21000` through `21011`.
- Confirmation seeds: `31000` through `31023`.

Generate independent maximum-size training and held-out banks with
`numpy.random.SeedSequence([seed, stream_code])`; held-out streams are independent,
not slices of training streams. Stream codes are fixed as signal train/eval `100/101`,
uniform nuisance train/eval `200/201`, contamination mask `300/301`, outlier Gaussian
`400/401`, and row order `500/501`.

Signal RNGs append the signal-state index to the seed sequence. Nuisance banks do not:
for each seed and pool, generate common uniform and outlier banks of shape
`(4, 320, 32)` plus a row-wise contamination mask of shape `(4, 320)`. Nuisance
dimension 4 is the first four coordinates of the dimension-32 bank. The same nuisance
banks are reused across signal states, scenarios, balances, counts, and `k`.

Each bank contains 320 rows per class before deterministic nested selection:

- **linear separated:** class centers on the first axis at
  `[-2.25, -0.75, 0.75, 2.25]`; add independent uniform signal noise in
  `[-0.20, 0.20]^2`;
- **nonlinear separated:** four concentric rings with radii
  `[0.50, 1.00, 1.50, 2.00]`, uniform angle, and uniform radial perturbation in
  `[-0.08, 0.08]`;
- **genuine half-overlap:** classes 0 and 1 each draw exactly half their maximum bank
  from the shared square `[-0.20, 0.20]^2` and half from private centers
  `[-1.15, 0]` and `[1.15, 0]`; classes 2 and 3 use private centers `[3.50, 0]`
  and `[5.00, 0]`, all with uniform `[-0.20, 0.20]^2` perturbation. Only pair
  `(0, 1)` has overlap truth `0.5`. Shared and private rows are interleaved before
  nested selection, so every even selected count retains exactly half overlap.

For a balanced small/large cell, take the first 40/160 bank rows per class. For an
imbalanced cell, size ranks 0..3 mean `[16, 24, 40, 80]` rows at small and
`[64, 96, 160, 320]` at large; assign ranks with Section 12.2's permutation. Concatenate
class selections and apply one deterministic row order shared by all nine scenarios.
Thus smaller cells are nested subsets of larger cells and every scenario in a selector
panel has identical signal rows, labels, and row order.

Reference heads remain linear logistic, degree-two logistic, 5-neighbor kNN, and RBF
SVC with the preceding experiment's fixed estimator settings. The linear selector
probe uses five-fold stratification and fold-local standardization as specified below.

### 5.2 Nuisance construction

For class `c`, use four log-spaced class scales between `R**(-1/2)` and
`R**(1/2)`, then normalize the vector so its unweighted mean squared scale is one.
This varies heterogeneity without silently increasing average nuisance energy in the
balanced panel.

The held-out `reversed` shift assigns the class-scale vector in reverse class order;
the training vector is unchanged. Held-out parameters may be used to calculate truth
diagnostics but never to fit or select the transform. Scale-to-class permutations are
hashed in the manifest: the 12 development seeds use a position-balanced subset in
which every scale rank occurs three times at every class label, and the 24 confirmation
seeds use all 24 permutations exactly once.

Imbalanced class sizes use a separately hashed, position-balanced size-to-class
permutation. The pre-run manifest check must verify the exact marginal and joint rank
counts in Section 12.2. The same seed's scale permutation is reused across
H2/T/C/X/ALL.

Base nuisance distributions are:

- Gaussian: standard normal;
- heavy-tail: Student `t(5)` multiplied by `sqrt(3 / 5)` to give unit variance;
- contaminated: a row mixture of `0.95 * N(0, 1)` and `0.05 * N(0, 64)`, where 64
  is the outlier component's variance (standard deviation 8), divided by
  `sqrt(4.15)` so the marginal feature variance is one. The contamination mask is
  paired across candidates.

For a class with assigned scale rank, form the four log-spaced scale values, normalize
their unweighted mean squared value to one, and set

`nuisance = amplitude * class_scale * base_noise`.

Append this nuisance matrix to the two signal columns. For reversed held-out shift,
replace training rank `r` by held-out rank `3-r`; training ranks never change.
Contamination is row-wise: one Bernoulli mask applies to every nuisance coordinate of
that row. Draw uniforms in the open interval using float64 clipping; Gaussian noise is
the normal inverse CDF of the uniform bank, Student-t noise is the `t(5)` inverse CDF
times `sqrt(3/5)`, and the outlier component uses the independent Gaussian stream.

All scenario contrasts use common random numbers, not merely common candidate rows.
For every seed/base cell, generate the maximum signal row bank, uniform variates,
Gaussian variates, contamination mask, and row-order indices once. Derive Gaussian
and Student-t nuisance from the same uniform ranks, derive contaminated nuisance from
the shared Gaussian/outlier banks, select smaller/imbalanced row sets deterministically
from the maximum bank, and then apply scenario-specific amplitude and class scales.
The raw bank hashes and derivation recipe are artifact provenance.

### 5.3 Predeclared scenario blocks

Use the following block design instead of an uncontrolled full Cartesian sweep.

| Block | Amplitude | Scale ratio R | Distribution | Held-out scale assignment | Purpose |
|---|---:|---:|---|---|---|
| S0 | 0 | 1 | Gaussian | stable | clean negative control |
| S1 | 1 | 1 | Gaussian | stable | homoscedastic control |
| S2 | 2 | 1 | Gaussian | stable | strong homoscedastic control |
| H1 | 1 | 2 | Gaussian | stable | moderate heteroscedasticity |
| H2 | 2 | 4 | Gaussian | stable | strong heteroscedasticity |
| T | 2 | 4 | Student t(5) | stable | heavy tails |
| C | 2 | 4 | contaminated Gaussian | stable | row contamination |
| X | 2 | 4 | Gaussian | reversed | transform transfer shift |
| ALL | 2 | 4 | contaminated Gaussian | reversed | hard interaction |

Each scenario is crossed with balanced and imbalanced layouts, both nuisance
dimensions, both sample-count levels, both `k` values, and all three signal states.
This gives 432 cases per seed, 5,184 development cases, and 51,840 paired
A/B/L/candidate rows.

Confirmation uses the same 432-case layout for 24 untouched seeds. To confirm the
four mechanistic claims without reopening candidate selection, it always runs A, B,
L, P50-SW, W50-SW, and W50-CB, plus the locked candidate if distinct. This gives
10,368 confirmation cases and between 62,208 and 72,576 method rows. The fixed
non-promotable comparators cannot replace a failed locked candidate.

Within each stage, execution order is a deterministic hash-seeded cyclic schedule.
For every method, the counts in any two execution positions may differ by at most one
over the complete stage. Exact equality is required only when the case count is
divisible by the number of distinct methods. The manifest records the planned
position table, and analysis verifies the observed table exactly against it.

If implementation reveals that these counts are wrong, execution must stop and the
manifest must be corrected and re-frozen before any outcomes are viewed.

### 5.4 Selector panels

For selector outcomes, a complete panel is identified by

`(seed, balance, sample_count, nuisance_dim, k, signal_state)`.

Its nine selectable embedding items are exactly
`[S0, S1, S2, H1, H2, T, C, X, ALL]`. All nine items share the same signal rows,
labels, train/held-out split, and latent nuisance bank; only the predeclared nuisance
transformation differs. Higher OI score is better. The primary nuisance selector
subpanel is `[H1, H2, T, C, X, ALL]`; the full nine-item panel is diagnostic, and
`[S0, S1, S2]` is a homoscedastic negative-control ordering panel. A panel with a
missing, duplicate, errored, or non-finite item is incomplete and cannot enter any
selector summary or promotion decision.

For each downstream head and reported subpanel, the reference best is the largest
held-out accuracy among that subpanel's items. Regret is best accuracy minus the
accuracy of the item selected by the selector. OI-score ties use the frozen scenario
order above; a selected item is exact-best when its reference accuracy is within
`1e-12` of the reference maximum and within-one-point when its regret is at most
`0.01`.

The linear-probe comparator ranks items by five-fold out-of-fold training accuracy
using `StratifiedKFold(5, shuffle=True, random_state=seed)` and a fold-local
`StandardScaler` plus `LogisticRegression(C=1, solver="lbfgs", max_iter=2000,
n_jobs=1, random_state=seed)`. First-observed deterministic integer label encoding is
used for scikit-learn split/fit compatibility; original labels still reach OI and
conditioner fit. Held-out reference accuracy is never used as the probe selector score.

Held-out reference heads use integer-encoded labels and exactly: unscaled
`LogisticRegression(max_iter=500, solver="lbfgs", random_state=seed)`; unscaled
`PolynomialFeatures(degree=2, include_bias=False)` plus that logistic estimator;
`KNeighborsClassifier(n_neighbors=5)`; and `SVC(kernel="rbf", C=1, gamma="scale")`.
Rank correlation is tie-safe Spearman between selector score and held-out reference
accuracy within each complete panel. Rank AUC is the normalized trapezoid integral of
that Spearman value across the small and large total-sample levels on a
`log2(total_rows)` axis, grouped by all remaining panel axes.

The promotion nuisance-linear estimand is the paired complete-seed difference
`candidate regret - linear-probe-selector regret` on the six-item nuisance subpanel
for the held-out linear head, pooling balance only after balanced and imbalanced
results are also reported separately. Nonlinear retention is the paired
`candidate regret - B regret` on that same subpanel for each quadratic, kNN, and RBF
head. Missing either balance stratum makes the associated gate inconclusive. The
prior locked-candidate regression, not this new panel, supplies the unchanged frozen
clean-linear selector gate.

## 6. Mechanistic measurements

All measurements are paired at case, seed, split, and latent row bank. Execution
position is counterbalanced but need not be identical across methods in a cell.

### 6.1 Geometry before OI

On held-out rows, compare each fitted metric with the known signal-only oracle:

- top-`k` neighbor-set Jaccard with oracle neighbors;
- cross-class neighbor impurity;
- Spearman correlation of pairwise distances on a fixed, seed-selected pair sample;
- transform stability across five deterministic stratified folds: fit the transform
  on four folds using integer-encoded labels only for split planning, compare all ten
  fold-pair feature-scale vectors, and record Spearman plus median and maximum absolute
  log-scale differences. Original scalar labels still reach conditioner fit unchanged.

The oracle is diagnostic only and must never fit or select a candidate.

### 6.2 Prototype and adjacency stages

- normalized held-out label entropy per occupied prototype, macro and support-weighted;
- mixed-prototype rate over occupied prototypes, empty-prototype rate over all
  prototypes, weighted majority-label assignment purity, and prototype-owner label
  agreement. Backend ownership entropy is not reported because it is pure by
  construction;
- exact directional class-pair overlap-event support, hits, and evidence, asserted
  against OI's post-`score_fixed` sparse adjacency and pairwise index state;
- refinement eligible/applied counts divided by all prototypes;
- which prototype parents changed and the paired total refined-versus-unrefined score
  movement. No per-prototype score contribution is inferred from unavailable state.

### 6.3 End outcomes

- false-overlap evidence on separated pairs;
- AUROC, archived-order AUPRC, FPR, FNR, Brier, ECE, severity Spearman, and ordering;
- OI score movement relative to B;
- rank AUC, selection regret, exact-best, and within-one-point selection rates against
  linear, quadratic, kNN, and RBF references;
- conditioning fit, OI fit/refinement, adjacency, `score_fixed`, wall/CPU time, and
  fresh-process peak memory on the frozen resource cells.

Production timing treats held-out adjacency/scoring/aggregation as the single
externally measured `score_fixed` stage because OI exposes no separate adjacency
timer. Fit timing separates conditioning fit from inner OI fit/refinement, but does not
derive either by subtracting noisy outer totals. Any event-trace replay is labeled
diagnostic cost, not production adjacency cost.

Designated-pair evidence uses the prior convention: the arithmetic mean of directed
`0 -> 1` and `1 -> 0` evidence for the known pair. Separated truth is zero; the genuine
overlap signal state retains the prior half-overlap truth and pair labels.

### 6.4 Resource cells

Fresh-process peak memory and externally timed stage measurements are mandatory for
all development seeds on these exact cells:

| ID | Scenario | Balance | Count | Nuisance d | k | Signal |
|---|---|---|---|---:|---:|---|
| R0 | S0 | balanced | small | 4 | 2 | linear separated |
| R1 | H2 | balanced | large | 32 | 8 | linear separated |
| R2 | C | imbalanced | large | 32 | 8 | genuine overlap |
| R3 | ALL | imbalanced | large | 32 | 8 | nonlinear separated |

Development measures B, L, and every promotable candidate. Confirmation measures B
and the locked candidate on the same four identities for every confirmation seed.
Missing or non-finite resource evidence makes the relevant candidate inconclusive.
Runtime and memory ratios are paired to B within seed/resource cell, then pooled over
all four cells and stage seeds (48 development or 96 confirmation ratios). Median and
p95 gates use `numpy.quantile(..., method="linear")`; the individual memory gate uses
the maximum paired ratio. Per-cell summaries are reported but do not replace the
frozen pooled gate.

## 7. Primary estimands and statistics

Use complete-seed paired block bootstrap with 10,000 resamples and seed `20260812`,
retaining the archived pooled-row AUROC/AUPRC conventions. Report point estimates and
two-sided percentile 95% intervals for all outcomes.

The four primary mechanistic estimands use held-out cross-class neighbor impurity on
the two **separated** signal states as the primary endpoint. Genuine-overlap rows are
excluded because cross-class neighbors can be correct there; they enter only the
reliability gates. For each case, impurity is the fraction of each held-out row's
top-`k` neighbors whose scalar class differs, averaged equally over held-out rows.
Within a seed/contrast/balance, average complete cell means equally over linear and
nonlinear separated signals, nuisance dimensions, count levels, and `k`. Lower
impurity is better:

1. **Heteroscedastic geometry penalty** on balanced cells:
   `neighbor_impurity_L(H2) - neighbor_impurity_L(S2)`. The diagnosis requires the
   98.75% lower bound to be above zero.
2. **Partial-whitening benefit** on balanced cells: the H2-minus-S2 penalty of
   P50-SW minus the same penalty of L. The 98.75% upper bound must be below zero.
3. **Robustness benefit** on balanced cells: for W50-SW versus P50-SW, compute both
   the T-minus-H2 and C-minus-H2 difference-in-differences. The primary statistic is
   the worse (larger) of the two contrasts in each bootstrap draw; its 98.75% upper
   bound must be below zero. M50-SW is a prespecified secondary replication, not a
   substitute if W50-SW fails.
4. **Class-balance benefit:** for H2, compute the imbalanced-minus-balanced penalty of
   W50-CB minus the same penalty of W50-SW. The 98.75% upper bound must be below zero.
   P50-CB/P50-SW and M50-CB/M50-SW are secondary replications.

Repeat each contrast descriptively for distance-rank recovery, mixed-prototype rate,
false adjacency, refinement activity, false-overlap evidence, family drift, and linear
selector regret. A primary geometry claim is not enough for promotion: the downstream
reliability and regret gates must also pass. Stable-to-reversed transfer is the
separate prespecified contrast `(X - H2)`; `ALL - C` is the contamination-plus-shift
interaction stress test. Both are computed separately within balance.

For mechanistic claims, provide ordinary 95% intervals and family-wise 98.75%
Bonferroni intervals across the four primary claims. A mechanism is supported only
when its interval satisfies the direction above and its negative controls do not
violate the existing reliability margins. Crossing zero is inconclusive, not evidence
of no effect.

Negative controls use separated-state cross-class neighbor impurity with the same
complete-seed aggregation as the primary endpoint:

- partial whitening: the worst, within each bootstrap draw, of P50-SW-minus-L on
  balanced S0, S1, and S2;
- robust estimation: W50-SW-minus-P50-SW on balanced Gaussian H2;
- class balancing: W50-CB-minus-W50-SW on balanced H2.

For each control, the upper paired 95% bound must be at most `0.01` absolute neighbor
impurity. The within-draw maximum handles multiplicity over S0/S1/S2. In addition,
every method must satisfy the previously frozen genuine-overlap gates in every
applicable block, and false-overlap/family-drift/selector-regret controls use their
explicit eligibility margins rather than the new `0.01` geometry margin.

Correlations along the pipeline are descriptive. Do not call them formal causal
mediation unless a separate mediation protocol is frozen before outcomes.

For development family-drift gates, a "family" is one of the nine scenario blocks.
Within each block, compute the complete-seed candidate-minus-B change in absolute OI
evidence error against known overlap truth. Within each seed, equally average the
complete `2 balance x 2 dimension x 2 count x 2 k x 3 signal = 48` cells before the
paired seed bootstrap. Every block's upper paired 95% bound must be at most `0.02`.
This definition is separate from, and does not replace, the prior panel's frozen
historical family-drift estimand.

For the lock rule, separated false-overlap robustness is the maximum, over separated
S2/H2/T/C/X/ALL-by-balance blocks, of the candidate-minus-B upper paired 95% bound for
designated-pair false-overlap evidence. Missing a block makes the candidate
ineligible; no new strength-AUC is inferred from the non-Cartesian block design.

## 8. Development eligibility and lock rule

A candidate is eligible for a one-shot confirmation only if all of the following pass:

1. exact A/B and gamma-zero parity;
2. exact deterministic fitted state, structural diagnostics, predictions, and scores;
3. train-only fit and no-refit tests;
4. no frozen genuine-overlap gate violation;
5. nuisance-linear regret versus the probe: upper paired 95% bound at most 0.01;
6. each nonlinear-head regret versus B: upper paired 95% bound at most 0.01;
7. family-drift worsening: upper paired 95% bound at most 0.02 in every family;
8. stable-shift degradation: within each balance, candidate-minus-B degradation for
   `X - H2` and `ALL - C` has upper paired 95% bound at most zero for pair AUROC,
   archived-order AUPRC, Brier error, clean MAE, and absolute nuisance-drift error;
9. no clear runtime or fresh-process memory violation of the preceding protocol's
   gates. Development fresh-process memory is mandatory on R0-R3 for B, L, and every
   candidate.

Among eligible candidates, lock exactly one using this order:

1. smallest upper 95% bound for nuisance-linear regret versus the probe;
2. then smallest worst-family family-drift upper bound;
3. then smallest worst-block separated false-overlap upper bound;
4. start with all eligible candidates, retain those within `0.001` of the minimum on
   criterion 1, then within `0.001` of the surviving minimum on criterion 2, then
   within `0.001` of the surviving minimum on criterion 3; if more than one remains,
   use simplicity order
   `P50-SW, P25, P50-CB, W50-SW, W50-CB, M50-SW, M50-CB`.

If no candidate is eligible, stop. After the lock, a confirmation failure rejects the
locked candidate and never promotes a runner-up.

The already observed prior nuisance panel is deliberately not used to choose among
new candidates. After the development rule locks one candidate, run only A, B, L, and
that candidate on the prior panel. Every unchanged historical gate must pass. Failure
rejects the lock and does not reopen the development ranking.

## 9. Confirmation and final interpretation

After the locked candidate passes the prior-panel regression, run the fixed comparator
set and locked candidate described in Section 5.3 once on seeds `31000..31023`. Reuse
the complete gates and bootstrap definitions without modification. Confirmation
passes only if the locked candidate passes every reliability, family-drift,
stable-shift, linear-regret, nonlinear-retention, determinism, runtime, and memory
gate. The fixed comparators confirm Q1-Q4 but remain ineligible for promotion.

Algorithm and mechanism conclusions are separate. Algorithmic confirmation requires
the locked-candidate gates above. Mechanistic confirmation requires all four primary
98.75% criteria to repeat in their predeclared directions on confirmation seeds; each
claim is also reported separately. Failure of a mechanism criterion marks that
diagnostic claim unsupported but does not reject a locked candidate that independently
passes every algorithmic gate. It never permits a different candidate to be promoted.

This plan does not authorize a Food-101 run. After synthetic confirmation, a separate
pre-outcome protocol may import the prior Food grid, controls, statistics, gates, and
source hashes and evaluate the locked candidate retrospectively. A public/default OI
change still requires a future one-shot panel of genuinely new embeddings or tasks.

Interpret outcomes narrowly:

- A partial-whitening win supports overcorrection by full pooled precision.
- A robust-estimator interaction win supports tail/outlier sensitivity.
- A class-balanced interaction win supports large-class domination.
- A stable-panel win with a shift-panel failure supports metric-transfer instability.
- End-metric improvements without upstream neighborhood recovery do not validate the
  proposed geometry mechanism.
- Pooled improvements cannot override a failed family, shift, or genuine-overlap gate.

## 10. Execution stages and stop rules

1. Recover/cite source hashes and commit this reviewed plan as a frozen JSON manifest.
2. Implement experiment-local transforms, fixtures, diagnostics, and raw schema.
3. Pass fixed-vector, parity, leakage, no-refit, permutation, invalid-config, and
   deterministic-repeat tests.
4. Obtain an independent outcome-blind implementation review and write a hashed
   `implementation_review.json` with status GO.
5. Run a 30-case mechanistic smoke containing every block and candidate. Smoke may
   validate schema, counts, finite outputs, determinism, and obvious invariant
   violations only. It must not emit candidate rankings, estimate gates, change a
   threshold, or authorize a design amendment from observed candidate performance.
6. Independently review smoke structure, transforms, pairing, truth metrics, counts,
   bootstrap code, execution counterbalancing, and artifact identity without exposing
   candidate rankings; write hashed `pre_screen_review.json` with status GO.
7. Run the 12-seed development panel.
8. Apply the immutable eligibility and lock rule.
9. If one candidate locks, run only that candidate and the required controls on the
   prior frozen regression panel unchanged.
10. If the locked candidate passes every prior-panel gate, run the 24-seed confirmation
   panel once.
11. Obtain independent final methodological and code review.
12. If confirmation passes, stop and draft a separate Food-101/product protocol.

Leakage, A/B or reference mismatch, nondeterminism, incomplete paired blocks, or
artifact identity mismatch invalidates the whole stage; finish the current atomic
block, emit a stopped artifact, and terminate. A genuine-overlap, family/shift, or
resource failure excludes that candidate after its complete paired evidence is
available; it does not truncate other candidates or reinterpret an incomplete block.
If all candidates are excluded, stop the pipeline. After locking, any corresponding
failure rejects the locked candidate and stops the pipeline. Do not stop only because
a small smoke point is noisy.

## 11. Artifact and implementation requirements

- New experiment directory; do not overwrite prior raw data, reports, or decisions.
- Immutable manifest with generator/config hashes, exact case identities, seeds,
  package versions, environment, starting/experiment commits, dirty status, and code
  identity hashes.
- Warm-ups structurally excluded from score, timing, memory, and selection summaries.
- Deterministic candidate-position counterbalancing with observed counts exactly
  matching the manifest and maximum position-count difference one.
- Atomic partial/stopped artifacts with resume identity checks.
- Canonically ordered normalized JSONL tables for case/method, geometry, prototype,
  directional-pair, selector-panel, and resource rows; a small manifest records each
  file's schema, row count, and SHA-256. Per-case checkpoints are atomic and final
  assembly never rewrites one monolithic raw-result object.
- Report plots for mechanism contrasts, nuisance curves, family/shift gates,
  neighbor-to-regret diagnostics, refinement activity, and resources.
- Long-running research remains outside the default unit suite.
- No public OI API change and no modification to Vertebrae.

## 12. Frozen implementation decisions for protocol conversion

These decisions are part of the draft and may change only during pre-outcome review.
Once converted to `protocol.json` and hashed, they are immutable.

### 12.1 Numerical definitions

- Weighted median: stable-sort `(value, original_row_index)` and choose the smallest
  value whose cumulative normalized weight is at least `0.5`.
- Ordinary SW OAS: call `sklearn.covariance.oas(U, assume_centered=True)` and use the
  returned diagonal and shrinkage coefficient exactly; this path must be bit-identical
  to legacy L.
- CB OAS: for preprocessed residuals `u`, compute
  `m_j = sum_i w_i * u_ij^2`, `target = mean_j(m_j)`, and
  `v_j = (1 - alpha_SW) * m_j + alpha_SW * target`, reusing the paired SW coefficient.
- Balanced SW/CB moments and transforms must be bit-identical by routing equal weights
  through one implementation path.
- Regularization is inherited exactly: clamp negative variance to zero; if the largest
  variance is nonpositive, use identity; otherwise floor at
  `max(largest * 1e-8, largest / 1e4)` before applying gamma.
- Conditioner arithmetic is float64 outside identity mode. Fixed-vector transform
  tests use `rtol=1e-12, atol=1e-12`; A/B, gamma-zero, and L parity remain exact.
- Distance-rank diagnostics use all unordered held-out pairs when there are at most
  4,096; otherwise sample 4,096 without replacement using SHA-256 of the case ID as
  the RNG seed. Pair identities are shared by all methods and scenarios in a base cell.
- Signal draws are class-major. Linear draws one `(4,320,2)` uniform tensor. Rings
  draw all `(4,320)` angles, then all `(4,320)` radial perturbations. Half-overlap
  draws class 0 shared then private, class 1 shared then private, then class 2 and 3
  private tensors; shared/private rows are interleaved.
- Uniform nuisance banks have shape `(4,320,32)` and are clipped to
  `[nextafter(0,1), nextafter(1,0)]`. Gaussian uses `scipy.special.ndtri`; Student-t
  uses `scipy.stats.t.ppf(..., df=5) * sqrt(3/5)`. The contamination mask is one
  Boolean per row shared across all features; contaminated rows use `8 *` the
  independent standard-normal outlier bank, while other rows reuse the common Gaussian
  bank, and the result is divided by `sqrt(4.15)`.
- Row-order RNG seeds are `[seed, stream_code, signal_state_index, balance_index,
  count_level_index]`. Apply the permutation only after class-major nested selection;
  reuse it across scenarios, nuisance dimensions, and `k`.

### 12.2 Scale and class-size permutations

Development seeds use these scale-rank permutations in seed order:

```text
0123 1230 2301 3012
0321 1032 2103 3210
0213 1320 2031 3102
```

Their size-rank permutations are, in the same order:

```text
0123 0123 0123 1032
2031 3210 3210 3210
1302 2301 1302 2031
```

The manifest validator requires every scale rank and every size rank exactly three
times at each class label and every `(scale_rank, size_rank)` pair exactly three times
over the 48 class-by-seed positions.

Confirmation assigns the 24 scale permutations in lexicographic order. For seed
offset `i`, set `size_rank[c] = (scale_rank[c] + shift[i]) mod 4`, with

```text
shift = [0,0,0,0,0,0, 1,1,1,2,2,2, 3,3,1,1,3,1, 3,2,3,3,2,2]
```

The validator requires every scale and size rank exactly six times at each class label
and every joint rank pair exactly six times over the 96 class-by-seed positions.

### 12.3 Structural smoke identities

The 30 smoke cells are:

1. seed 21000, every scenario-by-balance combination (18 cells), small count,
   nuisance dimension 4, `k=2`, linear separated;
2. seed 21001, all nine scenarios (9 cells), large count, nuisance dimension 32,
   `k=8`, alternating balance beginning with S0 balanced, and cycling signal state
   `linear, nonlinear, overlap` from S0;
3. seed 21002, ALL/imbalanced/large/nuisance-dimension-32/`k=8` for each of the three
   signal states (3 cells).

The smoke runner executes every method on every cell but exposes only structural and
invariant checks as specified in Section 10. Each directional pair row retains the
original scalar `source_label` and `target_label`, `exact_state_match=true`, and one
`pair_structure` object. That object has exactly `support`, `hits`,
`sparse_adj_hits`, `pairwise_index`, and `evidence`; each value is only a
`{present, type, finite}` descriptor. Descriptors never contain `value` or another
recoverable numeric outcome, and no direct-field or source/target alias is allowed.
This descriptor-only schema was clarified before any smoke or candidate outcome was
run and is recorded as a pre-outcome amendment in `protocol.json`.

The remaining post-freeze, pre-outcome work is mechanical: implement and hash the
exact generator, latent-bank recipe, split identities, environment, code, and
manifest. No numerical threshold or estimand remains outcome-dependent.

## 13. Source evidence

- Prior final report:
  `artifacts/nuisance_conditioned_distance/screen/analysis_final_v2/report.md`
- Prior raw results SHA-256:
  `9e88c48369ff3dea28d119391cf66120d740b5e3a19c6963be5ec334238c23e3`
- Prior manifest SHA-256:
  `9a289e261236e82bd94d188f44833971737267928240806be8ec98e06341137a`
- Prior protocol SHA-256:
  `1ec47e521b67ce39db5c887004a68f1f2319a0ed5ee075067d0818b0823df0ea`

The prior screen is negative evidence that constrains this plan. It is not a tuning
set on which failed candidates may be relabeled or retrospectively redefined.
