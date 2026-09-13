# Continuous OI benchmark V2: reliable best-head backbone ranking

Design date: 2026-09-13. Status: proposed design, not an executable frozen
protocol. No V2 outcomes have been run. The next implementation must resolve
the sample inventory, exact target recipes, grouping metadata, checkpoints,
seed lists, and resource estimates before hashing the execution manifest.

## 1. Question and estimand

Can an inexpensive continuous OverlapIndex score rank pretrained backbone
embeddings by their attainable regression performance across a declared head
library, with linear-probe-level regret on linear tasks, lower regret on
nonlinear tasks, and lower scoring cost?

The practitioner supplies embeddings and labels from one target population.
All candidate backbones use identical examples and label budgets. Distribution
shift is not an experimental factor. Correlated images, subjects, scenes, or
video frames must nevertheless be grouped to avoid duplicate observations
crossing data roles.

For backbone b, family h, and downstream training budget B, define
U(b,h,B) as expected held-out R² of the predeclared training/tuning procedure
using B distinct labeled examples. The reference quality is
Q(b,B) = max_h U(b,h,B). The maximum follows averaging over training draws;
averaging the maximum on every draw is a different estimand and is reported
only as a sensitivity analysis.

The primary downstream budget is B=600. Scorers receive nested n=300 and n=600
subsets of the same training draw and are compared against the same Q(b,600).
On sufficiently large sources, a separate B=1200 track uses n=300/600/1200.
Do not pool these tracks or attribute a budget effect to a changing target
roster. The head library is an operational approximation to best attainable
performance; a finite library cannot establish the best of every possible head.

## 2. Corrections required by the audit

The completed V1 run is preserved at implementation commit
`b1f0776355ab565cdbc57aefb7526f2193e332c9`, with its result note at `77c4d9c`.
An additional read-only audit on 2026-09-13 identified these limitations:

| Observation | V2 response |
|---|---|
| The 120 reference fits used 371–395 distinct labels, mean 380, because nominal 600-row cohorts were bootstrapped with replacement. | Train on exactly B unique examples per draw. Use resampling of saved predictions for uncertainty, not duplicated training records. |
| Replaying the first COCO/DINO reference draw reproduced the saved sample hash and found 97/200, 92/200, and 89/200 validation rows also present in the corresponding inner training fold. | Split on original sample/group identity at every level. Regression tests must explicitly detect duplicate IDs across fold roles. |
| PCA was fitted on the complete outer-training set before its inner hyperparameter folds. | Fit PCA and every learned transform inside each inner fold. Cache only transforms with identical training-row identities. |
| Every head used row normalization followed by PCA-32. | Include original embeddings in the reference library; make compression a tuning option or a named ablation. |
| Only three training draws and 200 evaluation images supported the reference uncertainty calculation. | Use ten unique-sample training draws and at least 1,000 evaluation images where feasible, with group-aware uncertainty. |
| Four of five unstable targets had top-two gaps of about 0.01–0.017 R². | Distinguish material ordering, practical equivalence, and unresolved comparisons. |
| Smooth-local synthetic reference values were approximately zero across all candidates. | Require learnability and material candidate separation before treating a family as a positive control. |
| The same complete four-head elapsed time was reported for both the linear and four-head scores. | Time standalone scoring methods independently. Existing runtime rows cannot establish their relative cost. |

Final V1 evaluation rows were separate from training. The duplicate issue is
inner-fold contamination and a changed effective training budget; this audit
does not quantify its effect on final rankings. The earlier LSP wrist finding
is a lead to replicate under corrected evaluation, not an established result.

Fitting preprocessing inside cross-validation follows the
[scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).
Model selection can overfit its evaluation criterion; independent assessment
and selection-aware uncertainty are required
([Cawley and Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html)).

## 3. Benchmark composition

Maintain three separately reported collections.

### A. Historical development panel

Retain all eight V1 targets for correction/reproduction, including failures.
LSP wrist and CUB wing may remain in a smaller-data supplement when they cannot
meet the main cohort sizes. Use the five existing scout backbones and reuse
their exact image embeddings for the first pilot. Never label this collection
as new-dataset confirmation.

### B. Broad application panel

Propose the following twelve tasks, chosen for application coverage before
seeing V2 reference outcomes:

| Source | Predeclared targets | Tasks |
|---|---|---:|
| CUB | Leg span / bounding-box diagonal; wing-axis cosine | 2 |
| COCO | Wrist span / torso length; ankle span / torso length | 2 |
| Pascal VOC | Largest-box area fraction; log aspect ratio | 2 |
| Oxford Pet | Foreground compactness | 1 |
| UTKFace | Age under its existing valid-label definition | 1 |
| MPII Human Pose | Wrist span / torso length; ankle span / torso length | 2 |
| KonIQ-10k | Published mean image-quality opinion score | 1 |
| SPAQ | Published overall image-quality opinion score | 1 |

MPII supplies annotated pose images from videos, requiring grouping by source
video and selection from publicly labeled examples; its official hidden test
labels are not available for local evaluation
([official dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset)).
KonIQ has 10,073 quality-scored images
([official dataset](https://database.mmsp-kn.de/koniq-10k-database.html)).
SPAQ has 11,125 smartphone photographs with quality scores
([official release](https://github.com/h4nwei/SPAQ)). These are proposed new
sources; access, prior exposure in the experiment archive, duplicate content,
and usable group counts still need auditing. Image-quality targets are
application coverage, not an assumption that they must favor nonlinear heads.

Use complete source annotations, not the old 600-row prepared subsets, to
measure capacity. Main-track eligibility requires B unique training examples,
a reusable training pool large enough for different draws, and 1,000 distinct
evaluation images with adequate independent groups. Rows below this threshold
remain visible in the supplement; never silently lower B or clone examples.
The exact main/supplement assignment is fixed from metadata before outcomes.
Thus twelve is the proposed roster size, not a promise of twelve eligible main
tasks. Do not fabricate a replacement if CUB wing or another task is too small.

Split source pools into a development/qualification partition and an unopened
confirmation partition by group identity. For previously explored sources,
describe confirmation as a new-sample replication. A new-dataset claim is
restricted to sources whose outcomes have never guided method development.
Tasks from the same source remain one statistical domain. KonIQ and SPAQ also
receive a shared image-quality-domain sensitivity analysis.

### C. Qualified nonlinear challenge

Tag challenge tasks using only the reference heads on qualification data.
Keep every broad-panel task in the broad-panel report, regardless of tag.
Challenge results answer a conditional question about verified nonlinear
opportunities and must not be advertised as representative average performance.

Qualification requires all of the following, using simultaneous uncertainty:

- The best reference backbone has a positive held-out R² lower bound and a
  point estimate at least 0.20. This is a proposed learnability screen, not a
  universal definition of an interesting task.
- At least one near-leading backbone has a nonlinear-over-linear gain whose
  lower bound exceeds 0.02 R².
- Linear-only ranking incurs a best-head top-one regret lower bound above
  0.01 R², or at least two material backbone pair orderings reverse. Merely
  improving every backbone equally does not qualify.
- At least one top-three boundary or winner comparison is identifiable.

Before a general method-comparison claim, require at least three qualified
nonlinear tasks from three source datasets and three linear-dominated controls
from three source datasets. Classification as a linear control requires an
upper bound on the material nonlinear ranking advantage, not simply failure
to detect one. Other tasks are mixed or unresolved. If these counts cannot be
met, publish the qualification shortfall and expand the source roster in a new
version; do not relax thresholds after inspecting OI outcomes.

## 4. Data roles and repeated evaluation

Within every locked partition, allocate a training pool and disjoint reference
evaluation pool E. Preserve a single intended population and its natural target
distribution; target bins may balance fold assignment, but do not oversample
target extremes in reported test utility. Honor published splits where they
provide the desired population and labels; explicitly document any internal
split of a public training set.

Take ten training draws T_r of B distinct examples without replacement within
each draw. Draws may overlap each other when the source is finite; record their
overlap and do not call them independent datasets. Every backbone and head uses
the same T_r and inner folds. Scorer samples S_r,300 are deterministic subsets
of T_r; S_r,600=T_r in the primary track.

Use five inner folds for head hyperparameters, all grouped on identity.
Preprocessing and target standardization are fitted separately inside each
fold and on the full T_r for the final fit. No validation or evaluation label
can influence these fits. Fit on float64 for the reference numeric audit.

Use E=1,000 images initially, chosen by a frozen group rule. For sources with
enough data, preregister a separate E=2,000 endpoint and report both. Never
repeatedly add rows until a significance threshold passes. Where groups contain
multiple images, report group counts, preserve grouping in resampling, and
state whether loss weights images or groups. Primary loss weights images;
groups control independence and uncertainty.

Scorers see only S_r,n embeddings and labels. Expensive reference heads use
T_r. Evaluation predictions are saved, but evaluation labels and head quality
never enter the scorer. The primary target is shared Q(b,B), estimated across
the ten draws; per-draw envelope selection regret is a secondary diagnostic.

## 5. A reference library that can express the intended relationships

Retain four headline families: linear Ridge, degree-two Ridge, kNN, and RBF
regression. Strengthen their training recipe before freezing V2.

- Include full embeddings for every family. Degree-two regression on full
  embeddings may use the exact polynomial kernel to avoid explicitly storing
  millions of interactions. Record the precise feature/kernel scaling; it is
  a new reference recipe, not byte-identical to the old explicit polynomial fit.
- Candidate representation choices are original or row-L2 embeddings, with
  no PCA, unwhitened PCA-32, or unwhitened PCA-128. Retain only ranks feasible in
  every inner training fold. Choose representations on inner validation, never
  from evaluation performance. Each fitted representation uses a training-only
  center and global RMS scale; do not automatically equalize every nuisance
  coordinate to unit variance. A per-feature scaling view is a separately
  registered diagnostic if the pilot shows it is necessary.
- Search dimensionless Ridge regularization over a broad log grid, for example
  10^-6 through 10^2 for the mean-loss objective. Account for the conversion
  between this convention and solver alpha. Use float64 and a numerically
  stable solver. The library documents SVD as the most stable dense Ridge
  option for singular matrices
  ([Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)).
- For kNN, include uniform and distance weights and k in
  {3, 5, 9, 15, 31, 63, 127}, excluding values above the smallest inner-fold
  training size before constructing the grid.
- For RBF SVR, standardize y within each training fold. Search
  C in {0.1, 1, 10, 100}, epsilon in {0.01, 0.05, 0.1}, and bandwidth factors
  {1/4, 1/2, 1, 2, 4} times a training-only median pair distance. Specify the
  squared-distance/kernel convention exactly in implementation.
- Inspect boundary selections, solver warnings, negative R², and validation
  curves on development controls. A wider grid is not automatically adequate;
  the pilot must demonstrate that the reference actually learns each clean
  positive control. No expansion is allowed after confirmation is opened.

All four families have their own fair inner tuning, identical validation rows,
and a fixed deterministic tie rule. Treat representation selection as part of
head tuning, not another maximum chosen on the final test. Store every family
prediction so an RBF failure cannot be hidden by a maximum over heads.

## 6. Estimate the envelope without mistaking noise for an oracle

The desired Q(b,B) remains the maximum population utility across the head
families. Its plug-in estimate max_h mean_r R²_E(b,h,r) is useful but is not an
unbiased ground truth: taking a maximum over noisy test estimates selects
favorable noise.

Keep the plug-in envelope as a point estimate and build simultaneous bands
over all backbone/head utilities using paired resampling of evaluation groups
and training-draw indices. Share every resampling index across backbones and
heads. Use max-statistic or conservatively multiplicity-adjusted bands and
verify their coverage on synthetic data; do not rely on a naive percentile
interval for the selected maximum.

If [L_bh,U_bh] jointly cover all head utilities, use
[max_h L_bh,max_h U_bh] for the backbone envelope. For a pair (a,b), derive
conservative difference bounds [L_a-U_b,U_a-L_b], optionally tightening them
with a validated paired method. With delta=0.01 R²:

- Material order: the entire difference interval exceeds delta in one direction.
- Practical equivalence: the entire interval lies inside [-delta,+delta].
- Unresolved: neither condition holds. Nonsignificance alone is not equivalence.

Do not force these pairwise relations into transitive tie classes. Define an
acceptable winner set through bounded regret, retaining all candidates whose
optimality cannot be ruled out. Report a conservative regret interval for any
shortlist A as
[max(0,max_b L_b-max_a_in_A U_a), max(0,max_b U_b-max_a_in_A L_a)].

As a selection-bias audit, split E into two locked group-disjoint halves. Choose
each backbone's best head from mean performance on one half, score that fixed
head on the other, and swap. Report this as performance of a head-selection
procedure, a conservative approximation to the requested envelope, not as a
replacement for it. Large disagreements with the plug-in envelope mean the
reference needs more precision.

Training-draw resampling measures variability conditional on the available
training pool. Evaluation-group resampling measures finite test uncertainty.
For cross-dataset claims, average tasks within source and then resample source
domains; neither ten seeds nor two targets from one dataset create new domains.

## 7. Synthetic controls with known geometric mechanisms

First replicate the archived linear and quadratic controls, preserving their
generators as historical diagnostics. Add a new factorial suite with four
mechanisms: linear, quadratic interaction, smooth periodic, and local/multimodal
continuous dependence. Use paired latent draws and target noise across all
candidate embeddings in a panel.

Construct each mechanism first in 2–4 signal dimensions. Use smooth-local
functions with an explicit intrinsic scale (for example a low-frequency sine
on a bounded domain) so learnability can be verified independently of nuisance
features. Add known independent nuisance dimensions only in subsequent cells.

Each panel contains ten candidate embeddings combining signal fidelity,
invertible warps, information removal, and nuisance strength. Include a
high-fidelity nonlinear embedding and a degraded but linearly readable
embedding to create the actual ranking conflict of interest. A monotonic
signal-to-noise ladder alone does not ensure that a linear scorer must fail.

Use 20 independent confirmation seeds, n=300/600/1200, and 20,000 test draws.
Keep clean, nuisance, anisotropic/rotated, near-tie, and null cells separately
labeled. Calibrate on separate development seeds and freeze all parameters
before confirmation. Do not remove confirmation seeds whose rankings disagree.

A clean positive-control panel is eligible only when a known oracle/readout
and the reference library both attain R² >=0.5, and the candidate envelope has
at least two gaps >=0.05 R². This establishes observable signal before asking
OI to recover it. A hard nuisance cell that destroys recoverability remains a
stress test, not a failed sanity check. Random-label/null cells must produce no
systematic material ordering, and near-tie cells test uncertainty coverage.

Maintain two geometries in diagnostic output: the observed embedding and the
latent signal. Their feature counts can differ, but rows must align. Record
variance retained by PCA, nearest-neighbor signal agreement, and each head's
train/validation/test performance to distinguish representation loss,
under-regularization, and inadequate head bandwidth from a ranking failure.

## 8. Scorers, metrics, time, and success criteria

The benchmark-construction stage evaluates reference heads and two recoverability
baselines; it does not choose a new OI formula. Once it passes, compare at most
one preselected continuous OI candidate plus current continuous OI to:

1. A tuned full-embedding linear Ridge probe scored on held-out training folds.
2. Linear nested CV using the same representation options as the reference.
3. The expensive four-family nested-CV envelope as a recoverability baseline.
   This is not a theoretical upper bound on ranking quality; noisy maxima can
   rank worse than individual heads. Include per-head scores and an inner-fold
   family-selection / outer-fold evaluation sensitivity analysis.
4. Random ranking, evaluated analytically or with a large fixed permutation set.

For scorer benchmarking, every method receives the same S_r,n and no reference
evaluation data. Use common outer folds and inner training-only preprocessing
where a scorer uses cross-validation. Hash the one chosen OI configuration
before opening confirmation. Keep the thin Stiefel approach archived; choosing
it or another predictor here would be a separate method-development decision.

Report top-one selection regret and top-three opportunity regret separately:
the latter is Q_best minus the best Q in the returned three-backbone shortlist.
Use top-three mean-utility regret to detect filling a shortlist with weak models.
Report accuracy on material pairs and the number/fraction of pairs that were
resolvable. Spearman and exact winner rates are secondary, especially for
near-ties. Always publish all tasks, with source-balanced aggregation and
separate linear, nonlinear, mixed, and unresolved strata.

Time the actual deployable ranking operation from loaded embeddings to scores,
including preprocessing, target transforms, clustering, refinement, and any
resampling or probe fits it requires. Exclude one-time embedding extraction and
expensive ground-truth generation equally for all scorers. Use standalone runs,
one documented CPU/thread configuration, an excluded warmup, five alternating
timed repeats, geometric-mean cost ratios, p90 cost, and peak memory. Match score
quality and timing to the same settings. Do not compare a full pipeline's cost
to one component of another pipeline.

Proposed practical acceptance margins, to be frozen before confirmation:

- On qualified linear tasks, the upper 95% bound on OI-minus-linear mean
  top-one and top-three regret is <=0.005 R², and material-pair accuracy loses
  no more than 0.02.
- On qualified nonlinear tasks, the upper 95% bound on OI-minus-linear mean
  top-one regret is <=-0.01 R², with no >0.005 degradation in top-three regret.
  For a separately declared shortlist objective, reverse the roles of top-one
  and top-three; do not choose the objective after seeing results.
- On the broad panel, upper95 excess regret is <=0.005 for both endpoints.
- Geometric-mean total scoring time is <=0.5x the practical tuned linear probe,
  with p90 ratio <=1.0 on the intended n/d scaling grid. A <=1.0x result is
  time-competitive but does not meet the preferred speed target.

Treat these as separate quality, breadth, and speed decisions. A good mean
Spearman cannot compensate for a regret or runtime failure. With only a few
independent datasets, wide intervals imply inconclusive evidence, not a pass.
Power/precision should be estimated on qualification data to determine whether
the roster can resolve these margins before spending on confirmation.

## 9. Execution order and bounded scope

1. **Evaluator pilot:** reuse the five cached scouts on CUB leg, CUB wing, and
   LSP wrist. Correct duplicate-fold contamination and preprocessing while
   holding existing 200 evaluation rows fixed for comparability. Use each
   task's explicitly declared feasible unique-label budget. Include the clean
   synthetic families, float64 checks, and sample-ID invariants. This tests the
   evaluator, not final benchmark reliability.
2. **Inventory and precision pilot:** audit complete source labels and groups;
   prepare qualification partitions, enlarge evaluation sets, and run reference
   heads only. Freeze a ten-backbone roster from existing recorded checkpoints:
   DINOv2-small, DeiT-tiny, MobileNetV3-large, ResNet-50, ConvNeXt-tiny,
   OpenCLIP ViT-B/32, EfficientNet-B0, Swin-tiny, ViT-small/16, DenseNet-121.
   These are broader candidates, not presumed historically unseen models.
3. **Reference freeze:** publish every qualification outcome and all task
   inclusion reasons. Freeze the reference pipelines, groups, sample IDs,
   head grids, seeds, near-tie rules, and source hashes. Keep the three new
   proposed sources' confirmation partitions unopened throughout OI development.
4. **Method freeze and confirmation:** after reference learnability/precision
   and nonlinear-diversity gates pass, lock the candidate method and run the
   predeclared confirmation exactly once. All broad-panel tasks remain reported,
   including tasks that no longer satisfy their qualification tag.

For planning, 12 tasks x 10 backbones x 10 training draws x 4 head families
means 4,800 tuned reference fits for a complete one-budget panel, before inner
grid fits and before scorer evaluations. The actual eligible manifest defines
the exact count. A pilot with one target, two contrasting backbones, and two
draws supplies measured time per family; use these and extraction throughput
to publish an ETA before launching the larger panel. Do not extrapolate from
the old aggregate runtime fields or assume the expanded head library is cheap.

Use immutable stage directories for pilot, qualification, and confirmation;
atomic per-target/backbone/draw/head checkpoints; exact expected-grid checks;
source and protocol hashes excluding output/cache roots; package/environment
identities; and raw predictions with original sample/group IDs. Confirmation
must consume the locked decision hash and reject changed inputs. Tests should
target leakage, unique-label budgets, envelope uncertainty, undefined R²/rank
handling, standalone timing attribution, and resume identity. A finite result
or a green unit suite alone does not certify scientific validity.

The first actionable experiment is the corrected evaluator pilot. Its result
will determine whether the promising LSP wrist reversal survives valid head
tuning and whether benchmark expansion is justified.
