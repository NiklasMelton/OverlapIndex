# Continuous evaluation-null diagnostic result

The frozen diagnostic completed. It held each fitted geometry, target cover,
prototype model, and scorer fixed while replaying held-out target permutations.
It also used one common cross-fit partition per panel and nested common
permutation banks of 4, 16, and 32 permutations.

## Identity and validation

- Frozen implementation commit: `7634e05`
- Protocol SHA-256:
  `d08dd3990957c5af66aba4ddbe1afdacf45745d20b90d2305de24f42ccb54716`
- Runner SHA-256:
  `a5e69b15d536197f442838e3362b777059625743849fbf8e124d3e68de20ba9c`
- Parent raw-results SHA-256:
  `811a853635cdc4ccda726a8c2a68e2cbc3cbe83ff044d2c28c9077cf4143d24d`
- Raw-results SHA-256:
  `d044251b6c7bec73842315b642498db179b525a85b14efc473914fef331ea6b4`
- Summary SHA-256:
  `ff42103376ace703ba17e23ace51bdae05eecf9e47201a9eb74db44cfad62721`
- Manifest SHA-256:
  `5f789745c78a18560ff4271514c4538fdc203207fc55484b6f45d0d4db55c1e4`
- Complete grid: 18 panels, 288 candidates, 1,152 selector rows,
  4,608 fitted fold models, and 18 atomic checkpoints.

Every source hash and terminal-file hash was verified. Rebuilding the summary
from the raw panels matched byte-for-structure. Every common-4 bank was the
exact prefix of common-16 and common-32. The full test suite passed with 350
tests, 14 expected skips, and 34 existing under-prototyping warnings.

## Result: null variance is not the main failure

No geometry passed the frozen variance-repair rule. Replacing the independent
four-permutation estimate with 32 common permutations changed mean periodic
Spearman by only -0.014 to +0.033, far below the required +0.15. Increasing
precision therefore did not recover the periodic ordering.

| Geometry | Parent OI rho / regret | Common-fold candidate-4 | Common-16 | Common-32 | Negative actual loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAVE | .591 / .153 | .503 / .224 | .506 / .190 | .523 / .190 | .572 / .129 |
| RFF mean | .382 / .129 | .395 / .128 | .399 / .117 | .400 / .117 | .480 / .110 |
| SAVE + RFF mean | .551 / .106 | .621 / .073 | .615 / .065 | **.615 / .065** | .578 / .068 |
| SAVE + RFF mean + residual | .545 / .108 | .578 / .076 | .593 / .076 | .593 / .111 | .561 / .076 |

The parent column preserves the old candidate-specific folds and four-
permutation policy. The other columns use the new common folds. Consequently,
parent-to-new differences are descriptive and cannot be attributed solely to
the null bank.

Common-16 was already close to common-32. Across geometries, its mean absolute
null-loss difference from common-32 was .0032--.0039 and its mean absolute OI
difference was .0020--.0024. The per-fold null standard error fell from about
.023--.027 with four permutations to .0087--.0102 with 32. Precision improved,
but the relevant rank failure remained.

## The diagnostic supports a view-specific estimand mismatch

The clearest evidence is triangle wave:

| Geometry | Negative-actual-loss rho | Common-32 loss-ratio rho | Common-32 bounded-OI rho |
| --- | ---: | ---: | ---: |
| SAVE | **.606** | .256 | .097 |
| RFF mean | **.260** | -.100 | -.071 |
| SAVE + RFF mean | -.018 | .030 | .121 |
| SAVE + RFF mean + residual | .169 | .294 | **.320** |

For SAVE and RFF mean, useful triangle quality ordering exists in the actual
held-out adjacency loss and is removed by dividing by the candidate's null
loss. The common-32 null loss is negatively correlated with the best-head
envelope on triangle panels (mean rho -.557 for SAVE and -.682 for RFF mean):
better candidates receive smaller null denominators, which penalizes them and
cancels their actual-loss improvement.

This is not a universal indictment of normalization. For the two multiview
geometries, normalization improved triangle ordering relative to negative
actual loss. Their remaining weakness is therefore a geometry/view-
aggregation problem, not Monte Carlo null variance. The frozen decision
correctly records `null_estimand_mismatch_supported` because the systematic
cancellation is decisive for the individual SAVE and RFF views.

## Secondary finding: paired evaluation is useful

The common-fold/common-bank two-block method reached mean rho .615, mean
regret .065, exact-best .722, and within-.01 .833. On these same development
panels, the frozen linear-probe baseline was rho .354, regret .282, exact-best
.444, and within-.01 .500. This strengthens the case for paired folds and a
fixed common null bank in future experiments, but it is retrospective evidence
on exhausted synthetic panels and is not a confirmation result.

The three-block method shows why top-selection metrics still need care: its
common-16 and common-32 mean rho were both .593, but mean regret changed from
.076 to .111 because a small score change altered a panel winner. More null
draws reduce Monte Carlo error without guaranteeing a stable argmax when top
candidates are close.

## Recommendation

Stop increasing the permutation count as the primary remedy. Use 16 or 32
panel-common permutations and common folds as a stable experimental baseline,
then test a fixed-geometry target-continuity normalization:

1. fit geometry, target cover, prototypes, and adjacency once per fold;
2. compute actual local target-coherence loss on that fixed structure;
3. permute target values on the same fixed geometry and supports, without
   letting candidate-specific null scale erase improvements in actual loss;
4. preserve separate SAVE, RFF-mean, and residual score channels through the
   scoring stage before applying a frozen aggregation rule;
5. evaluate the best-head envelope on new seeds/generators before any real-
   data confirmation.

Negative actual loss alone is not a universal replacement: it helps SAVE and
RFF on triangle but is weaker for other views/families. The next change should
therefore be a scoring/null design experiment, not another whitening,
projection-rank, or prototype-count sweep.
