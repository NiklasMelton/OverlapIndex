# Bounded continuous shortlist result

Date: 2026-08-31

The frozen experiment executed from commit
`8836ed23ab283f1ba96e885fc8729224077ec605` and completed all 30 planned
new-seed panels. The decision is **fail** because one of seven frozen checks
did not pass: superiority of bounded reranking over broad RFF was not
established at the 95% interval level.

## Main result

| Method | Top-1 regret | Top-3 opportunity regret | Top-3 mean regret | Material concordance | Spearman |
|---|---:|---:|---:|---:|---:|
| broad RFF | .02334 | .00223 | .11430 | .68757 | .46170 |
| SAVE+RFF current | .13384 | .09903 | .14681 | .72617 | .56506 |
| unbounded winner-fill | .02334 | .00125 | .10811 | .75048 | .60208 |
| bounded-six reranker | .02334 | .00125 | .10732 | .75093 | .60276 |

Bounded reranking minus broad RFF top-three mean regret was `-.00699` R2,
95% interval `[-.02721, +.00861]`. The point estimate favors bounded
reranking, but the upper bound crosses zero. Bounded reranking improved over
SAVE+RFF by `-.03949`, interval `[-.05494, -.02711]`.

The bounded method preserved broad-RFF top-one regret exactly, was
noninferior on top-three opportunity regret, and improved both material
concordance and Spearman over SAVE+RFF. Its maximum family-mean degradation
relative to broad RFF was `+.01282` on triangle wave, within the frozen `.03`
limit. Linear and quadratic-product family mean regret improved by `.02150`
and `.03125`; sinusoid improved slightly rather than showing the prior
unbounded fusion's large degradation.

The mechanism therefore behaved as intended, but the evidence does not prove
that its small average shortlist-quality gain justifies replacing the simpler
broad-RFF method. Runtime remains descriptive and conservatively double-counted
at 29.34 seconds per panel versus 13.80 seconds for broad RFF.

## Verification

- `raw_results.json`: `1773505a03b886dd41a20ffde4fd5d8e7dbccfaad8cca868e994150b4ae701fe`
- `summary.json`: `6fb53de93f4940c64b609125e2cf02d15f0113bca90378a34012cc343df482f9`
- `report.md`: `5788fb39fc4baf6f9ddd72b1234a7ed74f9732af31b455b6bb5ab5400b49dfde`

Independent recomputation reproduced the summary exactly. All 28 current
source hashes match both raw and manifest identities; all terminal hashes
match; all 30 checkpoints share the same source identity; and the panel grid
contains exactly the five frozen seeds crossed with six frozen families.

One nonfatal scikit-learn ridge ill-conditioning warning occurred during a
reference-head fit. The fit and panel completed, and no selector or artifact
validation failed.
