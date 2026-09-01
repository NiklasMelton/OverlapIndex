# Real-panel OI rank-aggregation result

## Decision

**FAIL.** Selecting a broad/SAVE aggregation weight from the other real dataset group did not establish an improvement over broad RFF on held-out top-five mean regret.

This is a post-outcome developmental diagnostic. It consumed the three completed real target families and fit no new backbone, head, or OI model. UTKFace age and absolute-age tasks remained in one dataset group because they share exact images and embeddings.

## Aggregate results

| Method | Spearman | Top-1 regret | Top-1 exact | Top-5 regret | Top-5 precision |
|---|---:|---:|---:|---:|---:|
| Broad RFF | 0.7810 | 0.02518 | 0.667 | 0.01739 | 0.893 |
| SAVE/RFF | 0.6924 | 0.02026 | 0.600 | 0.03659 | 0.813 |
| Current bounded | 0.7602 | 0.02518 | 0.667 | 0.02758 | 0.827 |
| Pool Borda, alpha 0.25 | 0.7924 | 0.02026 | 0.600 | 0.01703 | 0.871 |
| Pool Borda, alpha 0.50 | 0.7988 | 0.01830 | 0.600 | 0.01425 | 0.893 |
| Pool Borda, alpha 0.75 | 0.7931 | 0.02518 | 0.667 | 0.01638 | 0.887 |
| Leave-dataset-out Borda | 0.7927 | 0.03045 | 0.500 | 0.01654 | 0.887 |
| Linear probe | 0.9827 | 0.00000 | 1.000 | 0.00000 | 1.000 |

`alpha` is the broad ordinal-rank weight; `1-alpha` is the SAVE weight. Every Borda method reranked only the complete broad top-10 retrieval pool and used broad order outside it.

## Transfer result

The frozen leave-dataset-out selector chose:

- `alpha=0.75` for held-out Oxford Pet, based on the ten UTKFace panels;
- `alpha=0.50` for held-out UTKFace, based on the five Oxford Pet panels.

Its top-five regret improvement over broad was only `-0.000846`, with 95% joint-seed interval `[-0.004521, +0.002734]`. The primary improvement criterion therefore failed.

Against the current bounded rule, the leave-dataset-out estimate was `-0.011040`, but its interval `[-0.025409, +0.004992]` also crossed zero. Against the linear probe, it remained definitely worse: `+0.016544`, interval `[+0.004392, +0.031611]`.

The transferred aggregation also worsened rank-one performance relative to broad by `+0.005272`, interval `[+0.000181, +0.010560]`, and reduced exact-best coverage from `0.667` to `0.500`.

## Why a fixed blend does not transfer

The target-family results have incompatible preferred weights:

| Target family | Broad top-5 regret | Alpha 0.50 | Alpha 0.75 | Linear probe |
|---|---:|---:|---:|---:|
| Linear age | 0.00000 | 0.00799 | 0.00554 | 0.00000 |
| Absolute-age | 0.01891 | 0.00951 | 0.01146 | 0.00000 |
| Pet occupancy | 0.03326 | 0.02525 | 0.03213 | 0.00000 |

Broad RFF was already perfect for the linear-age shortlist, so every SAVE contribution damaged it. Moderate SAVE weight improved the two targets where broad ordering was weaker. Training on one group therefore selected a blend that did not transfer cleanly to the other.

The pooled `alpha=0.50` result is descriptively promising—it reduced average regret from `0.01739` to `0.01425`—but it is not independently validated, was not the frozen leave-dataset-out primary, and remained materially behind the linear probe. It should not be promoted from this result.

## Interpretation

The experiment rejects one universal static broad/SAVE weight as the immediate solution. It also reinforces a distinction between objectives:

- SAVE or a balanced blend can reduce the magnitude of rank-one mistakes while lowering exact-best frequency.
- Broad is more robust for shortlisting in the linear regime.
- Moderate blending improves shortlist ordering in the less-linear regimes but can unnecessarily disturb a correct broad ranking.

The next useful experiment is an outcome-free reliability gate. Candidate panel-level signals include broad/SAVE rank concordance, broad top-k margin, broad raw-versus-null alignment, and SAVE actual/null loss ratios. The gate should predict when broad is already reliable enough to leave untouched and when moderate within-pool blending is justified. It must be trained outside the held-out dataset and compared with always-broad and fixed-alpha baselines.

## Artifact identity

- Protocol SHA-256: `23dddda871840f8b43014f2dbd43ac9b537f8ab817dd0f0313c6a31b3eaa5fdb`
- Summary SHA-256: `ae8646ffefc21107ecfdcf7d47852ea7b8560c945fa9c7f76e12960dd59b4b08`
- Report SHA-256: `1dbf1cb40462b2482307a4cfb03df7d530a01c742ba2b0afd8d561fa56dc064e`
- Terminal manifest SHA-256: `abb10b24cb13867a4124dbd64ef393d1d1e04ab8cc33be3be3f8c209bcce5006`
- Frozen implementation commit: `1ff1449329db45429981d80b17b08a1c36185787`
- Continuous focused suite: `226 passed`
