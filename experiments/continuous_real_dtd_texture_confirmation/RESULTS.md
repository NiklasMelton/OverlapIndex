# DTD texture-gradient new-target result

The frozen all-development winner-preserving gate **failed** its transfer criterion on this target. The result is nevertheless informative: the hard winner-preservation rule transferred exactly, while the learned decision about when to reorder the remaining shortlist did not transfer reliably.

## Frozen result

| Method | Spearman | Top-1 regret | Top-1 exact | Top-5 mean regret | Top-5 opportunity regret |
|---|---:|---:|---:|---:|---:|
| broad RFF OI | 0.4400 | 0.07937 | 0.200 | 0.04086 | 0.01546 |
| fixed winner-preserved Borda | 0.4140 | 0.07937 | 0.200 | 0.03431 | 0.02000 |
| fixed learned gate | 0.3735 | 0.07937 | 0.200 | 0.03812 | 0.02000 |
| linear probe | 1.0000 | 0.00000 | 1.000 | 0.00000 | 0.00000 |

The learned gate improved top-five mean regret by 0.00274 R2 at the point estimate, but the paired 95% interval was `[-0.01625, 0.01205]`; its upper bound was not below zero. Top-five opportunity regret was 0.00454 worse at the point estimate with interval `[-0.00650, 0.02012]`, exceeding the frozen noninferiority margin. Top-one regret and exact-best coverage remained bit-exact to broad OI in all five panels, as required by construction.

The gate selected the alternative for seeds 4103, 4201, 4303, and 4409, then selected broad for 4507. The regret-optimal choice was the alternative for 4103, 4201, 4303, and 4507, and broad for 4409. Thus the frozen four-feature model made exactly the two consequential boundary decisions incorrectly.

## Why the linear probe is exact here

The reference is the best-head envelope, but the linear head won 49 of 50 backbone-by-seed cells; quadratic won the remaining cell, with no kNN or RBF wins. Consequently, the linear-probe ranking is effectively the envelope ranking on this particular target. This panel is useful as a linear-target stress test but provides almost no evidence about nonlinear-head ranking.

## Interpretation

The result does not invalidate winner-preserved aggregation: the fixed Borda alternative still reduced mean shortlist regret, and the winner-preservation invariant completely eliminated top-one damage. It does invalidate the stronger claim that the current four OI-only reliability features generalize well enough to choose between broad and Borda on a new target.

The next iteration should therefore keep the hard winner-preserving construction but should not promote the learned gate. Additional new targets should be chosen before outcomes to include both a predominantly linear envelope and a genuinely nonlinear envelope. The primary question is whether a fixed, ungated winner-preserving aggregation is robust enough across those targets, or whether gate reliability requires richer OI-only diagnostics and substantially more independent development groups.

## Artifact identity

- Frozen implementation commit: `045a19dabee252a2e1a2c25350850f98bd3901ac`
- Protocol SHA-256: `d968f22cb0fbb2d51d2f902fee007ef86af6883b0637b1cb6a62860d28ebd3dd`
- Raw results SHA-256: `4d977dddddd338e7d1af839012b70b3e50fa3ff097332c4b7f8521c0c8febd8d`
- Summary SHA-256: `25e77ea66de5a21a3ed18700363df5120570e920a914f6304dc4f5f29a0667e9`
- Report SHA-256: `9d9f4c93aefe2d3d373c34f65b2cf4ccb4cbcfa4d782adf7b612a7b2330838bb`
