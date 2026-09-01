# Independent box-occupancy panel result

## Status

The frozen GTSRB and FGVC-Aircraft panels completed exactly: 2 dataset groups,
5 disjoint training cohorts per group, 10 shared backbone candidates, and 100
candidate evaluations. This stage is descriptive input to the expanded
leave-dataset-out reliability analysis; it has no promotion decision by itself.

| Method | Spearman | Top-1 regret | Top-5 regret | Top-5 precision |
|---|---:|---:|---:|---:|
| broad RFF OI | 0.4679 | 0.02349 | 0.02874 | 0.680 |
| SAVE OI | 0.4376 | 0.04819 | 0.03226 | 0.680 |
| bounded 2k OI | 0.5612 | 0.02349 | 0.02406 | 0.740 |
| linear probe | 0.9600 | 0.00158 | 0.00229 | 0.940 |

Bounded OI improved broad OI's shortlist ordering without changing its top
winner, but neither OI ranking was close to the linear probe. Across the 100
candidate panels, the best reference head was linear ridge 70 times and RBF
SVR 30 times; quadratic ridge and kNN never won.

## Reproducibility

- Frozen implementation commit: `52c2a4b39dbd0888f9730828d5e37167ab4717f9`
- Protocol SHA-256: `e3229f3d084cce914209016e10c4106286d8211e7b7b43198165ae40f17571b3`
- Registry SHA-256: `a6fd86518562cca32851e8ddf2ae0d874b61c32433614ba3ee91a8bbc86861b9`
- Cohort SHA-256: `485135316d576a20309a3cd50d209118f410d1b5f7a73bc629b502f11b5980c5`
- Raw results SHA-256: `e3d604f28f3160708a9d6158eac4fec1ebc1d18caa74dbc0ee016acd4a06722b`
- Summary SHA-256: `f032d2916a926620a6916d9765b92d4838b833cd749224b7458d0bdad4e564cb`
- Manifest SHA-256: `722a09c07b6f4118c71a5b7d21c710fcd20dfc4068146794984c0d6714f97fa8`
- Terminal reconstruction: exact from all 10 canonical checkpoints

