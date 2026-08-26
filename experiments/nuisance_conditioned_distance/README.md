# Nuisance-conditioned distance experiment

This research-only package tests whether fitting a regularized distance transform on
training-fold rows repairs the nuisance geometry that reaches KMeans, balanced-median
refinement, adjacency, and held-out `score_fixed`. It does not change the public
`OverlapIndex` API.

The immutable candidate definitions, datasets, statistics, gates, archived source
identities, and stop rules are in [`protocol.json`](protocol.json). That file was
committed before candidate outcomes were inspected. All recovered panels have already
been observed in earlier work, so every result here is retrospective development
evidence. There is no manufactured "untouched" confirmation panel.

## Frozen candidates

| ID | Method | Labels used by transform | Status role |
|---|---|---:|---|
| A | raw OI, unrefined | no | exact baseline |
| B | raw OI, balanced-median refinement | no | exact refined baseline |
| C | refined OI, globally centered full OAS covariance inverse-square-root | no | candidate |
| D | refined OI, pooled-within-class diagonal OAS inverse scaling | training labels; sample-weighted residual rows | candidate |
| E | refined OI, pooled-within-class full OAS covariance inverse-square-root | training labels; sample-weighted residual rows | candidate |
| F | absolute B/E disagreement | diagnostic only | never promoted |
| G | panel-level selective capped probe | training labels | product-policy comparator, not OI |

Every conditioner is fit only on training rows. Held-out rows are transformed using
that frozen fitted state and passed to `score_fixed`; neither conditioning nor
prototypes are refit. OAS shrinkage is automatic, the condition-number cap is `1e4`,
and no hard PCA truncation is used. Pooled-within-class covariance is estimated from
sample-weighted class residual rows; it is not class-balanced under imbalance, so
larger classes contribute proportionally more residual rows. Residual weighting is
not applicable (JSON `null`) for the raw and global candidates.

For the synthetic screen, a selection panel is fixed by seed, condition, `k`, and
nuisance strength; the three nuisance families are the selectable embedding items.
Higher OI scores rank better. The screen locks one candidate for full evaluation;
Food-101 and full-only gates are deferred, and a later failure rejects that locked
candidate rather than promoting a runner-up.

The secondary robustness statistic uses only separated cases. Within each complete
seed/family/`k`/balance/shift curve, it measures candidate-minus-B designated-pair false-overlap
evidence (the mean of directed 0->1 and 1->0 evidence) at the three frozen nuisance
strengths, averages matched strata within seed, and integrates the curve on a normalized
nuisance axis. Promotion minimizes its upper paired seed-block 95% bound;
genuine-overlap cases never enter this secondary statistic.

## Staged commands

Run focused correctness tests before research:

```console
python3 -m pytest -q tests/test_nuisance_conditioning.py tests/test_nuisance_runner.py tests/test_nuisance_food101.py tests/test_nuisance_runtime_benchmark.py tests/test_nuisance_analysis.py tests/test_nuisance_reporting.py
```

Run the mechanistic smoke and frozen development screen:

```console
python3 -m experiments.nuisance_conditioned_distance.runner smoke --output artifacts/nuisance_conditioned_distance/smoke
python3 -m experiments.nuisance_conditioned_distance.runner screen --output artifacts/nuisance_conditioned_distance/screen
```

Analyze a completed stage and freeze its promotion decision:

```console
python3 -m experiments.nuisance_conditioned_distance.analysis --input artifacts/nuisance_conditioned_distance/screen --output artifacts/nuisance_conditioned_distance/screen/analysis
```

Only after parity, determinism, leakage, smoke, promotion, and independent-review gates
pass, run the frozen retrospective synthetic suite and Food-101 replay:

```console
python3 -m experiments.nuisance_conditioned_distance.runner full --output artifacts/nuisance_conditioned_distance/full_synthetic
python3 -m experiments.nuisance_conditioned_distance.food101 --promotion-decision artifacts/nuisance_conditioned_distance/screen/analysis/promotion_decision.json --output artifacts/nuisance_conditioned_distance/food101
python3 -m experiments.nuisance_conditioned_distance.runtime_benchmark --promotion-decision artifacts/nuisance_conditioned_distance/screen/analysis/promotion_decision.json --output artifacts/nuisance_conditioned_distance/runtime
python3 -m experiments.nuisance_conditioned_distance.analysis --input artifacts/nuisance_conditioned_distance/full_synthetic --food101 artifacts/nuisance_conditioned_distance/food101 --runtime artifacts/nuisance_conditioned_distance/runtime --screen-promotion artifacts/nuisance_conditioned_distance/screen/analysis/promotion_decision.json --output artifacts/nuisance_conditioned_distance/analysis
```

Long-running research commands are intentionally absent from the repository's default
unit suite. The report records any stage stopped by a frozen gate and any environment or
resource check that could not be completed. The large-budget panel measures the locked
candidate against the full probe at 64/128/256/512/640 samples per class and applies the
runtime gate at budgets of at least 128. Peak memory remains explicitly inconclusive
unless a fresh one-process-per-cell measurement is available; parent-process
`ru_maxrss` is not treated as per-cell evidence.
