# Continuous evaluation-null diagnostic

This frozen exploratory diagnostic holds the continuous-OI geometry,
prototypes, target cover, adjacency scorer, and best-head reference envelope
fixed while varying only the held-out permutation-null policy.

It compares negative actual loss with loss-ratio and bounded-OI ranking under
an independent four-permutation bank and nested panel-common banks of 4, 16,
and 32 permutations. The common folds and banks remove avoidable
candidate-to-candidate Monte Carlo variation.

Run with:

```bash
python -m experiments.continuous_null_diagnostic.run \
  --output artifacts/continuous_null_diagnostic
```

This is an experiment-local diagnostic. It does not change public
`ContinuousOverlapIndex` behavior.
