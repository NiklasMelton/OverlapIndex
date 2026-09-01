# Winner-preserving reliability-gated shortlist

This experiment preserves broad RFF OI's complete first tie-group exactly.
When the unchanged OI-only reliability model predicts a benefit, only the
remaining candidates are reranked using the fixed 50/50 broad–SAVE ordinal
blend. The gate is trained leave-one-dataset-group-out on the regret benefit
of that winner-preserving alternative.

```bash
PYTHONPATH=. python -m experiments.continuous_real_winner_preserved_gate.run
```

This is a developmental diagnostic over already observed panels, not
confirmation.

