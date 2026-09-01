# Continuous winner-plus-fill shortlist experiment

This frozen new-seed experiment tests a structured ranking rather than an
averaged score: use the broad RFF characteristic-kernel winner for the first
slot, then fill the remaining order with the smoother SAVE+RFF Continuous OI
ranking.

The experiment is synthetic and exploratory. Its primary target is the mean
quality of a three-backbone shortlist against the frozen four-head envelope.
A pass permits a shared-computation implementation experiment and a real
continuous-target backbone panel; it does not authorize public integration.

Run after committing the frozen implementation:

```bash
CONTINUOUS_OI_COMMIT=$(git rev-parse HEAD) PYTHONPATH=. python -m \
  experiments.continuous_winner_fill_shortlist.run \
  --output artifacts/continuous_winner_fill_shortlist
```

