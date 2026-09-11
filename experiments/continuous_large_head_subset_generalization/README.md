# Large-panel cheap head-subset generalization

This no-refit development experiment tests whether the independently selected
quadratic-Ridge plus kNN training-CV envelope generalizes from the earlier
balanced synthetic panel to all eight recoverable targets in the frozen larger
real benchmark.

The primary pair is fixed before these subset outcomes are computed. All 15
nonempty subsets of the four archived head channels are reported only as a
mechanistic diagnostic. No post-outcome subset may be promoted, no new model is
fit, and no reserved backbone is evaluated.

Run with:

```bash
PYTHONPATH=. python -m experiments.continuous_large_head_subset_generalization.run
```

