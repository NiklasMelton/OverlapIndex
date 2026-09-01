# Continuous bounded top-five confirmation

This experiment is the single unused-seed confirmation authorized by the
bounded top-five post-outcome diagnostic. It freezes five new seeds across the
same six target families and uses the unchanged `bounded_2k` rule at `k=5`.

The broad-RFF winner is always preserved. SAVE+RFF may fill the remaining
shortlist positions only from the complete broad-RFF top-10 retrieval pool.
Unbounded filling and the component methods are retained as descriptive
comparators; only `bounded_2k` is eligible for the frozen decision.

```bash
CONTINUOUS_OI_COMMIT=$(git rev-parse HEAD) PYTHONPATH=. \
  python -m experiments.continuous_top5_bounded_confirmation.run \
  --output artifacts/continuous_top5_bounded_confirmation
```
