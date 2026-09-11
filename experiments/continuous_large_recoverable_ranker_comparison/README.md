# Recoverable-target continuous ranker comparison

This frozen development experiment evaluates the two established continuous
ranking surfaces on all eight recoverable targets from the 20-target larger
benchmark. Four targets have predeclared rank-changing nonlinear opportunity;
four are recoverable LP-dominated controls.

The methods are fixed before selector outcomes:

- training-only linear Ridge CV (practical baseline);
- prototype-free broad RFF alignment;
- prototype-based SAVE+RFF Continuous OI;
- the training-only four-head CV maximum (diagnostic ceiling only).

No reserved backbone is evaluated and no target or method may be selected or
replaced after the results. The held-out best-head envelope is used only to
score the already-fixed rankings.

```bash
PYTHONPATH=. python -m experiments.continuous_large_recoverable_ranker_comparison.run
```
