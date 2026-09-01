# Expanded continuous-OI reliability gate

This analysis applies the exact earlier four-feature, ridge-penalty-1,
zero-threshold reliability gate to four independent dataset groups. Every
group is first restricted to the common ten-backbone roster. The two UTKFace
targets remain one leave-dataset-out group because they share images and
embeddings.

No feature, threshold, blend weight, or estimator hyperparameter is changed
after the earlier two-group result. The analysis is still developmental:

```bash
PYTHONPATH=. python -m experiments.continuous_real_reliability_gate_expanded.run
```

