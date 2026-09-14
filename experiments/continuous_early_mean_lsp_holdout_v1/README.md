# Early-mean LSP fresh-evaluation benchmark

This experiment tests the frozen `early_mean` representation and continuous
ranking candidates on a new 300-image evaluation cohort for LSP wrist span.
The target was previously established as a stable rank-changing nonlinear
task, but none of the evaluation images selected here appeared in that prior
target evaluation.  The training pool is reused; the evaluation cohort and all
six intermediate-layer/pooling embeddings are new.

The best-head truth is the held-out maximum over linear Ridge, quadratic kernel
Ridge, distance-weighted kNN, and exact RBF-SVR.  All head hyperparameters and
continuous ranking methods are selected using training rows only.  The
`early_mean` result is primary.  The other five views are negative controls and
cannot replace it after outcomes are observed.

Run in order from a clean committed tree:

```bash
PYTHONPATH=. python -m experiments.continuous_early_mean_lsp_holdout_v1.prepare
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python -m experiments.continuous_early_mean_lsp_holdout_v1.extract
PYTHONPATH=. python -m experiments.continuous_early_mean_lsp_holdout_v1.run
```

This is an experiment-only benchmark.  It does not modify public OverlapIndex
or Vertebrae behavior and cannot by itself support library integration.

