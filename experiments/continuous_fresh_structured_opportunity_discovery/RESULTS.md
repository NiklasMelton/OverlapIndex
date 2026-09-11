# Results: fresh structured continuous opportunity screen

Status: completed prospectively under protocol SHA-256
`c414a4809f816192f589d07b3e95cd62abff2ff37d8970903ec95e70fcc9de8f`.
The experiment-level gate did not pass because only one of two datasets
contained an admitted target. No reserved backbone or OverlapIndex method was
evaluated.

## Result

- LSP `torso_axis_sine` was a rank-changing nonlinear opportunity: median
  best-head R2 0.2463, median envelope span 0.2218, material nonlinear
  prevalence 0.467, and LP winner-change rate 0.333. RBF-SVR was the unique
  held-out winner in 8/15 candidate rows, linear ridge in 6/15, and kNN in
  1/15.
- CUB `beak_tail_axis_sine` was recoverable (median best-head R2 0.2914) and
  frequently won by RBF-SVR (12/15), but LP still selected an envelope-optimal
  backbone in every replicate. Nonlinear readout advantage therefore did not
  create a backbone-ranking opportunity.
- CUB normalized beak-tail span and LSP knee extension were recoverable LP
  controls: their envelope rankings did not materially differ from LP.
- Both cosine orientation channels were nonlinear but unrecoverable at the
  frozen sample/PCA budget, so ranking changes there are not useful evidence.

## Interpretation

The prospective screen confirms the earlier distinction between nonlinear
prediction and nonlinear *backbone-ranking opportunity*. A task can prefer a
nonlinear head for many candidates while LP preserves the envelope ordering.
LSP torso vertical orientation supplies one clean new opportunity dataset, but
CUB does not. The correct next action is to freeze and screen another
independent dataset; it is not permissible to retarget CUB or relax admission
thresholds after seeing these results.

Canonical artifacts are in
`artifacts/continuous_fresh_structured_opportunity_discovery/results/`.
