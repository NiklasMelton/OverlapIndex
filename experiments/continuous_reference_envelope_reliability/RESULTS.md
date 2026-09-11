# Continuous reference-envelope reliability audit

Status: **completed development diagnostic; reference gate failed at 100
labels**.

The final protocol SHA-256 is
`f50461867aa04a9d5ca50f39d56a0de024b36f8fd18aaedb4f2eb3d9f97b7eac`.
The read-only terminal validator passes for all 60 archived panels and the 32
new 200/300-label panels. The run performed 160 new candidate evaluations,
18,880 underlying scikit-learn estimator fits, zero OverlapIndex fits, and no
reserved-backbone evaluation.

The archived 100-label best-head reference is not reliable enough for method
promotion:

- all-target CV-envelope versus held-out Spearman: `0.2217`;
- recoverable-target Spearman: `0.4750`;
- target-level held-out seed-rank stability: `0.4300`.

The larger-label diagnostic is materially better:

- fixed 300-label CV-envelope versus held-out Spearman: `0.5875`;
- mean new-budget top-1 regret: `0.02240`;
- the three fixed 200-label unions versus 300-label held-out rankings:
  `0.9375`, `0.8250`, and `0.9375`;
- the three archived 100-label cohorts versus 300-label held-out rankings:
  `0.6500`, `0.8500`, and `0.7750`.

One 200-label union still has only `0.4125` mean CV-versus-held-out Spearman,
so training-sample variance has not disappeared. The held-out maximum over
four heads is also a winner's-curse quantity rather than deployable
performance. The conclusion is therefore not that 300 labels make the oracle
perfect; it is that the 100-label reference used by earlier continuous-ranker
experiments was too noisy for promotion claims, while a fixed 300-label panel
is a substantially more defensible target for the next validation.
