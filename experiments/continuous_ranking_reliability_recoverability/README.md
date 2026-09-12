# Continuous ranking reliability and recoverability diagnostic

This experiment combines the two gates that must precede another continuous
OverlapIndex formulation:

1. estimate a stable best-head backbone-ranking target on the eight previously
   frozen recoverable real targets; and
2. ask whether that target is recoverable from the available labels at all.

The common downstream target is the expected held-out envelope of the frozen
linear Ridge, quadratic Ridge, distance-weighted kNN, and RBF-SVR heads trained
with 600 labels. The LSP wrist-span target uses 590 because its frozen training
split contains only 596 eligible rows. Three deterministic stratified training bootstraps and one
fixed 200-row evaluation cohort quantify fit and evaluation uncertainty. The
envelope is the maximum of each head family's mean held-out R2, rather than the
maximum of a different noisy winner on every evaluation resample.

Recoverability is measured without OverlapIndex. A three-fold nested-CV linear
Ridge score is compared with the corresponding full four-head nested-CV
envelope on the original PCA-32 embeddings. Both use exactly the same outer
folds. Budget 300 covers all eight targets; 600 excludes only LSP wrist span;
1,200 is evaluated for CUB leg span and both Pascal VOC targets, where the
source split supports it. Synthetic linear,
quadratic, and smooth-local controls verify that the nonlinear ceiling can
detect a recoverable nonlinear ranking when one exists.

This is a development diagnostic, not a method-selection or promotion run. It
does not modify public OverlapIndex or Vertebrae behavior and evaluates no
reserved backbones.

Run the stages in order after placing the hash-locked source archives described
by the parent protocols in their declared locations:

```text
PYTHONPATH=. python -m experiments.continuous_fresh_structured_opportunity_discovery.prepare
PYTHONPATH=. python -m experiments.continuous_ranking_reliability_recoverability.acquire
PYTHONPATH=. python -m experiments.continuous_real_pet_area_panel.prepare
PYTHONPATH=. python -m experiments.continuous_ranking_reliability_recoverability.prepare
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python -m experiments.continuous_ranking_reliability_recoverability.extract --resume
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
PYTHONPATH=. python -m experiments.continuous_ranking_reliability_recoverability.run --resume
```

Every stage is content-addressed and refuses to overwrite a terminal artifact.
