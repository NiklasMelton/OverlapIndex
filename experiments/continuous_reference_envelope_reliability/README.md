# Continuous reference-envelope reliability audit

This experiment asks whether the four-head held-out reference envelope used in
the continuous ranking studies is itself stable at the available label
budgets. It is diagnostic only. A passing diagnostic gate cannot promote a
ranker or authorize confirmation.

The archive-only part reads all 20 target channels and all 60 panels from the
large structured benchmark. It compares the training-only CV envelope with
the held-out envelope, records head-family agreement/confusion and score bias,
and measures target-level stability across the three archived 100-label
replicates. Archived prediction vectors were not retained, so evaluation-row
bootstrap is explicitly unavailable for this part. Target-block and
five-dataset-cluster percentile intervals are reported instead.

The budget part uses the eight source-selected frozen recoverable targets. It
reuses the three archived 100-label cohorts, then fits the unchanged
three-fold training-only four-head grids for all three fixed 200-label unions
and the fixed 300-label union. Row order is frozen by replicate seed
`4103, 4201, 4303`, preserving within-cohort order. PCA is fitted on each
training cohort only. The common 200-row evaluation cohort is used only after
all training/CV fits finish. New budget checkpoints retain each head's
200-row prediction vector, enabling paired evaluation-row bootstrap with the
same resampled row IDs across candidates, heads, cohorts, and budgets.

The exact new workload is 160 candidate rows, 640 head-family calls, 160 PCA
fits, and 18,880 underlying sklearn estimator fits. `n_jobs=1` is fixed and
the runner records thread environment variables. No OverlapIndex fit and no
reserved backbone evaluation occurs.

Run only after the protocol and focused tests have been reviewed:

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
LOKY_MAX_CPU_COUNT=1 \
/Users/niklasmelton/Library/Caches/pypoetry/virtualenvs/vertebrae-XeUus8FY-py3.9/bin/python \
-m experiments.continuous_reference_envelope_reliability.run
```

`--resume` validates the protocol/source hash identity before reusing any
checkpoint. Final JSON, Markdown, checkpoint, and terminal-manifest writes
are atomic. A source or protocol mismatch fails closed.
