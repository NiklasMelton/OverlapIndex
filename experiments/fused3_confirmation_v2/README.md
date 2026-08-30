# FUSED3 confirmation V2

This experiment is the separately versioned untouched confirmation for the
development-locked `FUSED3` backbone selector. It does not continue the failed
M50 candidate lock. Instead it binds two independent histories:

1. the five-dataset confirmation design that was selected before FUSED3 or any
   confirmation embeddings existed; and
2. the immutable FUSED3 screen and complete Food-101 decision that selected
   `FUSED3` with no runner-up reselection.

The panel is CIFAR-10, STL-10, GTSRB, torchvision FGVC Aircraft (100 variants),
and DTD partition 1, using the same ten frozen backbones. For each of five
sampling seeds, budgets 32 and 64 are nested within class. The sampled training
cohort is used both for selector scoring and for fitting the downstream heads;
evaluation uses a distinct fixed official-test cohort of exactly 20 rows per
class. Replicates are sampling-seed stability checks and may overlap.

Only `FUSED3` and `LP-FULL` are ranked. No other candidate can be introduced or
selected after outcomes. Linear, quadratic, 15-NN/cosine, and RBF reference
heads are frozen in `protocol.json` and `recipes.py`.

## Stages

1. Verify the dual lineage and source identity.
2. Download/audit the exact five datasets and select all cohorts before model
   extraction.
3. Extract only the union of frozen cohort rows with the exact Food backbone
   recipes; hash every dataset, recipe, checkpoint, preprocessing surface, and
   embedding matrix into a separate audited registry.
4. Run focused tests, an outcome-redacted structural smoke, an outcome-blind
   resource preflight, and an independent pre-outcome review.
5. Execute all 500 paired backbone panels with atomic checkpoints and no
   interim statistical inspection.
6. Analyze once with the predeclared hierarchical bootstrap and obtain an
   independent final review.

## Reproduction commands

All commands run from the OverlapIndex repository root. The full outcome run
is forbidden until the frozen sidecar, audited registry, structural-smoke
artifact, and independent post-smoke review all pass.

Focused verification:

```bash
PYTHONPATH=. pytest -q \
  tests/test_fused3_confirmation_lineage.py \
  tests/test_fused3_confirmation_datasets.py \
  tests/test_fused3_confirmation_recipes.py \
  tests/test_fused3_confirmation_runner.py \
  tests/test_fused3_confirmation_resource_preflight.py \
  tests/test_fused3_confirmation_statistics.py \
  tests/test_fused3_confirmation_analysis.py
```

Write the immutable lineage lock after the protocol and code identity are
frozen:

```bash
PYTHONPATH=. python3 -m experiments.fused3_confirmation_v2.prepare lock \
  --output artifacts/fused3_confirmation_v2/inputs/lineage_lock.json
```

Download the five exact torchvision datasets explicitly. This is the only
network-enabled input stage:

```bash
PYTHONPATH=. /Users/niklasmelton/code/vertabrae/.venv/bin/python \
  -m experiments.fused3_confirmation_v2.prepare download \
  --data-root artifacts/fused3_confirmation_v2/inputs/raw \
  --allow-download
```

Hash every raw source byte and extract the frozen cohort union using only the
already-cached model snapshots:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  /Users/niklasmelton/code/vertabrae/.venv/bin/python \
  -m experiments.fused3_confirmation_v2.prepare extract \
  --data-root artifacts/fused3_confirmation_v2/inputs/raw \
  --input-root artifacts/fused3_confirmation_v2/inputs \
  --lineage-lock artifacts/fused3_confirmation_v2/inputs/lineage_lock.json
```

Run the outcome-redacted structural smoke:

```bash
VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
PYTHONPATH=. python3 -m experiments.fused3_confirmation_v2.runner \
  --registry artifacts/fused3_confirmation_v2/inputs/audited_registry.json \
  --lineage-lock artifacts/fused3_confirmation_v2/inputs/lineage_lock.json \
  --output artifacts/fused3_confirmation_v2/smoke --smoke
```

Run the distinct fresh-process, outcome-blind resource preflight. Its wall,
CPU, and peak-RSS measurements include imports, validation, the excluded
warm-up, and the measured structural call. They are descriptive only and
cannot change any gate, recipe, candidate, or execution decision:

```bash
VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
PYTHONPATH=. python3 -m experiments.fused3_confirmation_v2.resource_preflight \
  --smoke artifacts/fused3_confirmation_v2/smoke \
  --registry artifacts/fused3_confirmation_v2/inputs/audited_registry.json \
  --lineage-lock artifacts/fused3_confirmation_v2/inputs/lineage_lock.json \
  --output artifacts/fused3_confirmation_v2/resource_preflight
```

Only after an independent review validates both the smoke and resource
preflight artifacts may the complete frozen panel be run or resumed:

```bash
VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
PYTHONPATH=. python3 -m experiments.fused3_confirmation_v2.runner \
  --registry artifacts/fused3_confirmation_v2/inputs/audited_registry.json \
  --lineage-lock artifacts/fused3_confirmation_v2/inputs/lineage_lock.json \
  --output artifacts/fused3_confirmation_v2/full --resume
```

Analyze exactly once after the terminal full manifest validates:

```bash
PYTHONPATH=. python3 -m experiments.fused3_confirmation_v2.analysis \
  --input artifacts/fused3_confirmation_v2/full \
  --output artifacts/fused3_confirmation_v2/analysis
```

This package is research-only. It does not modify Vertebrae or public
`OverlapIndex` behavior.
