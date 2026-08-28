# M50 backbone-ranking follow-up

This private experiment tests a product-scoped claim: can one fixed OI-based selector
rank transfer-learning backbones about as well as the exact full linear probe for
linear downstream heads, better for quadratic, kNN, and RBF heads, and at comparable
end-to-end cost? It deliberately does **not** use distribution-shift robustness or
generic overlap calibration as product vetoes.

The immutable contract is [`protocol.json`](protocol.json). The experiment is staged:

1. Re-run the archived Food-101 panel as retrospective development evidence with a
   directly executed full probe in the same hash-seeded cyclic schedule. Because
   600 panels is not divisible by seven methods, per-position counts differ by at
   most one; the artifact records and validates those counts.
2. Estimate the raw/M50 × refinement factorial. Only sample-weighted M50 can be
   selected from Food; class-balanced variants are diagnostic because Food sampling
   is already balanced per class.
3. If RBF is the only inconclusive gate, evaluate the frozen 50/50 raw+M50 rank
   fusion. Evaluate the capped-probe guardrail only as a separate product policy.
4. Execute the six primitive components needed by any possible locked chain at
   64/128/256/512/640 examples per class in each baseline, nonlinear, and nuisance
   arm. F/G costs are conservative sums of their unique measured components rather
   than falsely labeled direct estimator timings. Every arm must pass the resource
   gates separately.
5. Preserve an exact five-dataset confirmation plan as v1 planning evidence. V1 does
   not execute confirmation because those embeddings and a frozen evaluator do not
   exist; Food, CIFAR-100 coarse, Oxford Pets, and prior synthetic panels cannot be
   relabeled as untouched evidence.

Version 1 ends at a retrospective development/resource lock with status
`locked_for_future_confirmation`; it cannot make a confirmed product claim or later
resume confirmation under a changed code identity. A future versioned confirmation
protocol must hash-link the exact v1 lock, analysis manifest, protocol/code identities,
selected recipe and chain, resource summary, and planned registry. It may not reselect,
retune, or reinterpret the predeclared gates.

The final lock records the closed `selected_chain_recipe_bindings` object and its
`selected_chain_recipe_bindings_sha256`; the identical canonical object is stored as
`selected_chain_recipe_bindings.json` and hash-bound by the analysis manifest. This
binds every primitive and derived F/G recipe plus deterministic seed rules. A future
v2 must link those exact v1 bytes rather than reconstructing or substituting a recipe.

## Candidate table

| ID | Geometry | Refinement | Role |
|---|---|---:|---|
| A | raw direct upstream OI | off | exact control |
| B | raw direct upstream OI | on | exact control |
| M0-SW | pooled within-class MAD, gamma 0.5, row-weighted | off | promotable factorial candidate |
| M1-SW | pooled within-class MAD, gamma 0.5, row-weighted | on | prior promotable candidate |
| M0-CB | pooled within-class MAD, gamma 0.5, class-balanced | off | diagnostic |
| M1-CB | pooled within-class MAD, gamma 0.5, class-balanced | on | diagnostic |
| LP-FULL | exact five-fold normalized logistic probe | n/a | comparator |
| LP-CAPPED-2048 | the same probe capped at 2,048 panel rows | n/a | guardrail timing primitive |
| F | fixed equal fractional-midrank fusion | locked setting | conditional algorithmic selector |
| G | panel-wide selective capped probe | n/a | product policy only |

Conditioning is fitted on training-fold rows only. Held-out rows receive the frozen
transform and `score_fixed`; neither prototypes nor conditioning are refit. A and B
instantiate upstream `OverlapIndex` directly. The experiment-local M50 transform uses
elementwise diagonal scaling and must prove equivalence to the previously frozen dense
diagonal implementation before any outcomes are run.

For G, the trigger is panel-wide. If any backbone crosses either frozen training-only
diagnostic threshold, every backbone in that complete panel is reranked by
LP-CAPPED-2048. G never mixes probe and OI scores backbone by backbone, and it is never
reported as an algorithmic OI improvement. Triggered capped-probe panels and
untriggered M panels preserve the archived absolute-tolerance `1e-12` tie averaging;
an untriggered F panel preserves F's exact-equality, frozen-order winner.

## Authorization and expected duration

Do not execute development or runtime outcomes until the focused suite, protocol/source
freeze, independent pre-outcome review, redacted structural smoke, and independent
smoke-artifact review all pass. Long-running stages remain outside the default unit
suite and use resumable, identity-locked checkpoints. With the three-arm resource
grid, development plus runtime is expected to take roughly 8–12 hours on the current
machine; this is an estimate, not an outcome or a gate.

Warm-up calls are excluded from every score, wall-time, and CPU-time summary. Fresh
process peak RSS conservatively includes the warm-up by design and is labeled that way
in raw artifacts and reports.

Food determinism is checked once per selector ID (including LP-CAPPED-2048) using a
clock-free warm-up/measured comparison on that ID's first eligible panel in the frozen
schedule. The canonical panel identity is serialized and reused unchanged on resume;
this is not a claim that every one of the 600 panels is executed twice.

Run commands from the OverlapIndex repository root with the archived Python 3.9
environment on `PATH` or replace `python`/`pytest` below with its absolute executables.

```bash
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1

PYTHONPATH=. pytest -q \
  tests/test_m50_backbone_candidates.py \
  tests/test_m50_backbone_manifest.py \
  tests/test_m50_backbone_food101.py \
  tests/test_m50_backbone_runtime.py \
  tests/test_m50_backbone_statistics.py \
  tests/test_m50_backbone_reporting.py \
  tests/test_m50_backbone_analysis.py

PYTHONPATH=. python -m experiments.m50_backbone_ranking.food101 \
  --smoke --output artifacts/m50_backbone_ranking/smoke

PYTHONPATH=. python -m experiments.m50_backbone_ranking.food101 \
  --output artifacts/m50_backbone_ranking/development

PYTHONPATH=. python -m experiments.m50_backbone_ranking.analysis development \
  --input artifacts/m50_backbone_ranking/development \
  --output artifacts/m50_backbone_ranking/development_analysis

PYTHONPATH=. python -m experiments.m50_backbone_ranking.runtime_benchmark \
  --development-decision artifacts/m50_backbone_ranking/development_analysis \
  --output artifacts/m50_backbone_ranking/runtime

PYTHONPATH=. python -m experiments.m50_backbone_ranking.analysis finalize \
  --provisional artifacts/m50_backbone_ranking/development_analysis \
  --runtime artifacts/m50_backbone_ranking/runtime \
  --output artifacts/m50_backbone_ranking/analysis_final

```

V1 deliberately exposes no outcome-producing confirmation command. Its confirmation
module only proves the placeholder and even a structurally ready test registry fail
closed without creating an artifact. The future v2 protocol must copy and hash-link,
not mutate, the v1 registry plan. No command in this package modifies Vertebrae, the
public OverlapIndex API, or archived evidence.
