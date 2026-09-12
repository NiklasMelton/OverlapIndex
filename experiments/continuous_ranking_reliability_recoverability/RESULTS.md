# Continuous ranking reliability and recoverability: results

## Status

Completed and terminal-validated on 2026-09-11. This was a development-only
diagnostic. It fit no OverlapIndex variant, inspected no reserved backbone, and
promotes no method.

- Executed implementation commit: `b1f0776355ab565cdbc57aefb7526f2193e332c9`
- Protocol SHA-256: `8f84fa55a3ffc2eafd23c7449b7f99516fe9740db19a535e7dd9fe476ab450e1`
- Terminal manifest SHA-256: `dd126ea01db4a42210e6268604a859b954804c24e1da8fa0654bb5f7a931842f`
- Summary SHA-256: `5df1b4d471c150ff09f06c10d06a84c80484f24fd390ff6f38e5a690a601cbf1`

The outcome-independent capacity audit found that the frozen source splits do
not support the originally proposed 1,200 rows for CUB wing or 600 rows for LSP
wrist. Before any model outcome was computed, the protocol was corrected to use
900 available CUB-wing rows, a 590-row LSP-wrist reference, and exact
recoverability budgets only where supported. The final grid contains all eight
targets at 300 rows, seven at 600, and CUB leg plus both VOC targets at 1,200.

## What the diagnostic found

Only 3/8 best-head reference rankings passed all frozen stability gates:

- CUB leg span and CUB wing cosine were stable and entirely linear-head
  dominated.
- LSP wrist span was stable and genuinely nonlinear. Its best-head envelope
  changed the ranking materially: linear-to-envelope Spearman was 0.20, the
  linear top choice incurred 0.1395 R² regret, and individual nonlinear gains
  reached 0.2280 R².
- COCO wrist span, Pet compactness, LSP ankle span, and both VOC targets failed
  at least one uncertainty gate.

Four of the five unstable targets have nearly tied top two backbones: their
point gaps are 0.0097--0.0170 R². LSP ankle is different: its point gap is
0.0634 R², but its bootstrap rank interval is very wide (-0.4 to 0.9), showing
training/evaluation sampling instability rather than a simple near-tie.

Across all eight targets at 300 labels, the honest four-head nested envelope
improved mean rank Spearman only from 0.4750 to 0.5125, below the frozen 0.05
gate. It nevertheless reduced mean top-choice regret from 0.0460 to 0.0186 and
raised exact-best recovery from 5/8 to 6/8. On the three stable targets alone,
the improvement is clearer: Spearman 0.5667 to 0.7333, top-choice regret 0.0732
to 0.0000, exact-best recovery 2/3 to 3/3, and material-pair accuracy 0.8000 to
0.8667. The entire gain comes from correctly recovering the nonlinear LSP wrist
winner while preserving the two stable linear controls.

At 600 and 1,200 labels, the eligible stable targets are linear controls, so
the linear and four-head rankings are identical. Those rows cannot test
nonlinear recoverability at larger budgets.

The synthetic linear control passed exactly at every budget. The quadratic
control was also informative: at 300 labels the four-head envelope improved
Spearman from 0.70 to 0.90 and removed 0.1441 R² top-choice regret, reaching
perfect rank order by 600 labels. The smooth-local control was invalid as a
positive control because even its 600-label held-out reference envelope had
approximately zero R² for every candidate (-0.0016 to 0.0077). Its erratic
ranking therefore caused the combined synthetic gate to fail; it is evidence
of a deficient control construction, not evidence against nonlinear
recoverability.

## Interpretation

The experiment separates two problems that previous aggregate comparisons
conflated:

1. A scorer cannot recover a unique ordering when the downstream best-head
   target itself is dominated by sampling noise or practically tied candidates.
2. Where a stable nonlinear ranking difference exists, the frozen four-head
   evaluator can recover it from 300 labels and can materially outperform a
   linear-only evaluator.

The current eight-target panel is therefore not a sufficient general benchmark
for choosing a universal continuous OI formulation: six targets retain the
same linear and best-head order, one nonlinear target is unstable, and only one
target supplies a stable rank-changing nonlinear opportunity. The useful
signal is positive but rests on one real target plus the quadratic synthetic
control.

## Recommended follow-up

1. Re-estimate reference truth with more evaluation rows and independent
   train/evaluation resamples where the source permits it. Treat backbones whose
   best-head difference is below an uncertainty-aware material threshold as a
   tie, and report shortlist regret/pairwise ordering in addition to Spearman.
2. Replace the failed smooth-local synthetic control in a new, separately
   frozen experiment. Verify before method comparison that its held-out
   best-head envelope has positive R² and materially separated candidate
   quality; retain the successful linear and quadratic controls unchanged.
3. Expand the real panel with targets deliberately prequalified to have a
   stable nonlinear best-head advantage. Do that qualification using only the
   downstream reference evaluator, then freeze a disjoint method-evaluation
   split before comparing continuous OI candidates.
4. Only after those gates are reliable, compare continuous OI formulations to
   linear nested evaluation against the uncertainty-aware best-head target.
   The primary objective should be low selection regret and correct material
   pairs; exact rank correlation among near-ties should be secondary.
