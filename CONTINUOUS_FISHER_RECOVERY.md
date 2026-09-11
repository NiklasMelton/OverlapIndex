# Continuous Fisher experiment recovery

Recovery date: 2026-09-11

## What happened

The experiment checkout was created at
`/private/tmp/OverlapIndex-continuous-fisher`. A second reconstruction was
created at `/private/tmp/OverlapIndex-continuous-fisher-restored` on
2026-09-08. Both paths were on macOS temporary storage and disappeared after
the 2026-09-09 reboot. Searches of Codex task histories, command records,
application logs, Git worktree records, the Trash, Spotlight, and local APFS
snapshots found no user or Codex deletion command. The evidence is consistent
with operating-system temporary-directory cleanup.

The durable branch `codex/continuous-fisher-oi` survived at commit
`10d068eb7a77fa4ebadf2eeeb0b1d8eb38c3e197`. This recovery branch starts at
that commit.

## Recovered material

This commit restores the complete-patch experiment entries that were present
in the final recorded `git status --short`, plus the independently recovered
reference-envelope result note. Recovered Python files pass syntax
compilation. Outcome-bearing tests cannot be replayed because their ignored
embedding and result artifacts were stored only in the erased temporary
checkout.

The restored entries are:

- `experiments/continuous_coco_pose_opportunity_replication/`
- `experiments/continuous_fresh_structured_opportunity_discovery/`
- `experiments/continuous_large_head_subset_generalization/`
- `experiments/continuous_large_recoverable_ranker_comparison/`
- `experiments/continuous_large_structured_opportunity_benchmark/`
- `experiments/continuous_real_opportunity_stratified_discovery/`
- `experiments/continuous_reference_envelope_reliability/RESULTS.md`
- `experiments/continuous_residual_anchor_n300/`
- `experiments/continuous_rff_broad_fusion_scaling/`
- `experiments/continuous_rff_convergence_diagnostic/`
- `experiments/continuous_rff_postprojection_fusion/`
- `experiments/continuous_smallnorb_head_discovery/`
- `experiments/continuous_structured_head_discovery/`
- `tests/test_continuous_coco_pose_opportunity_replication.py`
- `tests/test_continuous_large_head_subset_generalization.py`
- `tests/test_continuous_large_recoverable_ranker_comparison.py`
- `tests/test_continuous_large_structured_opportunity_benchmark.py`
- `tests/test_continuous_residual_anchor_n300.py`

## Material that could not be recovered byte-for-byte

The final status recorded six additional untracked entries. Their edits are
partly visible in tool transcripts, but at least one initial runner payload was
truncated at 64 KB and no complete final file was persisted in Git. Partial
replays were deliberately excluded from this branch because their source
identity did not match the recorded hashes.

- `experiments/continuous_fixed_prototype_readout/`
- `experiments/continuous_information_ladder_n300/`
- `experiments/continuous_stable_prototype_readout_n300/`
- `tests/test_continuous_fixed_prototype_readout.py`
- `tests/test_continuous_information_ladder_n300.py`
- `tests/test_continuous_stable_prototype_readout_n300.py`

Recorded final identities that remain useful for detecting any future exact
copy are:

- stable-prototype runner SHA-256:
  `7b558908734dc04381580095bff2d9a5d0cfeb67667c820843481053537e7137`
- stable protocol SHA-256:
  `b0abf8a564287044a2437cf45dacdd0907498e78a1d5598c09579ff4442ec9da`
- information-ladder runner SHA-256:
  `0b0e87808b37e5b907c14375ccae3a85278a865fd243d9766a2aab4cb3bea14a`
- information-ladder protocol SHA-256:
  `005f60bd871cff01ce8d97235ac2718885ca20a638bf5f7fe3e7d14f71a622dd`
- information-ladder protocol-sidecar file SHA-256:
  `774273a8078c868e8a8b9dc1405c30a713d67e2cedc088a9c60ac852f124cc51`

The outcome conclusions themselves remain recorded in the surviving Codex
task history. In particular, the information-ladder result localized the main
loss before prototype readout: SAVE and RFF each lost roughly 0.24 held-out R²
relative to raw linear Ridge; fusion reduced but did not remove that gap, and
prototype soft normalization added further loss. The residual-anchor result
also remained negative: the raw linear anchor averaged 0.184350 held-out R²
and 0.7625 Spearman, while all tested residual variants were worse and the
pipeline was about 12.4 times slower.

## Verification boundary

This is a forensic source recovery, not a claim that the erased terminal
artifacts were reproduced. Do not regenerate or reinterpret frozen outcomes
from this branch without creating a new, explicitly versioned protocol.
