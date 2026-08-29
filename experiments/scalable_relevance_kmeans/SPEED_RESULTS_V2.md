# Speed-focused scalable relevance OI: v2 checkpoint

Recorded 2026-08-29 on branch
`codex/experiment-m50-fast-conditioning`, from pre-checkpoint HEAD
`7c92d60042c5efd427b392716e986f760d5c390e`.

This is retrospective Food-101 development evidence. It is not an untouched
confirmation and it does not change public `OverlapIndex` behavior.

## Frozen identity

- Corrected protocol SHA-256:
  `ad00716130c3a41d0e0fbbf159f75cd6006bafb1fe0a36d3735c124ce99ae6c2`
- Executed code identity SHA-256:
  `b0e567940ff3f58ef708c2f63b005a71c833b6422ebd0021a34b2a417503717d`
- Valid raw results SHA-256:
  `82bb209046d756da79d49ce05c98e442e621e6cdbbde09480a161c78d2e3ce31`
- Valid terminal manifest SHA-256:
  `7c9c859628c6786161e5e660775a38caa62260404acdfaa7df7ef3b7dbce4af2`
- Development lock semantic SHA-256:
  `5beb85b3e41fcca61be22926eaace9ac4ec4dc70711094dd16fef53a748eaa79`
- Development lock file SHA-256:
  `0e1d6b8cc937b83bb72b1d0a2262241e6e71c51cf33d7f7a19c54f9c1df9e4e6`

The earlier `speed_development_v1` artifact is explicitly invalid for
authorization: its written protocol used a stale model-name roster even though
the runner used the correct archived matrices. The v2 protocol records the
identity-only amendment. Candidate definitions, gate thresholds, promotion
precedence, and stop rules were not changed after seeing that preliminary run.

## Implemented variants

- `V1`: prior scalable one-update implementation.
- `F32`: retain float32 during fitting, with the old class-loop scorer.
- `FAST`: `F32` plus one score matrix multiplication per row tile.
- `CAP32`: `FAST` plus at most 32 deterministic relevance rows per class.
- `NOLLOYD`: `CAP32` without the final weighted Lloyd update.
- `LP-FULL`: freshly measured five-fold linear probe comparator.

Promotion precedence was frozen as `FAST`, `CAP32`, then `NOLLOYD`. `V1` and
`F32` were diagnostic only.

## Result

All three promotable variants passed every development ranking gate:

- zero clean-linear regret delta versus `LP-FULL`;
- zero nuisance-linear regret delta versus `LP-FULL`;
- zero quadratic, kNN, and RBF regret delta versus `V1` on the nonlinear arm.

Paired candidate / `LP-FULL` wall-time ratios were:

| Candidate | Baseline median / upper 95% | Nonlinear median / upper 95% | Nuisance median / upper 95% |
|---|---:|---:|---:|
| `FAST` | 0.693 / 0.769 | 1.183 / 1.526 | 0.807 / 0.955 |
| `CAP32` | 0.623 / 0.732 | 1.096 / 1.414 | 0.734 / 0.948 |
| `NOLLOYD` | 0.584 / 0.643 | 0.998 / 1.285 | 0.686 / 0.814 |

The frozen runtime gate required median <= 1.0 and upper 95% <= 1.10 in every
arm. `NOLLOYD` was typical-case competitive on nonlinear geometry, but its
backbone-block upper bound failed. Its nonlinear per-backbone ratios ranged
from 0.754 to 1.556, with `vit-small-16` and `deit-tiny` the clearest slow cases.

The immutable decision is therefore:

```text
status: stopped_no_eligible
selected_candidate: null
full_600_panel_replay: not authorized
```

## Runtime attribution

Approximate median nonlinear-arm stages for `NOLLOYD` at 80 rows per class:

| Stage | Wall time |
|---|---:|
| Class-owned MiniBatchKMeans fits | 0.758 s |
| Capped relevance estimation | 0.123 s |
| Final weighted Lloyd update | 0.001 s |
| Fast held-out OI scoring | 0.060 s |
| Outer end-to-end selector | 1.010 s |
| Full linear probe | 1.097 s |

The implementation work succeeded: the held-out scorer fell from roughly
0.25 s to 0.06 s, capping reduced relevance estimation from roughly 0.21 s to
0.12 s, and omitting Lloyd removed roughly 0.09 s. The remaining performance
floor is the repeated class-owned scikit-learn MiniBatchKMeans setup and fit,
which now accounts for about three quarters of end-to-end time.

## Recommended continuation

Test a separately versioned fused class-batched prototype fitter. It should
retain class ownership and the same raw-distance objective, but represent all
classes in padded/bucketed tensors and perform a small fixed number of batched
assignment/update passes. Compare it first against the exact current prototype
centers, downstream selector regret, and the two slow nonlinear backbones.
Do not change the v2 artifact or reopen its stopped decision.

## Reproduction and verification

See `README.md` for exact commands. The valid artifacts are local under:

- `artifacts/scalable_relevance_kmeans/speed_development_v2`
- `artifacts/scalable_relevance_kmeans/speed_development_analysis_v2`

Final repository verification after the corrected run:

```text
650 passed, 39 warnings
```

The warnings are the existing, expected under-prototyped-label warnings.
