# Post-outcome verification note

The frozen experiment executed from commit
`34123d8efa2daa58af59b9c1cc04f487bc1d8321` and completed all 30 planned
panels. The terminal artifact hashes are:

- `raw_results.json`: `5b0ddae20f7a2fc8475349532a0606827a3b81b9091fab0cd08342beec993c56`
- `summary.json`: `75d511df167bf15fd6177f224ae71cd14ca1ad12cdbd3729aa8b4b0aae5c5342`
- `report.md`: `82771280b4ce5c07228392b6cbafd36c9b09fe0ec33809033249b1a0093bb6be`

The result is frozen as a failure: winner-fill passed five of the six decision
checks but did not improve top-three mean utility regret over broad RFF.

During post-outcome verification, the protocol validator was found to compare
the insertion order of the `ranking_methods` JSON object. The atomic JSON
writer uses sorted keys, so a protocol reloaded from `raw_results.json` failed
that check even though JSON object order has no meaning. This note accompanies
an order-independent method-set validation and a regression test. The
correction does not change the protocol bytes, ranking construction, panel
outcomes, metrics, bootstrap, decision, or existing artifacts. Outcomes were
not rerun.

Independent recomputation from the frozen raw panels reproduced `summary.json`
exactly. All 30 checkpoint source-hash mappings match the raw artifact, and the
30 seed-family identities are unique and equal the frozen grid.
