# Continuous shortlist-depth audit result

Date: 2026-08-31

The hash-bound audit reanalyzed 60 completed panels and 2,400
panel-depth-method rows. It fit no model and generated no new selector or head
outcome. All five frozen checks passed, so one unused-seed synthetic top-five
confirmation is justified for `unbounded_winner_fill`.

## Depth response

| k | Broad mean regret | Unbounded mean regret | Difference |
|---:|---:|---:|---:|
| 1 | .01461 | .01461 | .00000 |
| 2 | .07408 | .08070 | +.00662 |
| 3 | .10743 | .11040 | +.00297 |
| 4 | .10934 | .09472 | -.01462 |
| 5 | .11083 | .08579 | -.02504 |
| 6 | .09947 | .08355 | -.01592 |
| 7 | .09685 | .08600 | -.01086 |
| 8 | .10587 | .08344 | -.02243 |

At `k=5`, unbounded winner-fill minus broad-RFF mean utility regret was
`-.02504` R2 with 95% interval `[-.03533, -.01508]`. Both input cohorts
favored winner-fill (`-.01458` and `-.03549`). Top-five opportunity regret
was exactly noninferior at the interval upper bound of zero, and material
precision passed its noninferiority gate.

The result supports a shortlist-size-dependent interpretation rather than one
universal total order: broad RFF owns the winner and very small shortlist,
while SAVE+RFF filling becomes useful when four or more candidates are
returned. Oracle headroom inside broad top-`2k` was substantial, but the
tested outcome-free diagnostics were not strong enough to justify a learned
per-panel router. At `k=5`, realized gain correlated most with outcome-using
oracle headroom (Spearman `.648`); broad bandwidth and fold agreement were
only moderate (`.487` and `.340`), and broad/SAVE set agreement was nearly
uninformative (`-.063`).

## Verification and operational note

Canonical artifact hashes:

- `summary.json`: `0ba8793441ea6967bc2d9f490b39caeb9ea327a1d0489a563143c88304591ee5`
- `report.md`: `8ab2b355e21513d9cf728bfcd206c40ca53436c230869155c3a488e20429068f`
- `manifest.json`: `74d3bd801ae9bffe6b84bdf048572ae8b7bc8b7d914c396d51888e8a37421a49`

The summary, source identities, input bundles, terminal hashes, and exact Git
commit were independently recomputed and matched. An initial audit invocation
used an incorrect but syntactically valid commit argument. That bundle is
preserved at `artifacts/continuous_shortlist_depth_audit_invalid_commit_argument`
and is not evidence. The canonical bundle was regenerated from identical
frozen source and inputs with the correct commit. A post-outcome regression
now requires the argument to equal the repository HEAD.
