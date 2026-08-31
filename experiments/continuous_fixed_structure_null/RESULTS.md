# Strict fixed-structure continuous-OI null result

The frozen strict-null experiment completed and rejected the hypothesis. It
held the actual held-out row-to-prototype assignments, prototype supports, and
prototype adjacency fixed, then globally permuted only scaled target values
before recomputing the existing support-weighted adjacency Wasserstein loss.

## Identity and validation

- Initial implementation commit: `dfeb610`
- Fail-closed constant-score correction: `01e8556`
- Protocol SHA-256:
  `7df37ac32b4063e01748a3bba1c42744af92cd97c140bc45d45618d4c662fb21`
- Runner SHA-256:
  `8dec3cb86c3d970d216481f0b778c1f80b3c4c168f2705c70e77e1c41aaa3b9d`
- Parent raw-results SHA-256:
  `d044251b6c7bec73842315b642498db179b525a85b14efc473914fef331ea6b4`
- Raw-results SHA-256:
  `4ce42178560c5cc99dc074ff958520f75434fdd33b3f17a9b436ddc3a883743a`
- Summary SHA-256:
  `abeb2f4fb9bf0e2c3fead26c17e796fbb324265fae6ce8d6fb962543d5193d7f`
- Manifest SHA-256:
  `3d00b7e979ab3b9b890aa788331526dfa3188ca259d5fb6a45b995c564fc07d5`
- Complete grid: 18 panels, 288 candidates, 1,152 selector rows,
  4,608 fitted fold models, 147,456 strict-null losses, and 18 checkpoints.

Every replayed actual fold loss and common-bank seed matched the parent null
diagnostic exactly. All file/source hashes and row counts were verified, and
rebuilding the summary from raw panels matched exactly. The complete suite
passed with 358 tests, 14 expected skips, and 34 existing under-prototyping
warnings.

The first execution completed all checkpoints but stopped during summary when
a constant bounded score produced undefined Spearman. The reporting logic was
corrected fail-closed: undefined correlation is retained as null and fails the
gate. That attempt is preserved separately; the canonical artifact was fully
rerun under the corrected source identity.

## Result: strict fixed structure is the wrong null for this loss

No method passed. The strict null made permutation loss smaller than actual
loss across most nonlinear panels. Consequently the bounded index clipped
every candidate to zero and could not rank them.

| Geometry | Panels with defined bounded rho | Strict-ratio rho / regret | Current-ratio rho / regret |
| --- | ---: | ---: | ---: |
| SAVE | 6 / 18 | .365 / .246 | .537 / .153 |
| RFF mean | 9 / 18 | .232 / .119 | .396 / .117 |
| SAVE + RFF mean | 9 / 18 | .531 / .114 | .597 / .103 |
| SAVE + RFF mean + residual | 9 / 18 | .533 / .114 | .587 / .076 |

The unbounded strict loss ratio also worsened ordering, so clipping is not the
only problem:

| Geometry | Current triangle rho | Strict triangle rho | Current sinusoid rho | Strict sinusoid rho |
| --- | ---: | ---: | ---: | ---: |
| SAVE | .256 | -.717 | -.100 | -.262 |
| RFF mean | -.100 | -.767 | .830 | .496 |
| SAVE + RFF mean | .030 | -.200 | .810 | .775 |
| SAVE + RFF mean + residual | .294 | -.055 | .856 | .819 |

For all four geometries, strict bounded OI was constant zero for all 48
sinusoid candidates and all 48 triangle candidates. It was also constant for
all radial-quadratic candidates. Median strict actual/null ratios on triangle
were 1.60--1.74, guaranteeing clipping at zero.

## Why it fails

The current loss is a Wasserstein distance between empirical target
distributions attached to adjacent prototypes. Under the strict permutation,
every frozen prototype support receives a random sample from the same global
target distribution. With enough rows, those empirical distributions become
similar, so their Wasserstein distance approaches zero. The permutation null
therefore looks *more locally coherent* than the real target structure.

This is the wrong direction for a lower-is-better null. It explains both the
mass clipping and the reversed unbounded-ratio ranking. On triangle panels,
the strict null remained strongly correlated with actual loss (rho .57--.81,
depending on view) and generally became even more negatively correlated with
the best-head envelope. It retained candidate-specific graph/support effects
while erasing the target-distribution differences needed to set a useful loss
scale.

The current target-reassigned null is therefore not merely unnecessary
recomputation: it prevents this particular degeneracy by reassigning rows
through permuted target cells. However, the preceding diagnostic showed that
it can still normalize away useful SAVE/RFF triangle ordering. Neither null is
universal.

## Next recommendation

Do not alter the permutation policy again while retaining prototype-to-
prototype Wasserstein distance as the statistic. Instead test an edge-level
target-continuity score whose permutation direction is naturally correct:

1. freeze a feature-neighbor graph or the existing prototype-adjacency graph;
2. measure mean pairwise target difference across actual graph edges, without
   collapsing each endpoint into an empirical distribution;
3. permute targets on the same fixed edges, where random endpoint pairing
   produces a large rather than vanishing reference loss;
4. normalize actual edge disagreement by that common fixed-edge null;
5. compare row-kNN edges with prototype-adjacency-weighted cross-row edges,
   using the same four frozen geometry views and best-head envelope.

This experiment should remain isolated. The strict null should not be added to
public `ContinuousOverlapIndex`.
