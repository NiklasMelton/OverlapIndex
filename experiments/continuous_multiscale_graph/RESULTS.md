# Continuous multiscale graph result

The frozen new-seed experiment completed and rejected both unified multiscale
graph formulations. Density adaptation helped modestly, and SAVE + RFF
multiscale continuity repaired periodic ordering, but no formulation combined
that gain with adequate linear ordering, full-panel correlation, and
best-candidate selection.

## Identity and validation

- Frozen pre-outcome implementation commit: `10b600c`
- Protocol SHA-256:
  `8d9f9b109b7e86f483e4d7e5c73861c154f9e230740c657dc2af57ca2224364b`
- Runner SHA-256:
  `79c8e12d57534770d0e9da41613e328ee92f44855a91f1f2e30c2337099a62cc`
- Motivation raw-results SHA-256:
  `cbfd4223e43024c47544a497ad58567b387ac09b5d173c3753995fe84c435654`
- Raw-results SHA-256:
  `ca10227cc0c4dcc861e4d176758326de1c6a29778c279c4f62b3a6d9436f2860`
- Summary SHA-256:
  `19b2412c42de5ae7745d017ca3a0983113ff61403b12cb5a7dfcc9464eb0e3ef`
- Manifest SHA-256:
  `de49fd131e3eb5e0d4c9c54c5dff1c552e25f4098825e94a261ea00bcf3a4655`
- New panel seeds: 1301, 1409, and 1511.
- Complete grid: 18 panels, 288 candidates, 576 selectors, 2,304 fitted
  fold models, 442,368 graph-null samples, and 18 checkpoints.

All source and terminal hashes validated. Every panel/candidate/fold identity,
head result, graph size, edge count, null-bank length, scale integration, and
checkpoint was checked independently. Rebuilding the complete summary from
raw panels matched exactly. The full project suite passed with 379 tests, 14
expected skips, and 34 existing under-prototyping warnings.

## Primary result

No candidate passed all frozen gates.

| Geometry / estimator | Mean rho | Mean regret | Exact best | Within .01 |
| --- | ---: | ---: | ---: | ---: |
| Linear probe | .432 | .221 | .333 | .500 |
| RFF mean / current OI | .433 | .099 | .611 | .667 |
| RFF mean / uniform multiscale | .375 | .132 | .611 | .667 |
| RFF mean / self-tuning multiscale | .403 | **.091** | **.667** | **.722** |
| SAVE + RFF / current OI | .515 | .129 | .556 | .611 |
| SAVE + RFF / uniform multiscale | .510 | .121 | .500 | .556 |
| SAVE + RFF / self-tuning multiscale | **.520** | .121 | .500 | .556 |

Every multiscale score was defined on all 18 panels, and every graph candidate
beat the linear probe's mean regret by a wide margin. However, none reached
the required .60 mean rho. The best graph regret was .091, while the best
exact-best and within-.01 rates were .667 and .722, short of the joint frozen
requirements.

## The useful partial result

For SAVE + RFF, self-tuning multiscale continuity increased mean periodic
Spearman by .137 relative to matched current OI, passing the frozen +.10
periodic-gain requirement:

| Family | Current OI | Self-tuning multiscale |
| --- | ---: | ---: |
| Sinusoid | .727 | .851 |
| Triangle wave | -.139 | .012 |
| Linear | .772 | .567 |
| Quadratic product | .656 | .601 |
| Radial quadratic | .501 | .547 |
| Absolute value | .571 | .540 |

The periodic repair was therefore real, but it was purchased with a .205 loss
in linear Spearman. The linear-preservation gate failed, as did overall rho,
absolute regret, exact-best, and near-best gates.

The intermediate self-tuning scale, k=15, produced the strongest individual
triangle-wave correlation for SAVE + RFF (.205), but still achieved only .537
overall rho, .114 regret, .500 exact best, and .556 within .01. The coarse
k=45 graph weakened linear, quadratic, absolute-value, and triangle ordering.
Equal integration could not recover information that the coarse graph had
already blurred.

## Generalization of the previous row-edge signal

The striking RFF k=5 result from the motivating seeds did not reproduce on
the new panels:

| RFF row-kNN result | Motivation seeds | New seeds |
| --- | ---: | ---: |
| Mean rho | .455 | .382 |
| Mean regret | .032 | .123 |
| Exact best | .833 | .611 |
| Within .01 | .944 | .667 |

The new panels were harder in general—the matched current RFF OI also moved
from .400 rho / .117 regret to .433 / .099—but that does not explain the loss
of the unusually strong k=5 top-selection result. Fine-scale continuity is a
useful signal, not a stable universal selector by itself.

## Interpretation

The experiment rules out a simple version of the unified hypothesis:
averaging target smoothness over fixed neighbor counts is not sufficient to
unify global and local downstream learnability.

The failure has a geometric explanation. For a smooth periodic target, fine
and intermediate neighbors often have similar targets, but coarse feature
neighbors can cross folds of the target function. For a globally linear
target, increasing the neighborhood scale adds useful order only when the
learned metric preserves that global direction. One equal multiscale average
cannot distinguish useful coarse structure from cross-branch mixing. Local
density normalization changes edge influence but not this topological issue.

This also explains why the score improved periodic families while harming
linear ordering: scale is not merely a nuisance parameter. Different target
relationships genuinely have different informative graph-scale profiles.

## Recommendation

Do not integrate this multiscale graph score and do not tune k or scale
weights on these outcomes. Before accepting a late hybrid, the remaining
coherent single-statistic direction is a characteristic-kernel dependence
measure, such as a cross-fitted approximate HSIC/kernel-target-alignment
score with a frozen multi-bandwidth feature kernel. Unlike equal graph-scale
averaging, such a statistic integrates pairwise dependence continuously and
does not force distant neighbors to behave as locally smooth edges.

That should be treated as the last high-level unified diagnostic. If it also
shows the same local/global trade-off, the accumulated evidence would support
an explicit two-channel ranking method rather than another nominally unified
score.
