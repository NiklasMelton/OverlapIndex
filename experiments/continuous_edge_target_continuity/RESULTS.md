# Continuous edge-level target-continuity result

The frozen experiment completed and did not pass its universal-ranking gate.
It nevertheless isolated a useful signal: target continuity on a row-level
RFF-neighbor graph was a strong *best-backbone selector*, even though it was
not a sufficiently faithful ordering of the entire candidate panel.

## Identity and validation

- Frozen implementation commit: `da015cd`
- Protocol SHA-256:
  `67aa543ff73c5a44bf0a5cf603427a09c698ba6d3bb9f29d2c52bc7b9573e90c`
- Runner SHA-256:
  `ffc3152209819b397476909309ac607ed234471fcb354ceb02ce336e874ec4f8`
- Parent raw-results SHA-256:
  `4ce42178560c5cc99dc074ff958520f75434fdd33b3f17a9b436ddc3a883743a`
- Raw-results SHA-256:
  `cbfd4223e43024c47544a497ad58567b387ac09b5d173c3753995fe84c435654`
- Summary SHA-256:
  `a39b1786144aafb58cb41236cb0433bbc50523dbef9a048361d244fd06890ea4`
- Manifest SHA-256:
  `90e393308fd70e98574b300ad60d762664b5e9cc6c1600c61a0b53b2931ac2dd`
- Complete grid: 18 panels, 288 candidates, 1,152 selector rows,
  4,608 fitted fold models, 294,912 edge-null samples, and 18 checkpoints.

Every source and artifact hash validated. Each replayed parent actual loss and
common-bank seed matched exactly. Every fold contained 125 held-out rows, 625
directed row-neighbor edges, and prototype-support total 125. Rebuilding the
summary from the raw rows matched exactly. The complete project suite passed
with 367 tests, 14 expected skips, and 34 existing under-prototyping warnings.

## Result

No edge estimator passed all frozen criteria.

| Geometry / score | Defined panels | Mean rho | Mean regret | Exact best | Within .01 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RFF mean / current OI | 18 / 18 | .400 | .117 | .611 | .722 |
| RFF mean / row-kNN edge OI | 18 / 18 | **.455** | **.032** | **.833** | **.944** |
| RFF mean / prototype-cross OI | 16 / 18 | .465 | .109 | .556 | .667 |
| SAVE / current OI | 18 / 18 | .523 | .190 | .167 | .500 |
| SAVE / row-kNN edge OI | 18 / 18 | .562 | .151 | .222 | .611 |
| SAVE + RFF / current OI | 18 / 18 | **.615** | **.065** | **.722** | **.833** |
| SAVE + RFF / row-kNN edge OI | 18 / 18 | .541 | .133 | .556 | .667 |
| SAVE + RFF + residual / current OI | 18 / 18 | .593 | .111 | .556 | .667 |
| SAVE + RFF + residual / row-kNN edge OI | 18 / 18 | .499 | .105 | .722 | .833 |

The standout was `rff64_mean8_k4` with row-level 5-NN edges. It selected the
exact best candidate in 15 of 18 panels and a candidate within 0.01 R2 regret
in 17 of 18. It had zero mean regret in sinusoid and selected the exact best
candidate in every sinusoid panel. Its family Spearman was .804 for sinusoid
and .379 for triangle wave.

This estimator passed completeness, periodic gain, regret, exact-best, and
near-best requirements. It failed the overall-rank threshold (.455 versus
.600) and narrowly missed preservation on absolute value (.330 versus the
required .336). It therefore remains a failed exploratory candidate, not an
integration candidate.

The unbounded RFF row-edge ratio was weaker than the bounded score: rho .416,
regret .091, exact-best .722, and within-.01 .833. The improvement is not a
reporting artifact, but the frozen clipping/normalization does contribute to
the excellent top-selection behavior.

## What the experiment explains

The row-edge statistic fixes the null-direction pathology found in the prior
strict Wasserstein experiment. Randomly permuted target values on a frozen
row graph retain positive expected pairwise disagreement instead of making
two empirical endpoint distributions converge to the same global
distribution. All 18 row-edge panels consequently produced defined bounded
scores.

The prototype-cross version did not solve the broader problem. Although it
used cross-row absolute differences rather than Wasserstein distance, routing
the statistic through target-conditioned prototype supports and coarse
prototype adjacency still produced constant bounded scores in some periodic
panels. Its apparently high SAVE rho (.770) covered only 12 of 18 panels and
had zero exact-best selections, so it is not evidence of a superior ranker.

Row and prototype scores were usually similar at the candidate level (mean
within-panel Spearman .84--.90 where both were defined), yet their selection
behavior differed materially. This indicates that the important gain comes
from preserving fine row-neighborhood structure near the top of the panel,
not merely from changing the loss formula.

Most importantly, the experiment separates two objectives that the earlier
work treated as one:

1. SAVE plus RFF under current continuous OI remains the better full-panel
   ordering channel (rho .615, regret .065).
2. RFF row-neighbor continuity is the better top-candidate channel (regret
   .032, 83.3% exact best, 94.4% within .01).

The RFF row score behaves more like a best-candidate detector than a complete
rank statistic. It captures local and periodic target continuity well, but
middle-of-panel ordering remains sensitive to nuisance and family-specific
geometry.

## Next recommendation

Do not replace continuous OI with the row-edge score, and do not further tune
this exhausted synthetic panel. The next experiment should test a frozen
*score-level* combination rather than another feature concatenation:

- retain SAVE + RFF current OI as the full-order channel;
- add RFF row-kNN target continuity as the top-selection channel;
- predeclare a very small set of simple, deterministic rank combinations
  (for example equal fractional-rank averaging and a top-only guardrail);
- evaluate those formulas on new seeds or new generators, with no weight or
  threshold chosen from this result.

That direction follows the observed complementarity while preserving the
current OI signal. It is evidence for a two-channel continuous ranking method,
not yet evidence for a universal single score.
