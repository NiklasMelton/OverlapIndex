# Results

The frozen n=300 residual-anchor diagnostic completed under protocol SHA-256
`c39cb80277cb969c2fb5f178f7e1c72588de3836850e336921bf5ce20059ff23`.
It evaluated 8 source-selected targets, 5 scout backbones, and 4 outer folds
(160 candidate-folds). No reserved backbones were evaluated and no method is
promoted.

## Integrity and verification

- The terminal artifact reports `completed`; explicit `--validate-only`
  validation reports 8 panels and 40 candidates as valid.
- The post-outcome focused suite passes: 31 tests.
- The raw-PCA outer linear anchor matches the parent information-ladder run
  exactly across all 160 folds: identical outer splits and PCA seeds, with
  maximum absolute prediction and held-out-R2 differences of zero. The
  selected Ridge alphas also match exactly.
- Learned fitted state was unchanged by diagnostics.

Canonical artifact SHA-256 values:

| artifact | SHA-256 |
|---|---|
| `raw_results.json` | `fb5791791c2ce73506f461ce18142abcc7d2497511a39edecd11726c96574740` |
| `summary.json` | `f9efa96823954d2820a8f9663014b457331aea1042da2ed03520922d7416a8fe` |
| `manifest.json` | `f5fdc39805b0d15fe1096d072e1e15c451bd4c408ffe4f8b7ce7e87417e14d93` |
| `report.md` | `4122b4f9b763fa48ab27710df5420055991d1ffb91a0c9edf8d8e1b92cb919c9` |
| `terminal.json` | `46c73913c90c67e59bf1ffc39c8292bb1f66f2bebb12541559977ce447d1b3a0` |

## Primary results

| method | mean held-out R2 | Spearman | top-1 regret | material-pair accuracy |
|---|---:|---:|---:|---:|
| outer linear Ridge | 0.184350 | 0.7625 | 0.011905 | 0.8417 |
| + SAVE residual | 0.172380 | 0.6250 | 0.007954 | 0.7889 |
| + RFF-mean residual | 0.172336 | 0.6250 | 0.011905 | 0.7889 |
| + equal SAVE/RFF residual | 0.175694 | 0.6375 | 0.011905 | 0.8028 |
| + prototype squared-distance residual | 0.174978 | 0.6750 | 0.011905 | 0.8153 |
| + prototype orthogonal-log-contrast residual | 0.175480 | 0.6375 | 0.011905 | 0.8028 |

All methods had zero top-3 opportunity regret on this small five-backbone
panel. SAVE selected the exact best backbone on one additional target, which
lowered mean top-1 regret by 0.00395, but it simultaneously reduced Spearman
by 0.1375 and material-pair accuracy by 0.0528. This isolated winner change is
not a robust ranking improvement.

The intended nonlinear stratum did not benefit. Its outer-linear mean
held-out R2 was 0.169055, while residual variants ranged from 0.149945 to
0.154607. Its outer-linear Spearman was 0.725; residual variants ranged from
0.475 to 0.575, with identical top-1 regret (0.012702).

## Shrinkage and controls

The leakage-safe calibration selected the no-harm value lambda=0 in 56.25% to
62.5% of folds, depending on residual method. When it selected a nonzero
correction, that correction did not generalize reliably. Mean held-out R2 for
the real calibrated residuals was 0.1723 to 0.1757, below the corresponding
conditional-association nulls (0.1770 to 0.1793) and target-permuted-geometry
nulls (0.1739 to 0.1804).

Forcing lambda=1 was substantially worse: mean held-out R2 ranged from
-0.0597 to 0.1344. Constructing geometry from the original target rather than
the residual also remained worse, from 0.0157 to 0.1388. These controls show
that shrinkage prevented larger damage, but do not reveal recoverable
incremental nonlinear signal.

## Runtime

The outer linear anchor averaged 0.0530 seconds per candidate pipeline. Each
residual pipeline averaged about 0.655 seconds, or approximately 12.4 times
the anchor cost. Most of the residual cost came from nested OOF construction
and calibration rather than the final readout.

## Conclusion

This is a valid negative diagnostic. On these eight source-selected targets,
adding target-aware SAVE, RFF, or prototype residual features to an exact
linear anchor did not preserve the anchor's ranking quality, did not improve
the nonlinear-opportunity stratum, did not beat its null controls, and was
materially slower. The residual-anchor family should not be promoted in its
current form.

Because target selection was not independent and only five backbones were
ranked, this result does not establish that every residual correction is
impossible. It does rule out these particular representations and this nested
calibration design as the next general continuous-OI method.
