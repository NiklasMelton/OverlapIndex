# Controlled nonlinear real-image panel result

## Decision

**FAIL.** The bounded shared continuous-OI shortlist did not match the linear-probe baseline on the frozen primary top-five mean-regret criterion.

This was a developmental stress test on the exact images, embedding caches, backbone roster, splits, preprocessing, reference heads, and OI implementation used by the preceding linear UTKFace panel. The only changed experimental variable was the target:

```text
abs(age - 44.5) / 44.5
```

The panel is not untouched confirmation.

## Results

| Method | Mean Spearman | Top-5 opportunity regret | Top-5 mean regret | Material precision | Panel seconds |
|---|---:|---:|---:|---:|---:|
| Broad RFF | 0.7735 | 0.00000 | 0.01891 | 0.880 | 6.502 |
| SAVE/RFF filler | 0.7106 | 0.00000 | 0.02541 | 0.840 | 0.000 incremental |
| Bounded top-2k | 0.7835 | 0.00000 | 0.01901 | 0.840 | 6.502 |
| Linear probe | 0.9894 | 0.00000 | 0.00000 | 1.000 | 0.292 |

The paired bounded-minus-linear-probe top-five mean-regret estimate was `+0.019006`, with a 95% training-cohort stability interval of `[+0.002249, +0.042360]`. Positive values favor the linear probe, so the frozen primary noninferiority rule failed.

The bounded-minus-broad estimate was `+0.000096`, with interval `[-0.008782, +0.010736]`. The filler therefore produced no reliable improvement over broad retrieval.

The best reference head across all 80 backbone/cohort cells was:

- linear ridge: 65
- RBF SVR: 15
- quadratic ridge: 0
- KNN: 0

## Interpretation

The target is nonlinear as a function of chronological age, but it was not reliably nonlinear in the pretrained embedding coordinates. Deep image backbones already expose nonlinear age cues as approximately linear features. Consequently, the linear reference head still won 65 of 80 cells and dominated the strongest backbones. RBF wins occurred mostly among weaker candidates and did not materially change the oracle top-five envelope.

This distinction matters: a nonlinear transformation of a semantic label does not guarantee a nonlinear downstream relationship after feature extraction. The panel therefore does **not** provide a decisive real-image test of whether continuous OI preserves an advantage when genuinely nonlinear heads determine the best backbones.

The broad shared RFF score still found the oracle-best backbone in every training cohort. The failure was shortlist ordering below the winner: bounded OI included the best candidate but assigned too much weight to several lower-utility candidates, while the linear probe reproduced the oracle top five exactly in every panel.

Runtime also remained noncompetitive in this implementation: approximately `6.50 s` per OI panel versus `0.292 s` for the linear probe, about `22.3x` slower. These are descriptive shared-table timings, not a separately counterbalanced resource benchmark.

## What this changes

Derived-label stress panels should stop here. The next real-image panel should use a naturally continuous task whose representation geometry makes nonlinear head wins plausible for the leading backbone candidates, chosen from task structure before inspecting candidate outcomes. Suitable directions include pose, depth, spatial localization, counting, or another target with independently supplied continuous labels.

## Artifact identity

- Protocol SHA-256: `161eb09340d1765a6116ec86ad6103683db1f147dc949357bd395b79d0516fef`
- Raw results SHA-256: `e1f82fa80fad11a7d9dea555b24ebf32b581b8796b03da854a3b41191ad8cf68`
- Summary SHA-256: `e1a029e2e474dfc9faa38b6289fd7362f8a0b87b9380aff9e910c6a344b6db4c`
- Report SHA-256: `c3e01d920a0dd746f7a84ca7aa7d2082e7bbc07a6f7604ae81ad6862cd8391ad`
- Manifest SHA-256: `6a3aff5571a3a6d12e846f4dac6cb74df314bacd9d6007ddef4556588cdc20aa`
