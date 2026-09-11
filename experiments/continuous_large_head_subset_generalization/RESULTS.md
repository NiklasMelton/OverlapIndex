# Large-panel cheap head-subset generalization result

## Outcome

The independently preselected quadratic-Ridge plus kNN envelope did **not**
generalize. It passed the runtime, control-Spearman, and overall top-three
checks, but failed the overall/opportunity Spearman and top-one-regret checks.
No model was fit: the experiment joined 24 frozen panels, 120 scout-backbone
rows, and their archived training-only inner-CV scores and timings.

| Method | Overall rho | Opportunity rho | Control rho | Top-1 regret R2 | Top-3 regret R2 | Cost vs Linear CV |
|---|---:|---:|---:|---:|---:|---:|
| Linear CV | 0.3792 | 0.3500 | 0.4083 | 0.04119 | 0.00900 | 1.00x |
| Quadratic Ridge | **0.4375** | **0.4083** | **0.4667** | **0.03313** | 0.01040 | **1.12x** |
| kNN | 0.3167 | 0.2750 | 0.3583 | 0.06265 | 0.01549 | 1.00x |
| Quadratic Ridge + kNN | 0.3458 | 0.3083 | 0.3833 | 0.04853 | 0.00971 | 2.12x |
| RBF SVR | **0.5167** | **0.5500** | **0.4833** | 0.03319 | **0.00636** | 4.87x |
| Quadratic Ridge + RBF SVR | 0.4958 | 0.5250 | 0.4667 | 0.03319 | **0.00636** | 5.99x |
| All four heads | 0.4792 | 0.4833 | 0.4750 | 0.03673 | 0.01072 | 7.99x |

## Why the selected pair failed

The failure is the raw `max` aggregation, not excessive cost. Quadratic Ridge
alone ranks better than quadratic Ridge plus kNN in every aggregate Spearman
slice. A noisy kNN inner-CV value can become the candidate maximum and replace
the better calibrated quadratic score. Adding a channel is therefore not
monotone in ranking quality. The same effect appears elsewhere: RBF alone
outperforms quadratic+RBF and the full four-head maximum in overall Spearman.

The quadratic model is unusually effective here because its feature map
contains the original linear coordinates as well as squares and pairwise
interactions. With the frozen 32-dimensional PCA input it evaluates 560
features, so it strictly contains the linear hypothesis class while adding a
useful low-order nonlinear readout. The current Ridge implementation remains
cheap at 100 training rows.

## Important diagnostic lead

Quadratic Ridge alone was diagnostic-only and cannot pass the formal frozen
decision. Descriptively, however, it satisfies every numerical threshold that
was applied to the primary pair:

- overall Spearman gain over Linear CV: +0.0583;
- opportunity Spearman gain: +0.0583;
- control Spearman gain: +0.0583;
- overall top-one regret reduction: 0.00807 R2;
- opportunity top-one regret reduction: 0.00935 R2;
- control top-one regret reduction: 0.00679 R2;
- overall top-three regret increase: 0.00140 R2, within the 0.002 allowance;
- archived runtime: 1.116 times Linear CV, within the 3x limit.

This is not uniformly positive target by target. Quadratic Ridge remains weak
on LSP ankle span and increases top-three regret on COCO wrist span, while it
is particularly strong on Pascal box area and both CUB targets. The aggregate
result therefore requires genuinely prospective validation rather than
immediate promotion.

## Recommendation

Do not build the multi-channel surrogate bank yet. First freeze quadratic
Ridge alone as the sole candidate on an untouched backbone roster or fresh
structured continuous datasets. Compare it with Linear CV and the held-out
best-head envelope using the same ranking, regret, shortlisting, and runtime
gates. If it replicates, the simplest viable continuous transferability proxy
may be a degree-two Ridge probe rather than a prototype-based Continuous OI.

If quadratic Ridge fails specifically on smoother or periodic relationships,
then approximate the RBF channel and combine channels only after solving their
cross-family calibration. Another raw maximum should not be tested.

## Verification and identity

- Protocol SHA-256:
  `4e279b9a7ff4866430efbcfc361d577910b355295b2c39ebae1542ab84ca65ef`
- Runner SHA-256:
  `f0fe81c3a8f870850e8f134bf53b4f713ae93358c293deebbcd522b02f24684b`
- Raw results SHA-256:
  `bc6001766d0d634603bb627558bd6882251c8986a2b8f8cdfbe4a8be414875c6`
- Summary SHA-256:
  `daf7ca9e6ac270f659aa3d0e0552533e8fc3fd942f8c0b746b51cc9a7d35b2cd`
- Report SHA-256:
  `195bef98dc6401526753b4ff9f237aac5a84ea889b876193b1928646cf0a55cf`
- Manifest SHA-256:
  `9fbdb3709f61551f98e927423e82b2c599e1b5fbe7d62f209288979b06ec534a`
- Combined regression suite: 29 passed.
- Terminal counts: 24 panels, 120 candidates, 15 subsets, zero model fits,
  zero OverlapIndex fits, and zero reserved backbones.
