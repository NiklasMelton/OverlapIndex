# Results

The frozen SmallNORB screen completed all 9 panels and 45 candidate rows using
only the five scout backbones. The five reserved backbones were not evaluated.

One of three targets was admitted:

- **Elevation:** admitted. Median best-head R2 was 0.3489; median nonlinear
  advantage over linear ridge was +0.0453; 10/15 rows had at least +0.02
  nonlinear advantage; four backbones had a stable nonlinear winner; median
  backbone-envelope span was 0.1960; training-only family selection accuracy
  was 0.733. RBF-SVR was the modal nonlinear winner.
- **Azimuth sine/cosine:** rejected. Their median best-head R2 values were
  negative, so occasional nonlinear wins did not constitute useful nonlinear
  geometry.

For admitted elevation, the descriptive continuous SAVE-RFF OI ranker had
mean Spearman 0.8667, mean top-1 regret 0.0101 R2, and exact-best rate 2/3.
Linear CV had higher mean Spearman (0.9333) but worse mean top-1 regret
(0.0266) and exact-best rate (1/3). These three development panels are
promising but are not confirmation evidence.

Canonical artifacts:

- `artifacts/continuous_smallnorb_head_discovery/inputs/prepared/manifest.json`
- `artifacts/continuous_smallnorb_head_discovery/embeddings/manifest.json`
- `artifacts/continuous_smallnorb_head_discovery/results/manifest.json`

## Reserved-backbone validation

The immutable elevation-only validation subsequently evaluated the five
previously untouched backbones and passed all six frozen target gates:

- Median best-head R2: 0.3453.
- Median nonlinear advantage: +0.0297 R2.
- Material nonlinear rows: 10/15.
- Stable nonlinear winner: all five backbones.
- Median envelope span: 0.1544.
- Training-only head-family accuracy: 0.8667.
- RBF-SVR was the unique held-out winner in 13/15 rows.

The target therefore generalizes across the reserved backbone roster. The
ranker result is less favorable: on the reserved roster, linear CV selected the
best backbone in 3/3 seeds with zero regret, while broad and SAVE continuous OI
selected it in 2/3 seeds with mean regret 0.0246 R2. Broad OI retained higher
mean Spearman than linear CV (0.8000 versus 0.7333), whereas SAVE did not
(0.6333).

As explicitly post-outcome context, combining all ten scout and reserved
backbones gives broad OI mean Spearman 0.8020 and mean top-1 regret 0.0167,
versus linear CV 0.8465 and 0.0190. That pooled view is descriptive, not a
frozen gate. It reinforces that global ranking and top-1 shortlisting remain
distinct objectives and that the broad continuous variant is the more useful
baseline for the next ablation.

Canonical validation artifacts:

- `artifacts/continuous_smallnorb_head_validation/embeddings/manifest.json`
- `artifacts/continuous_smallnorb_head_validation/results/manifest.json`
