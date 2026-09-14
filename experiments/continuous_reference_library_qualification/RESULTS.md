# Reference-library qualification V1 results

## Identity

- Executed commit: `55d4f08c7d77d899853d603809c70f5fbe78af21`
- Protocol SHA-256: `635793e89504a4081d213401f9879d827312597eb13942db5df0ce010afaa5cd`
- Manifest SHA-256: `e49f8e86a49e3b680bbb8fab6353c098cd2f8a7ff361d3c1ea3422ccf5651f3b`
- Raw-results SHA-256: `a3df974be45e8a71838924cd456ea6dce4f2147ade8445beab977fb81b9af859`
- Summary SHA-256: `ac41b2c15cdc8808a2a22a5ccfa0d919a082f61b0cfded381df6d328ecc69bce`
- Pre-execution verification: 19 combined focused tests passed
- Executed scope: 25 real candidate draws, 9 synthetic panels, 45
  synthetic candidates, no OverlapIndex fits, and no reserved backbones

## Decision

The frozen qualification **failed** because 14 selected parameter endpoints
had at least a 0.01 R2 inner-validation advantage over the best setting away
from that endpoint. The eight-target screen was therefore not opened.

All other gates passed: the linear, quadratic, and smooth-periodic controls;
both real CUB linear controls; the real LSP nonlinear-head control; and the
zero-warning requirement.

## Behavior

The stronger scale-aware recipes preserved the corrected rankings:

| Target | Envelope winner | Linear winner | Linear regret | Maximum nonlinear gain |
|---|---|---|---:|---:|
| CUB leg span | DINOv2-small | DINOv2-small | 0.000 | 0.025 |
| CUB wing axis | DINOv2-small | DINOv2-small | 0.000 | 0.017 |
| LSP wrist span | ConvNeXt-tiny | ConvNeXt-tiny | 0.000 | 0.184 |

Every synthetic panel ranked the clean latent representation first. The
minimum intended-head R2 was 0.9985 for linear, 0.9982 for quadratic, and
0.9983 for smooth-periodic controls. Mean linear-probe regret was 0.126 on the
quadratic controls and 0.213 on the smooth-periodic controls.

## Endpoint failures and response

The actionable failures were:

- five real RBF fits selected the maximum standardized epsilon of 0.15 with a
  material advantage;
- LSP ConvNeXt quadratic Ridge selected the maximum mean-loss lambda of 100
  with a 0.050 inner-R2 advantage;
- all three degraded smooth-periodic controls selected both the minimum RBF C
  of 0.03 and the minimum bandwidth factor of 0.25 materially.

Two further failures were maximum-k selections for the known noise-only
synthetic candidate. A null candidate has no population-optimal local scale;
choosing among equally uninformative heads is selection noise. V1 reports
these failures exactly, but the next protocol will predeclare the known null
as diagnostic-only for endpoint adequacy while retaining it in every outcome
and ranking report.

The next protocol version must widen epsilon upward, quadratic shrinkage
upward, and RBF C and bandwidth downward. It must also include the largest
feasible k as a structural endpoint. No V1 outcome may be overwritten and the
new version must rerun the complete qualification panel before the historical
eight-target screen is allowed.
