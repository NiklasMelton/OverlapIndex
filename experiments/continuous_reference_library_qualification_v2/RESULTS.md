# Reference-library qualification V2 results

## Identity

- Executed commit: `fc2f31e5dda3ce26a768e6a7555052adbc224cde`
- Protocol SHA-256: `39c413d69fe0638624f0ee5ce719623382a7722f46902784a5a130320bd1f100`
- Manifest SHA-256: `e214d800e7dc6c1234a9a08721b5de58e25b227422237bc4c40b23ce4653f2cf`
- Raw-results SHA-256: `cf1ae029da372e95ef2adffc64d1c2b01b5d49f315d1c2f1092ed02e1195289c`
- Summary SHA-256: `a28f0bd3fc8e759901fca4a940c45c64a6ae4eb4bc892497a962f151b7389b0e`
- Pre-execution verification: 24 focused tests passed
- Scope: full rerun of 25 real candidate draws and 45 synthetic
  candidates; no OverlapIndex fits and no reserved backbones

## Decision

V2 **failed** the endpoint gate and did not authorize the eight-target screen.
All synthetic, real control, nonlinear-head, and warning gates passed.

The expanded ranges reduced material endpoints from 14 to two:

- CUB wing / DeiT-tiny / RBF selected standardized epsilon 0.5, with a
  0.0111 inner-R2 advantage over the best non-endpoint setting.
- Smooth-periodic seed 2026093401 / degraded latent / RBF selected epsilon
  0.5, with a 0.0160 advantage.

The first RBF remains below the candidate's winning linear head (held-out R2
0.119 versus 0.150). The second has inner R2 0.023 and held-out R2 0.012, below
the candidate's winning kNN head at 0.028. These endpoints therefore expose a
gate-definition problem as well as a remaining regularization boundary:
forcing an interior optimum for a head that has no useful fit drives the grid
toward a constant predictor without strengthening the envelope.

## Preserved behavior

- Both CUB tasks retain DINOv2-small as linear and envelope winner with zero
  linear top-1 regret.
- LSP retains ConvNeXt-tiny as both winners, with a material nonlinear-head
  gain of 0.184 R2.
- All nine synthetic panels rank the clean latent candidate first.
- Intended clean-head R2 remains at least 0.9982.
- Mean linear regret is 0.126 on quadratic and 0.213 on smooth-periodic
  controls.

## Next protocol

A successor may add larger standardized epsilon values, but must not keep
expanding unlearnable cells indefinitely. Before outcomes it should declare a
minimum inner-CV learnability threshold for actionable endpoint assessment,
retain every excluded endpoint in diagnostics, and always keep intended clean
synthetic heads actionable. It must rerun the complete panel and preserve V2.
