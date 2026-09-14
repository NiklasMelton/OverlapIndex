# Continuous reference historical screen results

## Identity

- Executed commit: `316ca33802e3c7f9e6a64d751d7b105ffad66631`
- Protocol SHA-256: `a0080fa46b92ddc7de09e8bf84588e894e1d7e12b707d71fef5a88fa398ec238`
- Qualified-library protocol SHA-256: `807b9687a48d8025c9df54d742a06613e4bd89ecd04f7a934861c3e561961eb3`
- Manifest SHA-256: `6f87e6f497d303d0505008f138a166f424f06d06f786c011c8fa906966cf512a`
- Raw-results SHA-256: `911c03a591b1e2139e364c8a30ce55060ab8233b9d493ff9732cd2860d861a56`
- Summary SHA-256: `4c61922a2cde7afd76af180d17391e8d168e45ac958713068b78eba45c5708cb`
- Pre-execution verification: 37 combined tests passed
- Scope: eight historical targets, five scout backbones, 40 candidate-target
  cells, and 220 candidate draws; no OverlapIndex fits and no reserved
  backbones
- Evaluation: 200 fixed evaluation rows per target, 2,000 paired bootstrap
  resamples, and Bonferroni-adjusted simultaneous utility intervals

The manifest records a clean source tree at execution. The runner consumed the
exact qualified V3 head library without tuning or reselection.

## Decision

The historical screen found **no qualified nonlinear-challenge lead** and did
not authorize an OverlapIndex comparison:

- method promotion: `none_reference_only`;
- next stage: `expand_real_sources_before_oi`;
- nonlinear-challenge leads: 0;
- independent nonlinear-challenge sources: 0;
- sufficient historical nonlinear diversity: false.

The frozen classification produced seven `mixed` targets and one `unresolved`
target. `Mixed` denotes a learnable target with a point nonlinear-head gain;
it is not evidence of a reliable backbone-ranking reversal. No target had a
positive simultaneous lower bound for nonlinear opportunity, and no material
pair reversal passed the frozen sign-probability gate.

## Target-level findings

| Target | Point envelope winner | Point linear winner | LP-to-envelope Spearman | LP top-1 point regret | Maximum point nonlinear gain | Winner top-1 probability |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| COCO wrist span | DINOv2-small | DINOv2-small | 1.00 | 0.000 | 0.006 | 0.592 |
| CUB leg span | DINOv2-small | DINOv2-small | 1.00 | 0.000 | 0.023 | 0.943 |
| CUB wing axis | DINOv2-small | DINOv2-small | 1.00 | 0.000 | 0.032 | 1.000 |
| LSP ankle span | ConvNeXt-tiny | MobileNetV3-large | 0.70 | 0.014 | 0.106 | 0.508 |
| LSP wrist span | ConvNeXt-tiny | ConvNeXt-tiny | 1.00 | 0.000 | 0.159 | 0.825 |
| Pet compactness | ResNet50 | ResNet50 | 0.90 | 0.000 | 0.071 | 0.943 |
| Pascal box area | ResNet50 | ResNet50 | 1.00 | 0.000 | 0.052 | 0.943 |
| Pascal box aspect | DINOv2-small | DINOv2-small | 1.00 | 0.000 | 0.029 | 0.644 |

Nonlinear heads improved absolute utility by as much as 0.159 R2, but usually
improved the same backbone that the linear probe already preferred. LSP ankle
was the only point-estimate winner change. It was not stable: the envelope
winner's top-1 probability was 0.508, the point regret was only 0.014 R2, and
the conservative regret interval included zero.

## Interpretation

This is a negative benchmark-qualification result, not a negative result for
any OverlapIndex formulation. The experiment deliberately did not fit OI.
Instead, it tested whether the available historical targets contain the kind
of stable nonlinear backbone-ranking conflict needed to distinguish a
nonlinear-aware ranking method from a linear probe.

They do not. On seven of eight targets the linear-probe and best-head-envelope
rankings had Spearman correlation of at least 0.90, and the point top-1 regret
was zero. The panel shows that nonlinear heads can materially increase
predictive utility without changing which backbone should be selected. That
explains why prior continuous-OI experiments could show useful nonlinear
sensitivity yet fail to demonstrate a general ranking advantage.

The uncertainty is also too large for confirmation claims. Four targets have
only one complete archived training draw, every evaluation cohort has 200
rows, and simultaneous regret upper bounds range from roughly 0.24 to 0.50 R2.
The absence of a qualified lead therefore means that the present panel is
insufficiently discriminating; it does not prove that nonlinear ranking
conflicts are absent in the broader population.

## Authorized next stage

Expand the real-data source panel before evaluating OI. The next benchmark
should preserve the qualified V3 head library and frozen decision thresholds
while adding harder, more diverse continuous targets, larger evaluation
cohorts, reusable pools supporting ten draws, and a broader scout-backbone
set. Candidate sources should be screened using the best-head envelope first.
Only sources with stable, material linear-versus-envelope ranking reversals
should advance to an OI comparison. No threshold relaxation, head retuning,
or reserved-backbone use is authorized by this result.
