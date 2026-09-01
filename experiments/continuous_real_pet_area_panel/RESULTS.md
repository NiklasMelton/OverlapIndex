# Oxford-IIIT Pet foreground-occupancy result

## Decision

**FAIL.** The bounded shared continuous-OI shortlist did not match the linear-probe comparator on the frozen top-five mean-regret criterion.

This panel used a naturally continuous spatial target: the fraction of pixels in each official Oxford-IIIT Pet trimap labeled foreground or not-classified boundary. Images, targets, cohorts, backbone checkpoints, evaluator settings, and decision rules were frozen at commit `3dd48f48cbb42e52fe304d3eada923226707fc84` before embeddings or head/OI outcomes were computed.

## Primary results

| Method | Mean Spearman | Top-5 opportunity regret | Top-5 mean regret | Material precision | Panel seconds |
|---|---:|---:|---:|---:|---:|
| Broad RFF | 0.7659 | 0.00000 | 0.03326 | 0.800 | 6.044 |
| SAVE/RFF | 0.6453 | 0.00000 | 0.06358 | 0.720 | 0.000 incremental |
| Bounded top-2k | 0.7518 | 0.00000 | 0.04615 | 0.760 | 6.044 |
| Linear probe | 0.9588 | 0.00000 | 0.00000 | 1.000 | 0.305 |

The paired bounded-minus-linear-probe top-five mean-regret estimate was `+0.046154`, with a 95% training-cohort stability interval of `[+0.038039, +0.054221]`. Positive values favor the linear probe, so the primary criterion failed decisively.

The bounded-minus-broad estimate was `+0.012895`, with interval `[-0.022428, +0.048219]`. The bounded SAVE filler did not establish an improvement over the broad score and was worse at the point estimate.

## Head-family structure

Across all 80 backbone/cohort cells, the best reference head was:

- linear ridge: 39
- RBF SVR: 36
- quadratic ridge: 3
- KNN: 2

This is the first real-image panel in this series where nonlinear heads won a majority of candidate cells. However, among the 25 oracle top-five positions, 20 were linear-ridge winners and five were RBF winners. The strongest candidates were therefore largely linearly accessible even though nonlinear heads helped many weaker candidates.

The linear probe recovered the exact oracle top-five set in all five cohorts. For the five RBF-winning candidates that entered an oracle top five, their linear scores were still high enough to preserve their shortlist positions. Thus linear-probe ranking remained an excellent proxy for the best-head envelope without itself fitting the winning RBF head.

## OI failure mode

All three OI variants included the oracle-best candidate somewhere in every five-candidate shortlist, so top-five opportunity regret was zero. Their failure was ordering and filler quality:

- broad RFF top-five material precision averaged `0.80`;
- bounded top-2k precision averaged `0.76`;
- SAVE/RFF precision averaged `0.72`;
- the linear probe achieved `1.00`.

Broad RFF did not rank the oracle-best candidate first in any cohort; its mean top-one opportunity regret was `0.07553`. SAVE/RFF ranked it first in three of five cohorts, but its remaining shortlist was poorer. The bounded rule deliberately preserves the broad winner, so it inherited broad's top-one error while allowing SAVE to displace some better broad candidates below it.

This reveals a more specific limitation than “continuous OI cannot detect nonlinearity.” Broad RFF sees useful dependence—the oracle winner is always retrieved—but the scalar dependence strength is not calibrated well enough to order the strongest representations. SAVE sometimes improves winner identification but is not reliable as an unconditional shortlist filler. The linear probe succeeds because linear quality is highly correlated with envelope quality among the leading backbones, even when an RBF head provides the final score for some of them.

## Runtime

The shared OI computation averaged `6.044 s` per 16-backbone panel versus `0.305 s` for the linear probe, approximately `19.8x` slower. These are descriptive shared-table timings rather than a separately counterbalanced resource benchmark, but they are not compatible with the current speed objective.

## Recommendation

Do not promote the current bounded broad-plus-SAVE formulation. The next experiment should focus on calibration or aggregation of the broad dependence score among the retrieved candidates, not on another unconditional SAVE filler. It should also explicitly separate two objectives:

1. best-candidate retrieval, where broad RFF is useful but imperfect at rank one; and
2. five-candidate shortlist quality, where current filler logic loses to the linear proxy.

A useful diagnostic is to compare per-candidate broad RFF, SAVE, linear-probe score, and best-head envelope across the leading retrieval pool, then test a frozen cross-fitted calibration/aggregation rule on held-out datasets rather than on another split of this panel.

## Verification and artifact identity

- Input manifest SHA-256: `acec9fbd00a3c7445e5c492204d3421ce753b9af5090ddf5c744fbc03ad1a151`
- Cohort SHA-256: `de44c91481890b348ed8bf53bd096c24e4a80700601b90b9f48adc4b5a9ef7cb`
- Embedding manifest SHA-256: `214c4064ad3a3b9b7ed2bbf44accad3304c3ea4099ae1034d1fe4b7f0656efac`
- Protocol SHA-256: `10e85c4f737e911621f8a4153389dbae8a954c74818fe8bbc7c10009d954e898`
- Raw results SHA-256: `3d70e90f422d653d799035adb1859663f387484d41f071bd67fe71d28c1a7e3f`
- Summary SHA-256: `d1d9cc8c4d4fad4e557c91d8866ed19448f243e1dde5d13a270c8943723bfcca`
- Report SHA-256: `85c122a9ea77946e8deab71e5c2ac9eecb5f0aac8c670ed0b50becd34c216a59`
- Terminal manifest SHA-256: `ad941e577e8eba0e5ee08d7499e5c878e3a3ef2f72387b41bfe82694f4ff641e`
- Continuous focused suite: `221 passed`
