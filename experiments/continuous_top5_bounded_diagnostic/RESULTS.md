# Bounded top-five diagnostic result

The post-confirmation diagnostic passed every frozen criterion. The dynamic
`bounded_2k` rule preserves the broad-RFF winner and uses SAVE+RFF only to rank
the remaining members of broad RFF's complete top-10 retrieval pool.

Across the 30 stored confirmation panels, mean top-five regret was:

- broad RFF: `0.1104466`
- unbounded winner fill: `0.0978176`
- bounded `2k`: `0.0886252`

The paired bounded-minus-broad mean-regret contrast was `-0.0218214`, with a
frozen 95% interval of `[-0.0336637, -0.0103371]`. Top-one and opportunity
regret were exactly unchanged from broad RFF. Mean material precision improved
from `0.6333` to `0.6667`; its lower paired bound was nonnegative to numerical
precision.

The family-specific failure was localized. Relative to broad RFF, triangle-wave
mean regret changed by `+0.0120375`, below the frozen `+0.03` limit, versus
`+0.0671919` for the failed unbounded rule. Bounded and unbounded rankings were
identical in the other five families; bounding improved triangle-wave regret by
`0.0551544`. Thus the top-10 retrieval constraint removed harmful low-confidence
SAVE substitutions without discarding the useful replacements in absolute-value,
quadratic-product, and radial-quadratic panels.

This result is post-outcome diagnostic evidence only. It justifies one new
unused-seed bounded-top-five confirmation; it does not promote or integrate the
method.

## Frozen identities

- implementation commit: `c25fdf6a1af8263d2538f489109daa15ee64e49a`
- protocol SHA-256: `08e0578e7fcbb6e6352dfae65cea863efe10a57b3473016569f2d5aefc5fd1ed`
- summary SHA-256: `ab38aef89e2ba09350c876d8d112649f604ee3877a4a7c5d37d12ed93d6d290e`
- report SHA-256: `2e53925c9bd4ad44beb2a40a2642a0c32cc4157bd2421bee0f2948538c92ba97`
- manifest SHA-256: `5fa25f4325d916e5ebadca37570bc030f2fad2a50fcc2efa46c193f2e9887fa9`
