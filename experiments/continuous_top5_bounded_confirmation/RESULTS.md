# Bounded top-five confirmation result

The unused-seed confirmation passed every frozen primary and safety criterion.
The confirmed method preserves broad RFF's complete winner group and permits
SAVE+RFF to fill the remaining shortlist positions only from broad RFF's
complete top-10 retrieval pool.

Across 30 new panels, mean top-five regret was:

- broad RFF: `0.1181819`
- unbounded winner fill: `0.0810999`
- bounded `2k`: `0.0800649`
- SAVE+RFF current: `0.0998078`
- linear probe: `0.1358121`

The paired bounded-minus-broad mean-regret contrast was `-0.0381169`, with a
frozen 95% interval of `[-0.0493689, -0.0264003]`. Top-one regret was exactly
unchanged from broad RFF (`0.0110735`). The top-five opportunity-regret
increase was small (`0.0010672` point difference; upper 95% bound `0.0032019`)
and passed the frozen `0.01` noninferiority limit. Mean material precision
improved from `0.6467` to `0.6867`, with a nonnegative paired lower bound to
numerical precision.

Family-specific bounded-minus-broad mean-regret differences were:

- absolute value: `-0.0776882`
- linear: `-0.0089454`
- quadratic product: `-0.0557477`
- radial quadratic: `-0.0787233`
- sinusoid: `-0.0128422`
- triangle wave: `+0.0052451`

The maximum family degradation was therefore well below the frozen `+0.03`
limit. Bounded and unbounded rankings were identical in five families. On
triangle wave, bounding reduced regret by `0.0062097`; the aggregate
bounded-minus-unbounded interval was `[-0.0029197, -0.0000214]`.

This confirms the synthetic five-backbone shortlist formulation. It supports
implementation/shared-computation work and real-panel validation, but the
protocol does not treat this synthetic result alone as authorization for public
integration.

## Frozen identities

- implementation commit: `1a90e249b0bfc1f82d3fa77463aa2d83f963bdc3`
- protocol SHA-256: `53bfab23e96d54ddf29c0d3eab9d20f78483a722c527635cc295ea1cd54b9c36`
- raw-results SHA-256: `9ec90831f321463d25fdddf5b05f173e05e6558119a85a18bd4d147b28662745`
- summary SHA-256: `d42044865080bed8476062545fdf39d532aaa50a26becc4a815d871672f6e6fb`
- report SHA-256: `01ac23c16da6d41d5ab677e803c88d2ae5dd3f03e7b191e69d1f1f35dc6dd929`
- manifest SHA-256: `49c318aea87d7c966d2ebcc27162e16a74d7ebbb498688c6ce183e6304361e7c`
