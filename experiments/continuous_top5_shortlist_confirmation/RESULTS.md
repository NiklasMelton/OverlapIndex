# Continuous top-five shortlist confirmation result

Date: 2026-08-31

The frozen confirmation executed from commit
`379785a52addd5519c181840c4a48031b880d708` and completed all 30 unused-seed
panels. The overall decision is **fail**: the primary regret improvement
replicated, but two safety gates did not pass.

## Result

| Method | Top-1 regret | Top-3 mean regret | Top-5 opportunity regret | Top-5 mean regret | Top-5 material precision |
|---|---:|---:|---:|---:|---:|
| broad RFF | .01386 | .10480 | .0000003 | .11045 | .633 |
| SAVE+RFF current | .13468 | .12723 | .03747 | .10643 | .633 |
| broad winner + SAVE fill | .01386 | .10327 | .0000003 | .09782 | .640 |
| linear probe | .33498 | .26226 | .05533 | .17617 | .580 |

Winner-fill minus broad-RFF top-five mean utility regret was `-.01263` R2,
95% interval `[-.02471, -.00107]`; the primary superiority gate passed.
Top-one regret and top-five opportunity regret matched broad RFF exactly.

Material precision improved by `.00667` at the point estimate, but its 95%
interval `[-.03333, +.04667]` crossed the frozen `-.02` noninferiority
margin. More importantly, family behavior was heterogeneous:

- absolute value: `-.07247` R2 mean-regret improvement;
- quadratic product: `-.05867` improvement;
- radial quadratic: `-.01582` improvement;
- linear: no change;
- sinusoid: `+.00399` degradation;
- triangle wave: `+.06719` degradation.

The triangle-wave degradation exceeded the frozen `.03` family-safety limit.
Thus, the depth-dependent average gain is real, but unbounded SAVE+RFF filling
is not a universal top-five shortlist rule. Broad RFF continues to carry
important periodic ordering information below the winner.

## Verification

- `raw_results.json`: `b442e67a13cfb6c701090e5390f5c28a7553654f0520feca5a540520d191605c`
- `summary.json`: `aeb73edd8febb12ca03fbb3824d04273076abc0356c190bc2a45ebb281d1afee`
- `report.md`: `dca26051a4779143d3c8c379d2f23ff0b98890bea28faf887d180c786b2c8623`
- `manifest.json`: `b0f4603f7b1b854abf1b1c27df96d1003906bcd235d0c5c821bbd0a855261886`

Independent recomputation reproduced the summary exactly. The protocol and
authorizing audit validate, all current source hashes match the raw and
manifest identities, all terminal hashes match, all 30 checkpoints have the
same source identity, and the recorded implementation commit equals the
actual Git HEAD.

One nonfatal scikit-learn ridge ill-conditioning warning occurred during a
reference-head fit. The fit and panel completed, and no selector, checkpoint,
or artifact validation failed.
