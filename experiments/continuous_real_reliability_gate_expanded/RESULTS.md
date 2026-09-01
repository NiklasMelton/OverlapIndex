# Expanded reliability-gate result

## Decision

**FAIL under the frozen joint decision rule.** The primary top-five shortlist
criterion passed, but the top-one safety criterion failed.

This is post-outcome developmental evidence over four dataset groups and five
target families. Every group was restricted to the same ten backbones. The
feature schema, fixed 50/50 blend, ridge penalty, and zero decision threshold
were carried forward unchanged from the earlier two-group analysis.

## Aggregate result

| Method | Spearman | Top-1 regret | Top-1 exact | Top-5 regret | Top-5 precision |
|---|---:|---:|---:|---:|---:|
| broad RFF OI | 0.6587 | 0.01514 | 0.560 | 0.02239 | 0.760 |
| bounded 2k OI | 0.6679 | 0.01514 | 0.560 | 0.01886 | 0.792 |
| fixed 50/50 Borda | 0.6670 | 0.03106 | 0.507 | 0.01574 | 0.791 |
| LODO reliability gate | 0.6615 | 0.02885 | 0.480 | 0.01533 | 0.788 |
| oracle panel gate | 0.6779 | 0.02462 | 0.540 | 0.01301 | 0.811 |
| linear probe | 0.9656 | 0.00063 | 0.960 | 0.00155 | 0.960 |

## What the added datasets established

The reliability gate improved top-five mean regret relative to broad OI by
`-0.007059`, with a 95% interval of `[-0.012393, -0.000937]`. The effect was
favorable in three of four independent groups:

- Oxford Pet: `-0.016316`
- GTSRB: `-0.011312`
- FGVC-Aircraft: `-0.008941`
- UTKFace: `+0.000636`

Top-five opportunity regret also improved by `-0.001421`, 95% interval
`[-0.003677, -0.000080]`. Thus the earlier two-group interval crossing zero
was primarily an evidence-volume problem: the unchanged gate has a repeatable
shortlist benefit.

The same intervention damaged winner selection. Gate-minus-broad top-one
opportunity regret was `+0.013707`, with a 95% interval of
`[+0.003255, +0.024159]`. The harm was largest on Aircraft (`+0.049586`) and
GTSRB (`+0.013556`). The blend can replace broad OI's leading candidate even
when its broader top-five composition improves.

The gate agreed with the per-panel broad-versus-blend oracle in 14 of 25
panels: 4/10 UTKFace, 3/5 Oxford Pet, 5/5 GTSRB, and 2/5 Aircraft. Its useful
shortlist improvement therefore comes from average rank movement, not a
universally accurate binary regime classifier.

## Updated interpretation

This result supports an explicit product-level separation of objectives:

- **Shortlisting:** the adaptive blend is now supported over the expanded
  developmental panel. It produces materially lower top-five regret than
  broad OI.
- **Best-single-candidate selection:** do not use the blend. Broad OI, or the
  winner-preserving bounded method, retains substantially better top-one
  behavior.

The adaptive gate remains far behind the linear probe on both objectives; its
top-five regret exceeded the probe by `+0.013781`, 95% interval
`[+0.009056, +0.019039]`. It is therefore not a universal linear-probe
replacement.

The next experiment should not retune this gate. It should freeze two separate
user-facing estimands: a winner-preserving top-one mode and a shortlist mode,
then validate each on a genuinely untouched natural dataset group. A hybrid
rule can preserve the broad winner while using the gate/blend only to fill
positions 2–5; that directly targets the observed tradeoff and should be
tested without changing the gate's inference features.

## Reproducibility

- Frozen implementation commit: `cd66d9b946faf585d588e78bd9ba4331af4fd536`
- Protocol SHA-256: `0e17b979ceede91f18447ebf57e7bb7be32e8109ebf1244af4d7cede767452ab`
- Summary SHA-256: `9e1625fc943c855a05ce9f96d094e3c5a6f26b8c8140d6e357a1746b1cc4495e`
- Report SHA-256: `ef823aa88b4b4757b9d84dcac7fc2a4b8e0aca15079fa748bf6bd237ea9a9e34`
- Manifest SHA-256: `1ef2d0b494c9b3ac553a992b4cb9eaee73cf466d8e47b583c01bbd13d3f56af4`
- Terminal reconstruction: exact from all five source families

