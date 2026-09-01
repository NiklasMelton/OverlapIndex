# Winner-preserving reliability-gate result

## Decision

**PASS on the frozen four-group developmental panel.** The formulation
established better top-five shortlist regret than broad OI while retaining the
complete broad winner group and all top-one metrics exactly.

| Method | Spearman | Top-1 regret | Top-1 exact | Top-5 regret | Top-5 precision |
|---|---:|---:|---:|---:|---:|
| broad RFF OI | 0.6587 | 0.01514 | 0.560 | 0.02239 | 0.760 |
| bounded 2k OI | 0.6679 | 0.01514 | 0.560 | 0.01886 | 0.792 |
| fixed winner-preserved Borda | 0.6839 | 0.01514 | 0.560 | 0.01557 | 0.791 |
| LODO winner-preserved gate | 0.6810 | 0.01514 | 0.560 | 0.01515 | 0.788 |
| oracle winner-preserved gate | 0.6917 | 0.01514 | 0.560 | 0.01284 | 0.811 |
| unconstrained 50/50 Borda | 0.6670 | 0.03106 | 0.507 | 0.01574 | 0.791 |
| linear probe | 0.9656 | 0.00063 | 0.960 | 0.00155 | 0.960 |

Gate-minus-broad top-five regret was `-0.007235`, 95% interval
`[-0.012919, -0.000937]`. Top-five opportunity regret also improved:
`-0.000836`, interval `[-0.001924, -0.000080]`. Gate-minus-broad top-one
regret was exactly zero in every bootstrap draw.

Preserving the broad winner did not sacrifice the shortlist gain and improved
overall rank correlation relative to the unconstrained gated blend. This
supports the architecture of two distinct ranking layers: broad OI supplies
the immutable winner, while the reliability-gated blend fills the remaining
shortlist.

The result remains developmental and does not establish linear-probe parity.
The hybrid exceeded linear-probe top-five regret by `+0.013606`, 95% interval
`[+0.008880, +0.018511]`.

## Next gate

Freeze one final ridge model on all four development groups and evaluate it,
without refitting, on a new continuous target. The selected next target is a
deterministic DTD texture-gradient statistic over the already outcome-free
common-ten-backbone embedding bank. Because DTD classification evidence has
previously been observed, that test is untouched only with respect to the new
continuous target—not a claim of an entirely untouched dataset.

## Reproducibility

- Frozen implementation commit: `2ebf3df96db230362613cd1a570a027e32b2e13b`
- Protocol SHA-256: `2feff731275571403d0aa6839d5cea26e16dc4f1233d2aebffd8883f17040973`
- Summary SHA-256: `df62b137828cddd0c3c0b0c23e0d0564f6bcc9f5552cd6dfe94bbeec6af0e39c`
- Report SHA-256: `44b793c7eab643c7a61cb855c1474bd8715edcf0adc9f56c04534b0fa149f913`
- Manifest SHA-256: `8ca9a85e6e400271a077b42cb554d03bb9eb03e3deb256487cf7b0f0be6d2940`
- Artifact reconstruction: exact

