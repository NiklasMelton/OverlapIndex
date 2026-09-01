# Real-image continuous shortlist result

The frozen UTKFace age panel **failed its primary decision rule**, but the failure is diagnostic rather than an implementation failure.

## Result

Across the five disjoint training cohorts, linear ridge was the best frozen reference head for all 80 backbone-by-cohort evaluations. Consequently, the best-head envelope and the linear-probe ranking were identical in this panel.

| Method | Mean Spearman | Top-5 opportunity regret | Top-5 mean regret | Top-5 material precision |
|---|---:|---:|---:|---:|
| broad RFF | 0.8035 | 0.00000 | 0.00000 | 1.000 |
| SAVE+RFF | 0.7212 | 0.00000 | 0.02079 | 0.880 |
| bounded top-2k fill | 0.7453 | 0.00000 | 0.01759 | 0.880 |
| linear probe | 1.0000 | 0.00000 | 0.00000 | 1.000 |

The bounded-minus-linear-probe top-five mean-regret contrast was `+0.01759`, with training-seed-stability 95% interval `[+0.00515, +0.03004]`. Positive regret favors the linear probe. Opportunity regret remained exactly zero: the bounded shortlist always retained the best backbone.

Broad RFF alone selected the exact oracle top five in every cohort. Structured filling changed nothing in two cohorts, but in three cohorts it replaced `dinov2-small` with a lower-envelope backbone. The degradation therefore came from applying the filler when the broad ranking was already correct, not from failure of the broad retrieval view.

## Interpretation

This real panel validates several important pieces:

- the 16-backbone extraction and ranking pipeline is complete and deterministic at the artifact level;
- the top-10 retrieval bound is genuinely active;
- broad RFF can produce an excellent real-image linear-regime shortlist without fitting a supervised head;
- the current unconditional SAVE filler is not universally safe.

It does **not** test the intended nonlinear advantage, because no nonlinear reference head won for any candidate. The next real panel should therefore have a predeclared target/dataset where nonlinear heads materially contribute to the best-head envelope. Separately, the current result motivates an outcome-blind rule that retains broad ordering when broad and structured evidence disagree or when broad ranking is stable, rather than always invoking structured filling.

## Runtime accounting caveat

Broad and SAVE scores were computed by one shared evaluator call. The raw table assigns the complete shared call to broad and zero incremental seconds to SAVE so bounded total cost is charged exactly once. The displayed zero for SAVE is therefore **incremental shared-path cost**, not standalone SAVE runtime.

## Artifact identity

- Frozen implementation commit: `2af1ce3`
- Protocol: `d1bf2401ea9da29c1bca4f34de8fcdd2886cba705bdc34770aff7dea39b3a680`
- Raw results: `64e3046d2cd36061b504c4fb870f55f5637e414a91700d0589dbc994e8a971a3`
- Summary: `1663fb1b2ac01b3190767309b2f5ad20c7244055bfbb205d91bc7cd397798cf5`
- Report: `e66d6a4dae272e68fdef8c1fccb626c791620b546a5847038a9775d65935db69`
- Manifest: `12b813bde607c05f1c0c912a0ffd8d0bc986c1e44b049ed115996fb7a60b9b79`

