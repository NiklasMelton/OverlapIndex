# Continuous real-image age panel

This is the first real-image regression validation of the bounded shared continuous-OI shortlist. It uses UTKFace age as a genuine scalar target and 16 frozen pretrained backbone candidates, so the top-five rule's broad top-10 retrieval bound is active rather than vacuous.

The official UTKFace page describes the filename age labels, 0–116 age range, and non-commercial research license. This study uses an immutable third-party mirror of the official aligned/cropped archive because the original download link is unavailable. It retains only ages 0–89 and balances nine decade bins. Results must not be used outside the dataset's research-only license.

The outcome-free input bank has one 225-image evaluation cohort and five non-overlapping 225-image training cohorts. All six splits contain 25 images from each decade bin. Candidate embeddings are row-L2 normalized and reduced to 32 dimensions by training-only PCA. The four frozen reference heads—linear ridge, quadratic ridge, tuned KNN, and tuned RBF SVR—define the best-head envelope.

Run with the archived Vertebrae extraction environment and offline weights:

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /Users/niklasmelton/code/vertabrae/.venv/bin/python \
  -m experiments.continuous_real_age_panel.extract --resume

PYTHONPATH=. LOKY_MAX_CPU_COUNT=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/niklasmelton/Library/Caches/pypoetry/virtualenvs/vertebrae-XeUus8FY-py3.9/bin/python \
  -m experiments.continuous_real_age_panel.run --resume
```

This panel measures one dataset and uses a shared evaluation cohort, so its bootstrap intervals describe training-sample stability only. A pass is a gate to broader real-data validation, not a public integration decision.

