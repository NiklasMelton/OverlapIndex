# Structured-target nonlinear-head discovery

This development-only screen asks a narrower question than whether a dataset
is generally "hard": does a frozen image representation contain a target that
is both recoverable and materially easier for a nonlinear readout than for a
linear ridge readout?

It reuses already frozen real-image embeddings, but derives previously untested
position and shape targets from GTSRB/Aircraft boxes and Oxford-IIIT Pet masks.
Five scout backbones are used for task admission. Five other backbones are
reserved and must not be evaluated by this screen.

The four reference heads are tuned only on each training cohort. A target is
admitted only if it is recoverable, has a stable nonlinear advantage, separates
backbone utilities, and has sufficiently reliable inner-CV head-family
selection. Existing continuous-OI scores are descriptive; this experiment does
not promote a method or constitute confirmation.

Run after verifying the frozen protocol hash:

```bash
PYTHONPATH=. python -m experiments.continuous_structured_head_discovery.run
```

Results are written once to
`artifacts/continuous_structured_head_discovery/`.

