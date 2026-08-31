# Continuous OI oracle bottleneck diagnostic

This is a mechanism experiment, not a deployable method comparison. It uses
privileged generator state to determine where continuous OI loses ranking
information.

Each panel independently crosses four signal-quality levels with four nuisance
levels. OI receives either the observed representation, nuisance-free signal
channels, minimal sufficient coordinates, or a fixed quadratic lift. Matched
4- versus 8-prototype configurations test topology capacity. Every held-out
fold records actual loss, permutation-null loss, loss ratio, and final OI so
normalization can be diagnosed separately.

Because nuisance-free oracle views deliberately tie candidates that differ
only in nuisance scale, their primary metric is quality ordering within each
fixed nuisance level. Nuisance ordering at fixed quality is reported
separately and may correctly be undefined for an oracle view.

Run after the protocol and runner are committed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
CONTINUOUS_OI_COMMIT=$(git rev-parse --short HEAD) PYTHONPATH=. \
python experiments/continuous_oracle_diagnostic/run.py \
  --output artifacts/continuous_oracle_diagnostic
```

The runner writes atomic per-panel checkpoints and refuses to overwrite
terminal artifacts.
