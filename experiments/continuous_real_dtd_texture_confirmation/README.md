# DTD texture-gradient new-target confirmation

This experiment applies the already frozen all-development winner-preserving gate to a new deterministic continuous target on DTD images. The gate is not refit, and the ranking reference remains the best held-out R2 across the frozen linear, quadratic, kNN, and RBF heads.

DTD classification outcomes existed before this experiment. The holdout is therefore the continuous texture-gradient target and its rankings—not the dataset or embedding bank as a whole.

```bash
LOKY_MAX_CPU_COUNT=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
PYTHONPATH=. python -m experiments.continuous_real_dtd_texture_confirmation.run
```

The frozen gate must improve broad OI's top-five mean regret with an upper 95% bound below zero, preserve broad OI's top-one result exactly, remain noninferior on top-five opportunity regret, and produce the complete five-seed panel.
