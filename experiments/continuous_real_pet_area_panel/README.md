# Oxford-IIIT Pet foreground-occupancy panel

This is the next natural-target real-image panel for the bounded shared continuous-OI shortlist. The scalar target is the fraction of pixels in the official Oxford-IIIT Pet trimap labeled animal foreground or not-classified boundary. It is an independently annotated spatial property, not a transformation of a semantic label.

The cohort contains one 200-image evaluation split and five disjoint 200-image training splits. Every split has 20 images from each of ten equal-count foreground-occupancy strata. Candidate embeddings, training-only PCA-32, the four reference head families, and the bounded top-five continuous-OI rule are unchanged from the prior real-image panels.

The [official dataset page](https://www.robots.ox.ac.uk/~vgg/data/pets/) describes 37 pet categories with variation in scale, pose, and lighting and supplies pixel-level trimap annotations. Follow the archive README and original-image terms for research use.

After downloading the two exact official archives into the default input root:

```bash
PYTHONPATH=. /Users/niklasmelton/Library/Caches/pypoetry/virtualenvs/vertebrae-XeUus8FY-py3.9/bin/python \
  -m experiments.continuous_real_pet_area_panel.prepare

PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /Users/niklasmelton/code/vertabrae/.venv/bin/python \
  -m experiments.continuous_real_pet_area_panel.extract --resume

PYTHONPATH=. LOKY_MAX_CPU_COUNT=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/niklasmelton/Library/Caches/pypoetry/virtualenvs/vertebrae-XeUus8FY-py3.9/bin/python \
  -m experiments.continuous_real_pet_area_panel.run --resume
```

The panel is single-dataset development evidence. Its five repeats measure sensitivity to disjoint training cohorts against one shared evaluation cohort, not external-dataset generalization.
