# Larger structured opportunity benchmark

This development-only benchmark asks whether real continuous annotation tasks
contain enough nonlinear head advantage to make backbone ranking against a
best-head envelope meaningfully different from linear-probe ranking.

The protocol freezes 20 target channels across CUB-200-2011 bird parts, Leeds
Sports Pose, COCO 2017 person keypoints, Oxford-IIIT Pet masks, and Pascal VOC
2007 object boxes. It evaluates five scout backbones with linear ridge,
quadratic ridge, distance-weighted kNN, and RBF SVR. It never fits OverlapIndex
and never evaluates the five reserved backbones.

Four datasets reuse already acquired images or embedding banks, but every
target channel in this benchmark is new. Pascal VOC is a fresh dataset. That
makes this a prospectively frozen target-channel benchmark, not an untouched
dataset confirmation.

Run, in order:

```bash
PYTHONPATH=. python -m experiments.continuous_large_structured_opportunity_benchmark.prepare
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python -m experiments.continuous_large_structured_opportunity_benchmark.extract
PYTHONPATH=. python -m experiments.continuous_large_structured_opportunity_benchmark.run
```

The protocol hash sidecar must match before preparation, extraction, or
outcome evaluation. All target results are reported; there is no replacement,
tuning, or reserved-backbone stage.
