# Independent box-occupancy panels

This package adds two natural continuous regression groups without extracting
new embeddings: GTSRB sign ROI occupancy and FGVC-Aircraft object bounding-box
occupancy. It reuses the outcome-free, hash-bound FUSED3 embedding registry but
constructs new target-decile-stratified, disjoint regression cohorts.

Every panel uses the exact ten-backbone intersection shared with the earlier
continuous panels. Embeddings receive the unchanged row-L2 and training-only
rank-32 PCA preprocessing. The broad RFF and SAVE scores, four reference heads,
and best-head envelope are unchanged.

```bash
PYTHONPATH=. python -m experiments.continuous_real_bbox_panels.prepare
PYTHONPATH=. python -m experiments.continuous_real_bbox_panels.run --resume
```

The panel result is developmental evidence. Its five repeats measure disjoint
training-cohort stability against a shared official-test cohort, not five
independent datasets.

