# Continuous multiscale graph experiment

This isolated experiment tests whether a single target-continuity statistic,
integrated over fixed local, intermediate, and coarse feature-neighbor graph
scales, can rank representations against the frozen best-head envelope.

It uses new synthetic seeds. The implementation does not change public
`ContinuousOverlapIndex` behavior.

```bash
python -m experiments.continuous_multiscale_graph.run \
  --output artifacts/continuous_multiscale_graph
```

