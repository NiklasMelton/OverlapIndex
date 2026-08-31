# Continuous characteristic-kernel alignment

This isolated experiment tests a single null-adjusted centered-kernel-
alignment score as a head-family-neutral estimate of target information in a
representation. Feature and target kernels are fixed equal mixtures of three
RBF bandwidths estimated from training-fold distances.

It uses a new synthetic seed block and does not modify public
`ContinuousOverlapIndex` behavior.

```bash
python -m experiments.continuous_kernel_alignment.run \
  --output artifacts/continuous_kernel_alignment
```

