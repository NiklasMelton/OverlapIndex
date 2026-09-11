# Results

The frozen real-annotation discovery screen completed 36 panels and 180
candidate rows. None of the 12 targets met every admission criterion.

The most informative near misses were:

- Oxford-IIIT Pet mask area was recoverable and nonlinearly favored, but the
  training-only head-family choice was not reliable enough (0.533 versus the
  frozen 0.60 minimum).
- Aircraft horizontal center strongly favored kNN, but the best-head envelope
  was not recoverable on held-out data.
- Several box or mask targets were recoverable but remained primarily linear.

This rejects the idea that a difficult-looking annotation or a nonlinear
winner alone is enough. A useful ranking panel must jointly have recoverable
utility, a material nonlinear advantage, stable head selection, and real
between-backbone spread.

Canonical artifact:
`artifacts/continuous_structured_head_discovery/manifest.json`.

