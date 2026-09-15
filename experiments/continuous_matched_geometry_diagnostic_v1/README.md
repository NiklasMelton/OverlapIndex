# Continuous matched-geometry diagnostic V1

This frozen synthetic diagnostic compares representations that contain the
same noisy measurement of a predictive latent variable through four different
geometries. Direct, rotated, anisotropic, and sign-gated candidates at
`q=0.85` have exactly the same analytic conditional-mean R2. A sign-gated
`q=0.55` candidate provides a known material ordering.

Current observed OI, broad RFF, and SAVE OI use the exact evaluators from the
preceding mechanism diagnostic. A separate four-head held-out evaluator asks
whether an analytically invertible encoding remains equally accessible to the
finite head library at the available sample size.

The result distinguishes score sensitivity to geometry, actual selection of a
weaker representation, and finite-head accessibility. It is diagnostic only.

