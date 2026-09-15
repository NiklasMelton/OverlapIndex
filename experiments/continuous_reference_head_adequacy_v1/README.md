# Continuous reference-head adequacy V1

This frozen post-diagnostic calibration checks whether the practical head
library can recover the known direct and signed-pair signals in the completed
matched-geometry panel.

Two fixed-capacity heads select one feature or one pair using training rows
only, then tune Ridge regularization inside the same three-fold inner-CV
contract. Their held-out predictions are compared with the analytic conditional
mean. The resulting envelope is diagnostic; it does not revise or promote any
earlier continuous-OI candidate.

