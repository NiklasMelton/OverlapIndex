# Continuous cached-embedding smooth-head audit V1

The first cached-embedding audit showed that PCA-32 materially suppresses linear
predictive performance, but its scalable nonlinear diagnostics (kNN and Extra
Trees) did not beat Ridge.  That does not test the smooth nonlinear family that
won the archived PCA-32 screen.

This reference-only follow-up adds exact RBF-SVR to four predeclared views:
PCA-128 and the full cached embedding at 600 and 2,000 training rows.  RBF
distance matrices are computed once per fold and reused as precomputed kernels,
making the wider representation practical without approximating the model.
Linear, kNN, and Extra Trees scores are consumed unchanged from the completed
information audit.

Before interpreting outcomes, six calibration replays verify that precomputed
RBF kernels reproduce the archived direct-kernel PCA-32 predictions.  No
OverlapIndex method or reserved backbone is evaluated, and this diagnostic cannot
promote a method.

