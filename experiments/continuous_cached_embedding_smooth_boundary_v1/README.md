# Continuous cached-embedding smooth boundary V1

The compact smooth-head audit produced one backbone-ranking conflict: JHU Crowd,
600 training rows, row-L2 normalization, and PCA-128.  Its top-1 regret was only
0.0042 R² and every selected RBF-SVR configuration touched a search boundary.

This frozen three-cell diagnostic repeats exactly that view for DINOv2-Small,
ResNet-50, and ConvNeXt-Tiny using the complete previously qualified RBF grid and
one predeclared higher-C value.  All preprocessing, folds, cached rows, and linear
scores are unchanged.  It evaluates no OverlapIndex method and cannot promote a
method.

