# COCO pose opportunity replication

This prospective development experiment tests whether the LSP torso-orientation
opportunity replicates on independently sourced COCO 2017 person-keypoint
images. It uses only the five scout backbones and the previously frozen
four-head envelope. It does not fit an OverlapIndex method or inspect reserved
backbones.

The selected person is the largest eligible non-crowd person in each image.
Torso sine/cosine share one cohort; bilateral knee extension uses a separate
cohort. Training comes from COCO `train2017` and evaluation from `val2017`.
Images are downloaded only after the cohort identities have been deterministically
selected from the hash-locked annotation archive.
