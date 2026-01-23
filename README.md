This python package implements the OverlapIndex (OI), an Incremental Cluster Validity
index for identifying the degree of overlap of data classes. The OI is 1 optimal and ranges
from 0 to 1. A value of 1.0 indicates no overlap whatsoever in the data. A value of 0.5
indicates complete overlap. A value less than 0.5 indicates a degenerate case in the
data.

The OI can be used for many purposes:

- It can be used as an iCVI to provide insight into the quality of an unsupervised
 clustering partition.

- It can be used to determine the quality of class separation in raw, labeled data.

- It can be used to monitor representational separation in the layers of a deep neural
network as learning progresses.

- And it can be used to evaluate backbone models for transfer learning. Where the
optimal backbone will have the highest OI score.

