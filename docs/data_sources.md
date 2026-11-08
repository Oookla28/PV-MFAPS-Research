# Data sources and derived protocols

## PlantSeg / PlantSPD

The original resource is *A Large-Scale In-the-wild Dataset for Plant Disease
Segmentation*, Scientific Data, 2026, DOI
[10.1038/s41597-025-06513-4](https://doi.org/10.1038/s41597-025-06513-4).
PlantSeg was collected from in-the-wild imagery and provides disease-area
segmentation masks, metadata, licenses, and a 70/10/20 train/validation/test
split. The resource covers 7,774 diseased images, 34 host plants, 69 disease
types, and 115 host-disease combinations.

This project does not redistribute PlantSeg images. PlantSPD is a derived
multi-view covariance/classification protocol built from the permitted source
images and metadata. The public package stores only contracts, preprocessing
interfaces, hashes, and aggregate evidence summaries. The local FIT/HELD
protocol uses 69 disease classes with 4,184 FIT samples and 1,046 HELD samples.
The two paired FIT partitions preserve per-class counts while changing the
observations used for calibration and training.

The original dataset page and metadata must be consulted for license and
download conditions before reproducing the PlantSPD preprocessing.

## CIFAR-LT

The Stage-TRR transfer experiments use CIFAR-100 and CIFAR-10 with the
long-tailed imbalance-ratio-100 construction implemented in the repository.
The public package does not include the binary archives. The runner downloads
or accepts an external data root, computes a combination SHA256 over the
training files, and records the hash in the preflight and freeze manifests.

## Evaluation split

Training outputs never include test/HELD reads. Each group is frozen first,
then a separate evaluator reads the evaluation split exactly once for that
group. Exploratory pilot outputs and invalid configurations are excluded from
the aggregate tables and figure scripts.
