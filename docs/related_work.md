# Related work and distinction

This note keeps the literature framing attached to the corresponding research
module. The repository does not treat a familiar loss or a standard optimizer
as a new contribution.

## Long-tailed recognition

LDAM-DRW introduces label-distribution-aware margins and deferred
re-weighting. Balanced Softmax models the class prior inside the softmax
normalizer. Logit Adjustment adds a prior-dependent logit offset. These methods
operate on the prediction objective.

Stage-TRR operates on a different interface. It changes the encoder feedback
during an initial training phase by refining the readout on detached features,
then returns to the standard logit-adjusted objective. The comparison is
therefore against the same LA objective with the same data, seeds, schedule and
backbone.

## SPD representations

SPDNet and related matrix-function pipelines use congruence maps, matrix
logarithms, matrix powers or bilinear layers to construct features on the SPD
manifold. The present project uses the Loewner derivative of the chosen matrix
function to measure how readout residuals are transmitted back to the encoder.
The finite-action allocator is attached after the SPD representation and does
not claim to introduce a new SPD manifold or a new matrix logarithm.

## Adaptive and bilevel optimization

Variable-projection and bilevel methods solve an inner problem more accurately
than an outer problem. Stage-TRR uses the same two-timescale structure in a
finite-action training schedule. Its object of study is the class-conditional
encoder feedback induced by an approximate readout. The conditional switching
bound is stated for a finite surrogate-risk objective and is not presented as a
global convergence theorem.

## Systems execution

The Legacy/Fused comparison separates logical correction actions from physical
encoder execution. Fused reuses one encoder feature tensor and performs
detached correction at the readout. FLOPs, physical encoder forwards and
CUDA-event wall-clock are reported separately so that the system result is not
confused with an accuracy result.

## Data provenance

The real-world plant setting is derived from PlantSeg, whose Scientific Data
article describes disease segmentation under complex backgrounds, viewpoints
and illumination. The standard long-tail extensions use the public CIFAR-10
and CIFAR-100 archives with an explicit imbalance-ratio construction. Every
dataset-specific runner writes a data hash into its manifest.
