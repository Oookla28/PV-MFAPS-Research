# Static figures

This directory contains the publication-ready static figures and the LaTeX table
used by the public reproduction package.

The figures are derived from the small JSON summaries in artifacts/results.
The source JSON files identify the dataset, seeds, horizon, evaluation reads,
and comparison arms. Original images and checkpoints are not needed to inspect
the published summaries.

Included figures:

- stage_trr_cross_dataset.pdf
- stage_trr_horizon_boundary.pdf
- stage_trr_paired_seed_deltas.pdf
- fused_system_equivalence.pdf

The corresponding numerical summaries are in artifacts/results. The training
and evaluation entry points are in experiments/cifar100_lt and
src/validated_runtime. Figure-generation orchestration and internal plotting
scripts remain outside the public package.

