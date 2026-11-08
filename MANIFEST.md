# Release manifest

This release keeps code, small JSON summaries, protocol documents, tests, and
vector figures. It excludes private data, checkpoints, encoder states, SSH
configuration, machine-local paths, and raw server logs.

## Included

- src/ model, engine, evaluation, and validated runtime modules
- experiments/cifar100_lt/ CIFAR-LT runners, audited Stage-TRR runners, and
  offline policy tests
- scripts/ reproducible entry points
- configs/ policy and ablation configuration
- docs/ research timeline, derivations, protocol notes, and claims/evidence
  records
- artifacts/results/ selected frozen summaries and audits
- figures/ scripts, vector PDFs, and LaTeX table
- paper/aaai_stage_trr_20261108/ AAAI-style LaTeX draft and evidence map

## External inputs

PlantSPD E01 data, CIFAR archives, pretrained weights, checkpoints, and large
encoder states must be supplied separately through the paths described in
README.md. Reproduction commands write new outputs under a user-selected
OUTPUT_ROOT.
