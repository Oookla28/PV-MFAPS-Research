# Public release manifest

This repository is the reader-facing minimum reproducible unit.

## Included

- Core model, policy, runtime, training, and evaluation modules.
- CIFAR-LT and PlantSPD entry points.
- Configuration and action-policy files.
- DP and PC-FA contract tests.
- Small result summaries and static figures.
- Data-source, theory, module, plugin, and reproduction documentation.
- Citation metadata and license.

## Excluded

- Research-process journals and internal audit logs.
- GPU orchestration scripts and private shell wrappers.
- Raw images, checkpoints, encoder states, server paths, SSH information, and
  credentials.

External data and model states are supplied through the paths documented in
README.md. The public results are reference summaries. A full rerun produces
new output directories and new freeze/evaluation manifests.

