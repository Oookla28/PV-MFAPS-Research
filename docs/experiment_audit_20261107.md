# ARIS Experiment Audit — 2026-11-07

**Overall verdict: WARN**  
**Integrity status: WARN**  
Reviewer: fresh GPT-6-Astra ultra; same-family/provisional.

## What passed

- Ground truth is dataset-provided PlantSPD labels; no prediction-derived ground truth was found.
- Metrics are raw per-class recall/F1 aggregates; no self-normalization was found.
- Main PR clean result files exist and their reported values match the claims audit and blog data JSON.
- The manifest-linked clean freeze records 24 states with FIT `held_reads=0`, followed by a 24-state post-freeze HELD read of 25,104 samples with zero errors.

## High-severity integrity finding

The main runner `code_pr_pv_20261107/pr_pv_mfaps_runner.py:7-11` prepends the sibling `code_pv_2026-11-06` directory and imports `pivot_runner`, `tfgps_gpu_runner`, `tfgps_gpu_lib`, and `bs_plugin_runner` from there. The original freeze manifest hashes files under the root `code/` directory; at least `tfgps_gpu_lib.py` and `tfgps_gpu_runner.py` have different SHA256 values in the imported directory. This is a reproducibility/provenance mismatch, not evidence that the numbers were fabricated. A retrospective hash manifest is saved at `PlantGPU/.aris/audit_20261107/RETROSPECTIVE_EXECUTED_IMPORT_CODE_MANIFEST.json`; it does not retroactively make the original freeze manifest correct.

## Scope and status

- **EXECUTED:** main PR clean matched chain, freeze/HELD/claims audit, finite-action DP exactness, epsilon sweep, real-checkpoint Fused equivalence, exploratory McNemar capture.
- **REGISTERED_NOT_EXECUTED:** CIFAR-LT/ImageNet-LT, learnable spectral geometry, second SPD project, A100 correction full chain, independent semantic reviewer.
- **UNVERIFIED:** universal/generalization/SOTA, multi-budget empirical Pareto, robust-cost theorem, alpha-aware optimality, and 50-step main efficacy.
- **QUARANTINED/INVALID:** collision runs, invalid configurations, and the original A100 run for any universal algorithm claim.

## Supplemental artifact warning

The local 50-step directory contains a summary and manifests but not the referenced `matched/` rows or `states/` files. Until the source artifact is re-synced and hashes are checked, its values are metadata-only and must not be used as reproducible evidence. The per-class recall integrity check remains useful as a diagnostic of the arrays that were available, but it does not repair missing provenance.

## Required fixes before submission-grade claims

1. Rebind or rerun the main clean chain with the exact imported code snapshot in the freeze manifest.
2. Keep the 0.9974 clean wall ratio separate from old 1.1913 and 1.0379 protocol values.
3. Scope “one-time HELD” to the clean PR chain; later McNemar/50-step reads are exploratory.
4. Replace “Pareto frontier” with “aggregate matched-budget point” unless a multi-budget sweep is executed.
5. Do not promote the 50-step pilot, CIFAR, correction chain, mechanism theorem, or SOTA/generalization claims.
