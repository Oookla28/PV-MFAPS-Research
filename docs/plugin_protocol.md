# Balanced Softmax + PV-MFAPS plugin registration

Date: 2026-11-06

## Registered question

Does a Balanced Softmax objective, applied consistently to native, detached
inner-solver, reference, and fused correction losses, preserve the macro-F1
of the Balanced Softmax baseline while improving the worst-tail metric under
the same measured update budget?

This is a plugin question. It is separate from the earlier Plant-only
TF-GPS/PV-MFAPS claim and does not inherit its algorithm gate.

## Fixed protocol

- Dataset: PlantSPD E01, 69 classes, FIT/HELD = 4184/1046, counts 5--344.
- Partitions: 0 and 1.
- Training seeds: 3407, 3408, 3409.
- FIT is used for calibration and training; HELD is indexed once after FIT
  freeze and never used for selection.
- Backbone, checkpoint seed, optimizer, encoder update count, batch size,
  action grid, and wall/FLOP accounting match the fixed-k control.
- Fused and legacy implementations must pass real-checkpoint equivalence
  before matched execution.

## Required arms

1. `balanced_softmax` (baseline)
2. `bs_pcfa` (DP actions, fused)
3. `bs_fixed_k_matched_B02`
4. `native_balanced` and `effective_number` controls
5. `logit_adjustment` control
6. `native_plus_pv_mfaps` to isolate the prior from the scheduler
7. `two_forward_native` to isolate extra native forward/backward budget from
   reference correction

## Gate

The plugin passes only if all of the following hold on HELD, averaged over
both partitions and three seeds:

- `bs_pcfa` tail-10% is higher than `balanced_softmax` and
  `bs_fixed_k_matched_B02` under the same measured wall-clock budget;
- macro-F1 is no lower than `balanced_softmax - 0.01`;
- the improvement is not explained by extra FLOPs or extra physical encoder
  forwards;
- FIT tail improvement is reproduced on HELD;
- all manifests report `held_reads=0` before freeze and one held read after
  freeze, with SHA binding intact.

If the tail gate fails, report a Pareto or trade-off result only. If the macro
gate fails, do not call it a successful plugin even when tail improves.

## Current evidence

- **EXECUTED**: one real-checkpoint, one-step, FIT-only fused smoke at
  `<original-project-root>/pv_mfaps_artifacts/bs_pcfa_real_smoke_p1_s3407_v3/`.
- **EXECUTED**: finite-action DP and brute-force test.
- **EXECUTED**: Balanced Softmax loss contract test.
- **EXECUTED**: remote plugin control preflight covering all eight registered
  arms, including registered counters and static action legality.
- **REGISTERED_NOT_EXECUTED**: the multi-seed matched plugin matrix and HELD
  evaluation.
- **UNVERIFIED**: tail gain, macro preservation, generalization, SOTA, and
  cross-domain transfer.

The offline action file
`<original-project-root>/pv_mfaps_artifacts/BS_PCFA_ACTION_POLICY_P1_S3407.json`
is a FIT proxy registration only. It is not an efficacy result and is not
used by the running A100 pilot.

## Non-claims

No community SOTA, universal improvement, Macro-F1 generalization guarantee,
or CIFAR-LT/ImageNet-LT transfer claim is registered here.
