# File map

## Core algorithm

- `src/model/pv_mfaps_dp_solver.py` provides the envelope components, stability term and finite-action DP primitives.
- `src/model/pv_mfaps_dp.py` provides the Pareto-frontier solver and action backtracking.
- `src/model/pcfa_policy.py` provides the cost-aware per-class action policy.
- `configs/policies/` contains FIT-only action vectors used by the main and plugin runs.

## Training and execution

- `src/validated_runtime/pr_pv_mfaps_runner.py` is the main shared-policy trainer.
- `src/validated_runtime/tfgps_gpu_runner.py` contains the batch loop, action execution and counters.
- `src/validated_runtime/pcfa_policy.py` contains the cost-aware PC-FA policy and action rules.
- `src/validated_runtime/pcfa_cost_model.py` and `src/validated_runtime/pcfa_cost_smoke.py` implement the measured-cost calibration and smoke checks.
- `src/validated_runtime/pcfa_equivalence_smoke.py` compares legacy and Fused execution on a real checkpoint.
- `src/validated_runtime/tfgps_gpu_lib.py` contains device selection, FIT calibration and shared metrics.

## Evaluation and evidence

- `src/evaluation/evaluate.py` is the user-facing FIT/HELD entry point.
- `src/validated_runtime/held_eval.py` performs the one-time post-freeze HELD read.
- `src/validated_runtime/make_freeze_manifest.py` freezes FIT artifacts and state hashes.
- `src/evaluation/claims_audit.py` runs the run-local claims/evidence audit.

## Plugin

- `src/validated_runtime/bs_plugin_runner.py` applies Balanced Softmax consistently to native, reference, solver and Fused correction losses.
- `docs/plugin_experiment.md` describes the independent arms and shared cost protocol.
- `configs/plugin/bs_pcfa.yaml` declares the plugin matrix.

## Evidence summaries

- `artifacts/results/plantspd_pr_pv_mfaps_summary.json` is the compact reader-facing PlantSPD matched-result table.
- `artifacts/results/pv_mfaps_solver_benchmark.json` is the compact finite-action solver benchmark.
- `artifacts/results/fused_execution_summary.json` is the compact Fused/Legacy timing and equivalence summary.
- `artifacts/results/TRR_AUDITED_PREFIX10_VS_LA_SUMMARY.json` and `artifacts/results/TRR_AUDITED_C10_PREFIX10_VS_LA_SUMMARY.json` are the 30-epoch CIFAR-LT cross-dataset summaries.
- `artifacts/results/cifar100_ir100_200_v2_summary.json` is the three-seed 200-epoch standard matrix.
- `artifacts/results/TRR_AUDITED_HORIZON50_SUMMARY.json` records the separately measured 50-epoch horizon boundary.

Large checkpoints, raw profiler traces and internal orchestration logs are not part of
the minimal reader-facing package. The JSON summaries retain the configuration and
source-hash fields needed to trace each reported number.
