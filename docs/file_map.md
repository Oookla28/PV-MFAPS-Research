# File map

## Core algorithm

- `src/model/pv_mfaps_dp_solver.py` provides the envelope components, stability term and finite-action DP primitives.
- `src/model/pv_mfaps_dp.py` provides the Pareto-frontier solver and action backtracking.
- `src/model/pcfa_policy.py` provides the cost-aware per-class action policy.
- `configs/policies/` contains FIT-only action vectors used by the main and plugin runs.

## Training and execution

- `src/validated_runtime/pr_pv_mfaps_runner.py` is the main shared-policy trainer.
- `src/validated_runtime/tfgps_gpu_runner.py` contains the batch loop, action execution and counters.
- `src/validated_runtime/pcfa_runner.py` contains the cost-aware PC-FA arm.
- `src/validated_runtime/pcfa_equivalence_smoke.py` compares legacy and Fused execution on a real checkpoint.
- `src/validated_runtime/tfgps_gpu_lib.py` contains device selection, FIT calibration and shared metrics.

## Evaluation and evidence

- `src/evaluation/evaluate.py` is the user-facing FIT/HELD entry point.
- `src/validated_runtime/held_eval.py` performs the one-time post-freeze HELD read.
- `src/validated_runtime/make_freeze_manifest.py` freezes FIT artifacts and state hashes.
- `src/evaluation/claims_audit.py` runs the run-local claims/evidence audit.

## Plugin

- `src/validated_runtime/bs_plugin_runner.py` applies Balanced Softmax consistently to native, reference, solver and Fused correction losses.
- `docs/plugin_protocol.md` describes its independent arms and shared cost protocol.
- `configs/plugin/bs_pcfa.yaml` declares the plugin matrix.

## Evidence directories

- `artifacts/results/main_pr_pv_mfaps/` holds the main PR/PV summary, rows, calibration and audit-linked JSON files.
- `artifacts/results/theory/` holds finite-action DP and mechanism materials.
- `artifacts/results/fused_equivalence/` holds real-checkpoint Fused equivalence materials.
- `artifacts/results/plugin/` holds plugin artifacts.
- `artifacts/results/supplemental_mcnemar/` holds exploratory paired-prediction arrays and summary.
- `artifacts/results/supplemental_50step_metadata_only/` holds the metadata-only long-horizon supplement.
