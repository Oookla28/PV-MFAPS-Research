#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing E01/ and Plant_SPD/}"
POLICY="${ACTION_POLICY:-$REPO_ROOT/configs/policies/PR_POLICY_eps001.json}"
mkdir -p "$OUTPUT_ROOT/ablations"
python3 "$REPO_ROOT/src/validated_runtime/test_pcfa_fused_equivalence.py"
python3 "$REPO_ROOT/src/validated_runtime/test_pv_mfaps_dp.py"
python3 "$REPO_ROOT/src/validated_runtime/pr_pv_mfaps_runner.py" \
  --root "$REPO_ROOT" --out "$OUTPUT_ROOT/ablations" \
  --action-policy "$POLICY" \
  --partitions "${PARTITIONS:-0,1}" --seeds "${TRAINING_SEEDS:-3407,3408,3409}" \
  --steps "${STEPS:-8}" --batch-size "${BATCH_SIZE:-8}" \
  --arms "${ABLATION_ARMS:-pr_pv_mfaps,bs_fixed_k_matched_B02,effective_number}"
printf 'Ablation matrix written to %s\n' "$OUTPUT_ROOT/ablations"
