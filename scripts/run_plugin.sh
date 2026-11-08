#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing E01/ and Plant_SPD/}"
POLICY="${ACTION_POLICY:-$REPO_ROOT/configs/policies/PR_POLICY_eps001.json}"
mkdir -p "$OUTPUT_ROOT/plugin"
python3 "$REPO_ROOT/src/validated_runtime/bs_plugin_runner.py" \
  --root "$REPO_ROOT" --out "$OUTPUT_ROOT/plugin" \
  --action-policy-dir "$REPO_ROOT/configs/policies" --partitions "${PARTITIONS:-0,1}" \
  --seeds "${TRAINING_SEEDS:-3407,3408,3409}" \
  --steps "${STEPS:-8}" --batch-size "${BATCH_SIZE:-8}" \
  --arms "${PLUGIN_ARMS:-balanced_softmax,bs_fixed_k_matched_B02,bs_pcfa,native_balanced,effective_number,logit_adjustment}"
