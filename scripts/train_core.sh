#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing E01/ and Plant_SPD/}"
POLICY="${ACTION_POLICY:-$REPO_ROOT/configs/policies/PR_POLICY_eps001.json}"
PARTITIONS="${PARTITIONS:-0,1}"
SEEDS="${TRAINING_SEEDS:-3407,3408,3409}"
STEPS="${STEPS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
mkdir -p "$OUTPUT_ROOT/core"
python3 "$REPO_ROOT/src/validated_runtime/pr_pv_mfaps_runner.py" \
  --root "$REPO_ROOT" --out "$OUTPUT_ROOT/core" \
  --action-policy "$POLICY" --partitions "$PARTITIONS" --seeds "$SEEDS" \
  --steps "$STEPS" --batch-size "$BATCH_SIZE" --arms pr_pv_mfaps
