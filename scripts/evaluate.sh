#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to a completed run directory}"
SPLIT="${SPLIT:-fit}"
SPLIT_UPPER=$(printf "%s" "$SPLIT" | tr "[:lower:]" "[:upper:]")
OUT="${EVAL_OUTPUT:-$RUN_DIR/${SPLIT_UPPER}_EVALUATION.json}"
if [[ "$SPLIT" == "held" ]]; then
  FREEZE_MANIFEST="${FREEZE_MANIFEST:-$RUN_DIR/PR_FIT_FREEZE_MANIFEST.json}"
  python3 "$REPO_ROOT/src/evaluation/evaluate.py" --run-dir "$RUN_DIR" --split held --freeze-manifest "$FREEZE_MANIFEST" --output "$OUT" --gpu "${GPU_ID:-0}"
else
  python3 "$REPO_ROOT/src/evaluation/evaluate.py" --run-dir "$RUN_DIR" --split fit --output "$OUT"
fi
