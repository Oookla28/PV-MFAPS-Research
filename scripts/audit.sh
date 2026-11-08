#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to a completed run directory}"
SUMMARY="${SUMMARY:-$RUN_DIR/PR_PV_SUMMARY.md}"
python3 "$REPO_ROOT/src/evaluation/claims_audit.py" --root "$REPO_ROOT" --out "$RUN_DIR" --summary "$SUMMARY" --policy "${ACTION_POLICY:-$REPO_ROOT/configs/policies/PR_POLICY_eps001.json}" --audit-json "$RUN_DIR/PR_PV_CLAIMS_EVIDENCE_AUDIT.json" --audit-md "$RUN_DIR/PR_PV_CLAIMS_EVIDENCE_AUDIT.md"
