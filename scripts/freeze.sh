#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to a completed FIT run directory}"
python3 "$REPO_ROOT/src/validated_runtime/make_freeze_manifest.py" --root "$REPO_ROOT" --output "$RUN_DIR/PR_FIT_FREEZE_MANIFEST.json"
printf 'FIT freeze manifest written to %s\n' "$RUN_DIR/PR_FIT_FREEZE_MANIFEST.json"
