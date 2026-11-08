#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export PYTHONPATH="$REPO_ROOT/src/validated_runtime:$REPO_ROOT/src:${PYTHONPATH:-}"
python3 "$REPO_ROOT/src/validated_runtime/test_pv_mfaps_dp.py"
python3 "$REPO_ROOT/src/validated_runtime/pr_pv_mfaps_math_smoke.py"
