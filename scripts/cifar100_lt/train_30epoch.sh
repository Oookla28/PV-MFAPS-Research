#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to a writable directory for CIFAR-100 download/cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/artifacts/results/cifar100_lt/ir100_30epoch}"
export PYTHONPATH="$REPO_ROOT/src/validated_runtime:$REPO_ROOT/src:${PYTHONPATH:-}"
python3 "$REPO_ROOT/experiments/cifar100_lt/cifar100_lt_runner.py" --data-root "$DATA_ROOT" --output "$OUTPUT_ROOT" --epochs 30 --imbalance-ratio 100 --batch-size "${BATCH_SIZE:-128}" --seeds "${SEEDS:-3407,3408,3409}" --arms "${ARMS:-erm,ldam_drw,logit_adjustment,fixed_k,pv_mfaps}" --execution "${EXECUTION:-fused}" --gpu "${GPU_ID:-0}"
