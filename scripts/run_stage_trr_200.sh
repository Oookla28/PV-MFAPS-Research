#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CIFAR_ROOT:?Set CIFAR_ROOT to a writable data directory}"
OUT="${OUTPUT_ROOT:-$REPO_ROOT/artifacts/cifar100_ir100_200/stage_trr10}"
SEEDS="${SEEDS:-3407 3408 3409}"
GPU="${GPU_ID:-0}"
mkdir -p "$OUT"
for seed in $SEEDS; do
  PYTHONPATH="$REPO_ROOT/experiments/cifar100_lt" python "$REPO_ROOT/experiments/cifar100_lt/trr_audited_runner.py" \
    --data-root "$CIFAR_ROOT" --output "$OUT" --epochs 200 --trr-epochs 10 \
    --batch-size 128 --depth 32 --width 16 --augmentation zero \
    --schedule standard_multistep --warmup-epochs 5 --milestones 160 180 \
    --lr-encoder 0.1 --lr-head 0.05 --weight-decay 0.0002 \
    --calibration-fraction 0.1 --seed "$seed" --gpu "$GPU"
done
PYTHONPATH="$REPO_ROOT/experiments/cifar100_lt" python "$REPO_ROOT/experiments/cifar100_lt/trr_audited_evaluate.py" \
  --run-dir "$OUT" --data-root "$CIFAR_ROOT" --output "$OUT/HELD_EVAL.json" --gpu "$GPU"
