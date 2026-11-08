#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CIFAR_ROOT:?Set CIFAR_ROOT to a writable data directory}"
OUT="${OUTPUT_ROOT:-$REPO_ROOT/artifacts/minimal_cifar100}"
PYTHONPATH="$REPO_ROOT/experiments/cifar100_lt" python "$REPO_ROOT/experiments/cifar100_lt/cifar100_lt_preflight.py" \
  --data-root "$CIFAR_ROOT" --output "$OUT/PREFLIGHT.json" --imbalance-ratio 100 \
  --depth 32 --width 16 --augmentation zero --gpu "${GPU_ID:-0}"
PYTHONPATH="$REPO_ROOT/experiments/cifar100_lt" python "$REPO_ROOT/experiments/cifar100_lt/cifar100_lt_runner.py" \
  --data-root "$CIFAR_ROOT" --output "$OUT/smoke" \
  --arms erm,balanced_softmax,ldam_drw,logit_adjustment --seeds "${SEED:-3407}" \
  --epochs 1 --batch-size 128 --limit-batches 2 --lr 0.1 --depth 32 --width 16 \
  --augmentation zero --schedule standard_multistep --warmup-epochs 5 \
  --milestones 160 180 --weight-decay 0.0002 --ldam-switch-epoch 160 \
  --execution fused --gpu "${GPU_ID:-0}"
