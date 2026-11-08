#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CIFAR_ROOT:?Set CIFAR_ROOT to a writable data directory}"
OUT="${OUTPUT_ROOT:-$REPO_ROOT/artifacts/cifar100_ir100_200}"
SEEDS="${SEEDS:-3407,3408,3409}"
GPU="${GPU_ID:-0}"
PYTHONPATH="$REPO_ROOT/experiments/cifar100_lt" python "$REPO_ROOT/experiments/cifar100_lt/cifar100_lt_preflight.py" \
  --data-root "$CIFAR_ROOT" --output "$OUT/PREFLIGHT.json" --imbalance-ratio 100 \
  --depth 32 --width 16 --augmentation zero --gpu "$GPU"
PYTHONPATH="$REPO_ROOT/experiments/cifar100_lt" python "$REPO_ROOT/experiments/cifar100_lt/cifar100_lt_runner.py" \
  --data-root "$CIFAR_ROOT" --output "$OUT/baseline_fused" \
  --arms erm,balanced_softmax,ldam_drw,logit_adjustment --seeds "$SEEDS" \
  --epochs 200 --batch-size 128 --lr 0.1 --depth 32 --width 16 --augmentation zero \
  --schedule standard_multistep --warmup-epochs 5 --milestones 160 180 --weight-decay 0.0002 \
  --ldam-switch-epoch 160 --ldam-C 0.5 --ldam-beta 0.9999 --ldam-scale 30 \
  --logit-adjustment-tau 1.0 --execution fused --gpu "$GPU"
PYTHONPATH="$REPO_ROOT/experiments/cifar100_lt" python "$REPO_ROOT/experiments/cifar100_lt/evaluate.py" \
  --run-dir "$OUT/baseline_fused" --data-root "$CIFAR_ROOT" \
  --output "$OUT/baseline_fused/HELD_EVAL.json" --gpu "$GPU"
