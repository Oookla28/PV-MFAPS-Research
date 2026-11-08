#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 experiments/cifar100_lt/cifar100_lt_runner.py \
  --data-root "$DATA_ROOT" \
  --output "$OUTPUT_ROOT" \
  --epochs "${EPOCHS:-8}" \
  --imbalance-ratio "${IMBALANCE_RATIO:-100}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --seeds "${SEEDS:-3407}" \
  --arms fd_sdpa \
  --execution fused \
  --gpu 0 \
  --fd_recalibrate_every "${FD_RECALIBRATE_EVERY:-2}" \
  --fd_calib_batches "${FD_CALIB_BATCHES:-2}" \
  --fd_grid "${FD_GRID:-0,1}" \
  --fd_reference_k "${FD_REFERENCE_K:-1}" \
  --fd_inner_lr "${FD_INNER_LR:-0.1}" \
  --fd_lambda_cost "${FD_LAMBDA_COST:-0.001}" \
  --fd_lambda_switch "${FD_LAMBDA_SWITCH:-0.001}" \
  --fd_margin "${FD_MARGIN:-0.0}"
