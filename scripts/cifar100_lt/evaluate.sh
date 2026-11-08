#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to a CIFAR100-LT run directory}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the CIFAR cache}"
python3 "$REPO_ROOT/experiments/cifar100_lt/evaluate.py" --run-dir "$RUN_DIR" --data-root "$DATA_ROOT" --output "${OUTPUT:-$RUN_DIR/CIFAR100_LT_EVALUATION.json}"
