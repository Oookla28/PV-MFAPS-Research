#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CIFAR_ROOT:?Set CIFAR_ROOT to a writable data directory}"
PYTHONPATH="$REPO_ROOT/experiments/cifar100_lt" python - <<'PY'
import os
from pathlib import Path
from cifar100_lt_runner import ensure_data
print(ensure_data(Path(os.environ['CIFAR_ROOT'])))
PY
