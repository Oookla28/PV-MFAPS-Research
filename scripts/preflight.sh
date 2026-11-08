#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing E01/ and Plant_SPD/}"
python3 - <<'PY'
import os, sys
print('python', sys.version.split()[0])
try:
 import torch, numpy, scipy
 print('torch', torch.__version__)
 print('cuda_available', torch.cuda.is_available())
 if torch.cuda.is_available():
  print('gpu_count', torch.cuda.device_count())
  for i in range(torch.cuda.device_count()): print('gpu', i, torch.cuda.get_device_name(i))
 print('numpy', numpy.__version__)
 print('scipy', scipy.__version__)
except Exception as e:
 print('dependency_check_error', repr(e)); raise
for key in ('PLANT_E01_ROOT','PLANT_SPD_ROOT','FIT_LABELS_PATH'):
 p=os.environ[key]; print(key, p, os.path.exists(p))
PY
python3 "$REPO_ROOT/src/validated_runtime/test_pcfa_policy_contract.py"
python3 "$REPO_ROOT/src/validated_runtime/test_pv_mfaps_dp.py"
python3 "$REPO_ROOT/src/validated_runtime/test_longtail_losses.py"
printf 'preflight complete. Output root: %s\n' "$OUTPUT_ROOT"
