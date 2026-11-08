#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${DATA_ROOT:-}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/artifacts/results/run_$(date +%Y%m%d_%H%M%S)}"
export REPO_ROOT DATA_ROOT CHECKPOINT_ROOT OUTPUT_ROOT
if [[ -n "$DATA_ROOT" ]]; then
  export PLANT_E01_ROOT="$DATA_ROOT/E01"
  export PLANT_SPD_ROOT="$DATA_ROOT/Plant_SPD"
  export FIT_LABELS_PATH="$DATA_ROOT/Plant_SPD/11_f1_f4_A_forensics_v1/inputs/FIT_LABELS.npy"
fi
export PIPELINE_DIAG="$REPO_ROOT/artifacts/manifests"
export TFGPS_PCFA_FUSED="${TFGPS_PCFA_FUSED:-1}"
export TFGPS_CUDA_WARMUP="${TFGPS_CUDA_WARMUP:-1}"
export TFGPS_VARIANT="${TFGPS_VARIANT:-a100_fp32_fused}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONPATH="$REPO_ROOT/src/validated_runtime:$REPO_ROOT/src:${PYTHONPATH:-}"
mkdir -p "$OUTPUT_ROOT" "$PIPELINE_DIAG"
