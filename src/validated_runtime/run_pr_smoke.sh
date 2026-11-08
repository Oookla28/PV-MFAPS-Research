#!/usr/bin/env bash
set -e
ROOT=<remote-root>
CODE=$ROOT/code_pr_pv_20261107
OUT=$ROOT/out/pr_pv_mfaps_20261107
export PLANT_E01_ROOT=$ROOT/data/E01
export PLANT_SPD_ROOT=$ROOT/data/Plant_SPD
export FIT_LABELS_PATH=$ROOT/data/Plant_SPD/11_f1_f4_A_forensics_v1/inputs/FIT_LABELS.npy
export PIPELINE_DIAG=$ROOT/diagnostics
export TFGPS_PCFA_FUSED=1
export TFGPS_CUDA_WARMUP=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TFGPS_VARIANT=a100_fp32_fused
export PYTHONPATH=$ROOT/code_pv_2026-11-06:$ROOT/code:$CODE
mkdir -p "$OUT/logs"
exec <remote-root>_PARENT/miniconda3/envs/leafguard/bin/python "$CODE/pr_pv_mfaps_runner.py" --root "$ROOT" --out "$OUT/smoke" --action-policy "$OUT/action_policy/PR_POLICY_eps010.json" --partitions 0,1 --seeds 3407 --steps 8 --batch-size 8 --arms pr_pv_mfaps
