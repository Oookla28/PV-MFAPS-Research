#!/usr/bin/env python3
"""Pivot preflight: GPU visibility, data contract, static contracts, manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected-gpu-name", default=os.environ.get("TFGPS_EXPECT_GPU_NAME", "3090"))
    ap.add_argument("--expected-gpu-count", type=int,
                    default=int(os.environ.get("TFGPS_EXPECT_GPU_COUNT", "2")))
    args = ap.parse_args()
    root = Path(args.root); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    e01 = Path(os.environ.get("PLANT_E01_ROOT", root / "data/E01"))
    spd = Path(os.environ.get("PLANT_SPD_ROOT", root / "data/Plant_SPD"))
    diag = Path(os.environ.get("PIPELINE_DIAG", root / "diagnostics"))
    errors = []
    warnings = []
    import torch
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch_info = {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "visible_count": int(torch.cuda.device_count()),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "cudnn": int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
    }
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.expected_gpu_count:
        errors.append(f"GPU visibility failed: available={torch.cuda.is_available()} "
                      f"count={torch.cuda.device_count()} expected>={args.expected_gpu_count}")
    if any(args.expected_gpu_name not in name for name in torch_info["devices"]):
        errors.append(f"visible devices do not match the registered {args.expected_gpu_name} "
                      f"target: {torch_info['devices']}")
    nvidia = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
                             "--format=csv"], capture_output=True, text=True).stdout
    # Data contract from SPLITS/meta; labels/masks only, no HELD sample indexing.
    x = np.load(e01 / "input/rgb_views_uint8.npy", mmap_mode="r", allow_pickle=False)
    with np.load(e01 / "input/meta.npz", allow_pickle=False) as z:
        labels = np.asarray(z["labels"], dtype=np.int64)
    with np.load(e01 / "history_origin/prepared/SPLITS.npz", allow_pickle=False) as z:
        fit0 = np.flatnonzero(z["fit_mask"][0]); held0 = np.flatnonzero(z["held_mask"][0])
        fit1 = np.flatnonzero(z["fit_mask"][1]); held1 = np.flatnonzero(z["held_mask"][1])
    counts = np.bincount(labels[fit0], minlength=69)
    data_contract = {
        "input_shape": list(x.shape), "input_dtype": str(x.dtype),
        "meta_labels": int(len(labels)),
        "fit_n_p0": int(len(fit0)), "held_n_p0": int(len(held0)),
        "fit_n_p1": int(len(fit1)), "held_n_p1": int(len(held1)),
        "classes": 69, "fit_counts_p0": counts.tolist(),
        "count_min": int(counts.min()), "count_max": int(counts.max()),
        "head_tail_ratio": float(counts.max() / max(counts.min(), 1)),
    }
    if len(fit0) != 4184 or len(held0) != 1046 or len(fit1) != 4184 or len(held1) != 1046:
        errors.append(f"FIT/HELD contract mismatch: p0 {len(fit0)}/{len(held0)} p1 {len(fit1)}/{len(held1)}")
    if int(counts.min()) != 5 or int(counts.max()) != 344 or int(len(labels)) != 5230:
        errors.append(f"count contract mismatch: min={counts.min()} max={counts.max()} total={len(labels)}")
    artifacts = {}
    for p in [e01 / "source/runner_v2.py", e01 / "assets/resnet18_torch_only_v1.py",
              e01 / "assets/resnet18-f37072fd.pth", e01 / "input/meta.npz",
              e01 / "history_origin/prepared/SPLITS.npz", e01 / "input/rgb_views_uint8.npy",
              out / "prompt.md"]:
        artifacts[str(p)] = {"exists": p.exists(), "sha256": sha256(p) if p.exists() else None}
        if not p.exists():
            errors.append(f"missing required artifact {p}")
    code_hashes = {}
    code_names = [
        "tfgps_gpu_lib.py", "tfgps_gpu_runner.py", "pcfa_cost_model.py",
        "pcfa_policy.py", "pcfa_cost_smoke.py", "pcfa_runner.py",
        "pcfa_audit.py", "pcfa_equivalence_smoke.py",
        "test_pcfa_fused_equivalence.py", "test_pcfa_policy_contract.py",
        "pcfa_actionspace_selector.py", "pivot_freeze.py",
    ]
    for p in sorted((root / "code").glob("pivot_*.py")) + [root / "code" / n for n in code_names]:
        if not p.exists():
            errors.append(f"missing code artifact {p}")
            continue
        code_hashes[str(p)] = sha256(p)
    contract_paths = {
        "tail_contract": diag / "TFGPS_LONGTAIL_CONTRACT.json",
        "cvar_contract": diag / "TFGPS_CVAR_EPIGRAPH_CONTRACT.json",
        "kappa_contract": diag / "KAPPA_MINIMAX_CONTRACT.json",
    }
    contracts = {}
    for name, p in contract_paths.items():
        if p.exists():
            d = json.loads(p.read_text())
            contracts[name] = {"path": str(p), "status": d.get("status"), "sha256": sha256(p)}
            if d.get("status") != "PASS":
                errors.append(f"{name} status {d.get('status')}")
        else:
            contracts[name] = {"path": str(p), "status": "MISSING"}
            errors.append(f"{name} missing")
    status = "EXECUTED" if not errors else "BLOCKED_ENV"
    manifest = {
        "status": status,
        "scope": "Pivot preflight: GPU0/1 visibility, FIT/HELD data contract, static contracts; no training and no HELD sample indexing",
        "held_reads": 0,
        "registered_config": {
            "visible_devices": list(range(torch.cuda.device_count())),
            "fit_tuning_only": True, "held_one_time_after_freeze": True,
            "expected_counts": {"fit": 4184, "held": 1046, "classes": 69,
                                "count_min": 5, "count_max": 344, "head_tail_ratio": 68.8},
            "trust_deltas": [0.05, 0.10, 0.20],
            "kmfaps_kappa": "A/(2*sigma-A), denominator>0",
            "kmfaps_b": "(kappa+beta*sigma)^2", "kmfaps_beta": 0.0,
            "kmfaps_budget": 0.2, "kmfaps_k_grid": [0, 1, 2, 4],
        },
        "execution_manifest": {
            "hostname": socket.gethostname(), "python": sys.version,
            "executable": sys.executable, "platform": platform.platform(),
            "torch": torch_info, "visible_devices": list(range(torch.cuda.device_count())),
            "nvidia_smi": nvidia,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "data_contract": data_contract, "artifacts": artifacts,
            "code_sha256": code_hashes, "contracts": contracts,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "errors": errors, "warnings": warnings,
        },
        "coverage_status": {
            "status": "proxy_only_unverified",
            "note": "FIT class proxies are empirical; no HELD access before freeze",
        },
    }
    write_json(out / "PIVOT_GPU_ENV_MANIFEST.json", manifest)
    preflight = {
        "status": status,
        "scope": "Pivot preflight gate before any training",
        "held_reads": 0,
        "registered_config": manifest["registered_config"],
        "execution_manifest": {
            "env_manifest": str(out / "PIVOT_GPU_ENV_MANIFEST.json"),
            "data_contract": data_contract, "contracts": contracts,
            "errors": errors, "warnings": warnings,
        },
        "coverage_status": manifest["coverage_status"],
    }
    write_json(out / "PIVOT_PREFLIGHT.json", preflight)
    print(json.dumps({"status": status, "errors": errors,
                      "torch": torch_info, "data_contract": data_contract}, ensure_ascii=False, indent=2))
    if status != "EXECUTED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
