#!/usr/bin/env python3
"""Server preflight + GPU_ENV_MANIFEST.json for the TF-GPS PlantSPD run."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tfgps_gpu_lib import E0, hash_file, write_json


def sha_or_none(path: Path):
    try:
        return hash_file(path)
    except Exception as exc:
        return f"ERROR:{exc!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--diagnostics", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    diag = Path(args.diagnostics)
    e01 = Path(os.environ.get("PLANT_E01_ROOT", E0.ROOT))
    spd = Path(os.environ.get("PLANT_SPD_ROOT", e01.parent / "Plant_SPD"))
    fit_labels_path = Path(os.environ.get(
        "FIT_LABELS_PATH", spd / "11_f1_f4_A_forensics_v1/inputs/FIT_LABELS.npy"))
    errors = []
    warnings = []
    try:
        import torch
        torch_info = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "cudnn_version": int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
        }
        if not torch.cuda.is_available():
            errors.append("torch.cuda.is_available() is false")
    except Exception as exc:
        torch_info = {"error": repr(exc)}
        errors.append(f"torch import failed: {exc!r}")
    try:
        import scipy
        scipy_version = scipy.__version__
    except Exception as exc:
        scipy_version = f"ERROR:{exc!r}"
        warnings.append("scipy unavailable; CVaR SLSQP check will be skipped")
    nvidia = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    # Data contract from SPLITS and labels, without touching HELD samples.
    data_contract = {}
    try:
        x = np.load(E0.INPUT, mmap_mode="r", allow_pickle=False)
        with np.load(E0.META, allow_pickle=False) as z:
            labels = np.asarray(z["labels"], dtype=np.int64)
        with np.load(E0.SPLITS, allow_pickle=False) as z:
            fit = np.flatnonzero(z["fit_mask"][0])
            held = np.flatnonzero(z["held_mask"][0])
            fit1 = np.flatnonzero(z["fit_mask"][1])
            held1 = np.flatnonzero(z["held_mask"][1])
        counts = np.bincount(labels[fit], minlength=69)
        data_contract = {
            "input_shape": list(x.shape), "input_dtype": str(x.dtype),
            "meta_labels": int(len(labels)),
            "fit_n_p0": int(len(fit)), "held_n_p0": int(len(held)),
            "fit_n_p1": int(len(fit1)), "held_n_p1": int(len(held1)),
            "classes": 69,
            "fit_counts_p0": counts.tolist(),
            "count_min_p0": int(counts.min()), "count_max_p0": int(counts.max()),
            "head_tail_ratio_p0": float(counts.max() / max(counts.min(), 1)),
            "count_sum_p0": int(counts.sum()),
        }
        if len(fit) != 4184 or len(held) != 1046:
            errors.append(f"FIT/HELD contract mismatch p0: {len(fit)}/{len(held)}")
        if len(fit1) != 4184 or len(held1) != 1046:
            errors.append(f"FIT/HELD contract mismatch p1: {len(fit1)}/{len(held1)}")
        if int(counts.min()) != 5 or int(counts.max()) != 344:
            errors.append(f"count contract mismatch: min={counts.min()} max={counts.max()}")
        if int(len(labels)) != 5230:
            errors.append(f"total sample contract mismatch: {len(labels)}")
    except Exception as exc:
        data_contract = {"error": repr(exc)}
        errors.append(f"data contract load failed: {exc!r}")
    fit_labels = {}
    try:
        fl = np.load(fit_labels_path, allow_pickle=False).astype(np.int64)
        fit_labels = {"path": str(fit_labels_path), "n": int(len(fl)),
                      "counts": np.bincount(fl, minlength=69).tolist(),
                      "sha256": sha_or_none(fit_labels_path)}
        if len(fl) != 4184:
            errors.append(f"FIT_LABELS length {len(fl)} != 4184")
        if "fit_counts_p0" in data_contract and fit_labels["counts"] != data_contract["fit_counts_p0"]:
            warnings.append("FIT_LABELS counts differ from SPLITS p0 counts; recorded, not silently reconciled")
    except Exception as exc:
        fit_labels = {"path": str(fit_labels_path), "error": repr(exc)}
        errors.append(f"FIT_LABELS load failed: {exc!r}")
    checkpoints = {}
    for method in ("raw", "log"):
        for part in (0, 1):
            for seed in (3407, 3408, 3409):
                p = E0.ROOT / "runs" / f"pst1r1_p{part}_s{seed}_{method}" / "FINAL_CHECKPOINT.pt"
                checkpoints[str(p)] = {"exists": p.exists(),
                                       "size": int(p.stat().st_size) if p.exists() else None,
                                       "sha256": sha_or_none(p) if p.exists() else None}
                if not p.exists():
                    errors.append(f"missing checkpoint {p}")
    artifacts = {}
    for p in [E0.SRC, E0.BACKBONE, E0.WEIGHTS, E0.INPUT, E0.META, E0.SPLITS]:
        artifacts[str(p)] = {"exists": Path(p).exists(),
                             "sha256": sha_or_none(Path(p)) if Path(p).exists() else None,
                             "size": int(Path(p).stat().st_size) if Path(p).exists() else None}
        if not Path(p).exists():
            errors.append(f"missing required file {p}")
    contract_paths = {
        "tail_contract": diag / "TFGPS_LONGTAIL_CONTRACT.json",
        "cvar_contract": diag / "TFGPS_CVAR_EPIGRAPH_CONTRACT.json",
        "kappa_contract": diag / "KAPPA_MINIMAX_CONTRACT.json",
    }
    contract_status = {}
    for name, p in contract_paths.items():
        if p.exists():
            try:
                d = json.loads(p.read_text())
                contract_status[name] = {"path": str(p), "status": d.get("status"), "sha256": hash_file(p)}
                if d.get("status") != "PASS":
                    errors.append(f"{name} status {d.get('status')}")
            except Exception as exc:
                contract_status[name] = {"path": str(p), "error": repr(exc)}
                errors.append(f"{name} parse failed")
        else:
            contract_status[name] = {"path": str(p), "status": "MISSING"}
            warnings.append(f"{name} not yet produced")
    status = "PASS" if not errors else "BLOCKED_ENV"
    manifest = {
        "status": status,
        "scope": "server environment and PlantSPD data/version contract; no training, no HELD sample indexing",
        "held_reads": 0,
        "registered_config": {
            "fit_tuning_only": True, "held_one_time_eval": True,
            "expected_counts": {"fit": 4184, "held": 1046, "classes": 69,
                                "count_min": 5, "count_max": 344, "head_tail_ratio": 68.8},
            "required_contracts": list(contract_paths),
        },
        "execution_manifest": {
            "hostname": platform.node(), "python": sys.version,
            "executable": sys.executable, "torch": torch_info,
            "scipy": scipy_version, "nvidia_smi": nvidia,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "data_contract": data_contract, "fit_labels": fit_labels,
            "checkpoints": checkpoints, "artifacts": artifacts,
            "contracts": contract_status,
        },
        "coverage_status": {
            "status": "proxy_only_unverified",
            "note": "environment/data version evidence only; class risk proxies remain empirical",
        },
        "errors": errors, "warnings": warnings,
        "note": "If status != PASS, no training arm is launched and BLOCKED_ENV is written.",
    }
    write_json(out / "GPU_ENV_MANIFEST.json", manifest)
    preflight = {
        "status": "PASS" if status == "PASS" else "BLOCKED_ENV",
        "scope": "pre-training static preflight; no training",
        "held_reads": 0,
        "registered_config": manifest["registered_config"],
        "execution_manifest": {
            "env_manifest": str(out / "GPU_ENV_MANIFEST.json"),
            "env_status": status, "errors": errors, "warnings": warnings,
            "contracts": contract_status,
        },
        "coverage_status": manifest["coverage_status"],
    }
    write_json(out / "TFGPS_GPU_PREFLIGHT.json", preflight)
    print(json.dumps({"status": status, "errors": errors, "warnings": warnings,
                      "torch": torch_info, "data_contract": data_contract}, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
