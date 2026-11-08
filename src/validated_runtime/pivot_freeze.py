#!/usr/bin/env python3
"""Freeze all pivot FIT artifacts with input/code/parameter/state hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def jsha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf8")).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.root); out = Path(args.out)
    arms = []
    for phase in ("pilot", "matched"):
        for p in sorted((out / phase).glob("*.json")):
            d = json.loads(p.read_text())
            arms.append((phase, p, d))
    invalid = [(str(p), d.get("status")) for _, p, d in arms if d.get("status") != "EXECUTED"]
    missing_state = []
    held_leaks = []
    artifacts = []
    param_rows = []
    for phase, p, d in arms:
        sp = d.get("state_path")
        if not sp or not Path(sp).exists():
            missing_state.append({"path": str(p), "state_path": sp})
        if int(d.get("held_reads", -1)) != 0:
            held_leaks.append(str(p))
        param = {
            "arm": d.get("arm"), "method": d.get("method"), "phase": d.get("phase"),
            "partition": d.get("partition"), "checkpoint_seed": d.get("checkpoint_seed"),
            "training_seed": d.get("training_seed"),
            "registered_config": d.get("registered_config"), "policy": d.get("policy"),
        }
        ph = jsha(param)
        param_rows.append({"path": str(p), "parameter_sha256": ph})
        artifacts.append({
            "phase": phase, "path": str(p), "arm": d.get("arm"),
            "method": d.get("method"), "partition": d.get("partition"),
            "checkpoint_seed": d.get("checkpoint_seed"),
            "training_seed": d.get("training_seed"),
            "status": d.get("status"), "held_reads": d.get("held_reads"),
            "state_path": sp, "state_sha256": sha(sp) if sp and Path(sp).exists() else None,
            "json_sha256": sha(p), "parameter_sha256": ph,
        })
    input_paths = [
        root / "data/E01/input/rgb_views_uint8.npy",
        root / "data/E01/input/meta.npz",
        root / "data/E01/history_origin/prepared/SPLITS.npz",
        root / "data/E01/assets/resnet18-f37072fd.pth",
        root / "data/E01/source/runner_v2.py",
        out / "prompt.md",
    ]
    code_paths = [root / "code" / n for n in (
        "pivot_runner.py", "pivot_preflight.py", "pivot_kmfaps_precheck.py",
        "pivot_freeze.py", "tfgps_gpu_lib.py", "tfgps_gpu_runner.py",
        "plant_e0_gpu.py", "pcfa_cost_model.py", "pcfa_policy.py",
        "pcfa_cost_smoke.py", "pcfa_runner.py", "pcfa_audit.py",
        "pcfa_equivalence_smoke.py", "test_pcfa_fused_equivalence.py",
        "pcfa_actionspace_selector.py", "test_pcfa_policy_contract.py",
        "held_eval.py")]
    status = "EXECUTED" if (arms and not invalid and not missing_state and not held_leaks) else "INVALID_CONFIG"
    manifest = {
        "status": status,
        "scope": "Pivot FIT artifact freeze before the single HELD evaluation; no HELD sample indexed",
        "held_reads": 0,
        "registered_config": {
            "fit_tuning_only": True,
            "held_one_time_after_freeze": True,
            "trust_deltas": [0.05, 0.10, 0.20],
            "kmfaps_kappa": "A/(2*sigma-A), denominator>0",
            "kmfaps_b": "(kappa+beta*sigma)^2", "kmfaps_budget": 0.2,
            "kmfaps_k_grid": [0, 1, 2, 4],
        },
        "execution_manifest": {
            "artifact_count": len(arms), "invalid": invalid,
            "missing_state": missing_state, "held_leaks": held_leaks,
            "input_sha256": {str(p): sha(p) for p in input_paths if p.exists()},
            "code_sha256": {str(p): sha(p) for p in code_paths if p.exists()},
            "aux_sha256": {str(p): sha(p) for p in [
                out / "COST_MODEL.json", out / "PCFA_COST_PRECHECK.json",
                out / "PCFA_COST_SMOKE.json", out / "PCFA_FUSED_EQUIVALENCE_SMOKE.json",
            ] if p.exists()},
            "parameter_manifest": param_rows,
            "parameter_manifest_sha256": jsha(param_rows),
            "artifacts": artifacts,
            "executed": [{"path": a["path"], "state_path": a["state_path"]}
                         for a in artifacts if a.get("status") == "EXECUTED"],
            "kmfaps_precheck_status": (json.loads((out / "PIVOT_KMFAPS_PRECHECK.json").read_text()).get("status")
                                       if (out / "PIVOT_KMFAPS_PRECHECK.json").exists() else "MISSING"),
            "boundary_present": (out / "BOUNDARY_KMFAPS.json").exists(),
        },
        "coverage_status": {
            "status": "proxy_only_unverified",
            "note": "FIT freeze only; HELD is read once after this manifest",
        },
    }
    write_json(args.output, manifest)
    print(json.dumps({"status": status, "n": len(arms), "invalid": len(invalid),
                      "missing_state": len(missing_state), "held_leaks": len(held_leaks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
