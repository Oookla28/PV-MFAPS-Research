#!/usr/bin/env python3
"""Freeze all FIT-phase TF-GPS artifacts before the one-time HELD read."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfgps_gpu_lib import hash_file, write_json


def collect(root: Path):
    rows = []
    for phase in ("pilot", "matched"):
        d = root / phase
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            try:
                obj = json.loads(p.read_text())
            except Exception as exc:
                rows.append({"path": str(p), "status": "INVALID_JSON", "error": repr(exc)})
                continue
            rows.append({
                "path": str(p), "phase": phase, "arm": obj.get("arm"),
                "method": obj.get("method"), "partition": obj.get("partition"),
                "checkpoint_seed": obj.get("checkpoint_seed"),
                "training_seed": obj.get("training_seed"),
                "status": obj.get("status"), "held_reads": obj.get("held_reads"),
                "validation": (obj.get("execution_manifest") or {}).get("status"),
                "state_path": obj.get("state_path"),
                "sha256": hash_file(p),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    rows = collect(root)
    invalid = [r for r in rows if r.get("status") not in ("EXECUTED",)]
    executed = [r for r in rows if r.get("status") == "EXECUTED"]
    missing_state = [r for r in executed if not r.get("state_path") or not Path(r["state_path"]).exists()]
    held_leaks = [r for r in executed if r.get("held_reads") != 0]
    freeze = {
        "status": "PASS" if executed and not missing_state and not held_leaks else "FAIL",
        "scope": "FIT-phase artifact freeze before the single HELD evaluation",
        "held_reads": 0,
        "registered_config": {
            "held_policy": "all FIT arms and states frozen; HELD indexed once afterwards",
            "exclude_from_summary": ["INVALID_CONFIG", "RUNNING", "INVALID_JSON"],
        },
        "execution_manifest": {
            "artifact_count": len(rows), "executed_count": len(executed),
            "invalid_count": len(invalid), "missing_state_count": len(missing_state),
            "held_leak_count": len(held_leaks),
            "artifacts": rows,
            "executed": [{"path": r["path"], "state_path": r["state_path"]} for r in executed],
            "invalid": invalid,
        },
        "coverage_status": {
            "status": "proxy_only_unverified",
            "note": "freeze covers FIT-phase evidence only; held evidence is produced once after this manifest",
        },
    }
    write_json(args.output, freeze)
    print(json.dumps({"status": freeze["status"], "executed": len(executed),
                      "invalid": len(invalid), "missing_state": len(missing_state),
                      "held_leaks": len(held_leaks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
