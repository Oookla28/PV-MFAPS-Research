#!/usr/bin/env python3
"""FIT-only PC-FA smoke: measured cost/action count consistency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfgps_gpu_lib import write_json
from tfgps_gpu_runner import run_single_arm
from pivot_runner import get_endpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cost-model", required=True)
    ap.add_argument("--precheck", required=True)
    ap.add_argument("--partition", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--tolerance", type=float, default=0.30)
    args = ap.parse_args()
    out = Path(args.out)
    cm = json.loads(Path(args.cost_model).read_text())
    pre = json.loads(Path(args.precheck).read_text())
    pol = pre["execution_manifest"]["policies"][str(args.partition)]
    if not pol.get("valid_nondegenerate") or not pol.get("feasible"):
        raise SystemExit("PC-FA precheck is not valid; smoke must not start")
    caldir = out / "calibration"
    runner, base, head, x, labels, fit, held, counts, cal, ckpt, payload, device = \
        get_endpoint("raw", 3407, args.partition, args.gpu, args.batch_size, caldir)
    r = run_single_arm(
        "pcfa", base, runner, head, x, labels, fit, counts, cal, "raw",
        args.partition, 3407, 3407, args.steps, args.batch_size, device,
        "pilot", out / "pilot", len(held), int(payload.get("epoch", -1)), str(ckpt),
        policy_override=pol, output_prefix="PCFA_COST_SMOKE", pcfa_fused=True)
    expected_total = float(args.steps * args.batch_size * pol["cost_per_sample"])
    actual_total = float(r.get("estimated_training_flops", 0.0))
    ratio = actual_total / max(expected_total, 1e-30)
    variant = (r.get("execution_manifest") or {}).get("pcfa_implementation_variant")
    timing_ok = (r.get("cuda_warmup_enabled") is True and
                 r.get("warmup_seconds") is not None and
                 r.get("cuda_training_seconds") is not None)
    ok = (r.get("status") == "EXECUTED" and
          (r.get("execution_manifest") or {}).get("status") == "PASS" and
          variant == "fused_single_encoder_graph" and timing_ok and
          abs(ratio - 1.0) <= args.tolerance)
    observed = (r.get("execution_manifest") or {}).get("observed", {})
    registered = r.get("registered_config", {})
    result = {
        "status": "EXECUTED" if ok else "UNVERIFIED",
        "scope": "FIT-only fused PC-FA smoke; measured cost vs model and action-count self-check; no HELD",
        "held_reads": 0,
        "pcfa_fused": bool(r.get("pcfa_fused")),
        "pcfa_implementation_variant": variant,
        "registered_config": pol.get("registered_config", {}),
        "execution_manifest": {
            "partition": args.partition, "steps": args.steps, "batch_size": args.batch_size,
            "arm_result": str(out / "pilot" / f"PCFA_COST_SMOKE_raw_p{args.partition}_s3407_pcfa.json"),
            "arm_status": r.get("status"), "validation": (r.get("execution_manifest") or {}).get("status"),
            "expected_total_flops": expected_total, "actual_total_flops": actual_total,
            "flops_ratio": ratio, "flops_tolerance": args.tolerance,
            "actual_wall_seconds": r.get("wall_clock_training_seconds"),
            "cuda_training_seconds": r.get("cuda_training_seconds"),
            "warmup_seconds": r.get("warmup_seconds"),
            "cuda_warmup_enabled": r.get("cuda_warmup_enabled"),
            "pcfa_implementation_variant": variant,
            "model_cost_per_sample": pol.get("cost_per_sample"),
            "fixed_k_cost_per_sample": pol.get("fixed_k_cost_per_sample"),
            "budget_per_sample": pol.get("budget_per_sample"),
            "registered": {k: registered.get(k) for k in
                           ("native_passes", "native_group_passes", "reference_action_passes",
                            "solver_steps", "sample_solver_steps",
                            "physical_encoder_forwards", "physical_encoder_backwards",
                            "physical_reference_actions")},
            "observed": {k: observed.get(k) for k in
                         ("native_passes", "native_group_passes", "reference_action_passes",
                          "solver_steps", "sample_solver_steps",
                          "physical_encoder_forwards", "physical_encoder_backwards",
                          "physical_reference_actions")},
            "action_count_match": all(
                int(registered.get(k, -1)) == int(observed.get(k, -2))
                for k in ("native_passes", "native_group_passes", "reference_action_passes",
                          "solver_steps", "sample_solver_steps",
                          "physical_encoder_forwards", "physical_encoder_backwards",
                          "physical_reference_actions")),
            "cost_model": str(args.cost_model), "precheck": str(args.precheck),
        },
        "coverage_status": {
            "status": "proxy_only_unverified",
            "note": "smoke validates measured cost/action accounting only; no algorithmic claim",
        },
    }
    write_json(out / "PCFA_COST_SMOKE.json", result)
    print(json.dumps({"status": result["status"], "flops_ratio": ratio,
                      "action_count_match": result["execution_manifest"]["action_count_match"]}, ensure_ascii=False))
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
