#!/usr/bin/env python3
"""Build FIT-only PV-MFAPS action policies for the Balanced Softmax plugin.

CPU-only.  Consumes the frozen pilot calibration (A, rho, sigma, counts) and the
measured fixed-k cost, solves the finite-action DP per partition, and writes an
explicit offline action policy.  Status is UNVERIFIED_OFFLINE: the actions are
a registration artifact and were not selected on HELD.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import pv_mfaps_dp as dp


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--partition", type=int, required=True)
    ap.add_argument("--checkpoint-seed", type=int, default=3407)
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--xi", type=float, default=1.01)
    ap.add_argument("--cost-model", default=None)
    args = ap.parse_args()
    pilot = Path(args.pilot)
    cal_path = pilot / "calibration" / f"CALIBRATION_raw_p{args.partition}_s{args.checkpoint_seed}.json"
    cal = json.loads(cal_path.read_text())
    cm_path = Path(args.cost_model) if args.cost_model else pilot / "COST_MODEL.json"
    cm = json.loads(cm_path.read_text())
    row = next(r for r in cm["execution_manifest"]["rows"]
               if int(r["partition"]) == args.partition)
    fixed_cost = float(row["fixed_k"]["per_sample_flops"])
    counts = np.asarray(cal["counts"], dtype=float)
    q = counts / counts.sum()
    A = np.asarray(cal["A"], dtype=float)
    rho = np.asarray(cal["rho"], dtype=float)
    sigma = np.asarray(cal["sigma"], dtype=float)
    n = counts
    order = np.argsort(n)
    tail = np.zeros(len(n), dtype=bool)
    tail[order[:max(1, int(round(0.15 * len(n))))]] = True
    res = dp.solve_pv_mfaps_dp(
        rho=rho, A=A, sigma=sigma, n=n, mu=args.mu, q=q, budget=args.budget,
        k_grid=(0, 1, 2, 4), alpha_grid=(0.25,), xi=args.xi,
        tail_mask=tail, return_frontier_stats=True)
    k = res.get("k") or []
    alpha = res.get("alpha") or []
    hist = {}
    for v in k:
        hist[int(v)] = hist.get(int(v), 0) + 1
    policy = {
        "status": "UNVERIFIED_OFFLINE",
        "kind": "PV_MFAPS_ACTION_POLICY",
        "scope": "FIT-only DP action policy for the Balanced Softmax plugin; no HELD indexed",
        "held_reads": 0,
        "partition": int(args.partition),
        "checkpoint_seed": int(args.checkpoint_seed),
        "calibration": str(cal_path),
        "cost_model": str(cm_path),
        "fixed_k_cost_per_sample": fixed_cost,
        "registered_config": {
            "k_grid": [0, 1, 2, 4], "alpha_grid": [0.25], "budget": args.budget,
            "mu": args.mu, "xi": args.xi, "budget_scale": 1000,
            "tail_fraction": 0.15,
            "solver": "pv_mfaps_dp.solve_pv_mfaps_dp (exact for quantised finite actions)",
        },
        "solver_status": res.get("status"),
        "certificate": res.get("certificate"),
        "k_discrete": [int(v) for v in k],
        "alpha_discrete": [float(v) for v in alpha],
        "k_hist": {str(kk): int(vv) for kk, vv in sorted(hist.items())},
        "k_unique": sorted(set(int(v) for v in k)),
        "n_classes": int(len(k)),
        "claim_boundary": "offline proxy action file; not an efficacy result and not used by the A100 selector",
    }
    out = Path(args.out)
    write_json(out / f"PV_MFAPS_ACTION_POLICY_p{args.partition}_s{args.checkpoint_seed}.json", policy)
    lines = [
        f"# PV-MFAPS action policy (partition {args.partition}, checkpoint seed {args.checkpoint_seed})",
        "",
        f"- status: **{policy['status']}**",
        f"- solver: {policy['solver_status']}",
        f"- k histogram: {policy['k_hist']}",
        f"- k unique: {policy['k_unique']}",
        f"- budget: {args.budget}, mu: {args.mu}, xi: {args.xi}",
        f"- certificate: {json.dumps(policy['certificate'], ensure_ascii=False)}",
        "",
        "Offline FIT-only registration artifact. Not an efficacy result; not used by the A100 selector.",
    ]
    (out / f"PV_MFAPS_ACTION_POLICY_p{args.partition}_s{args.checkpoint_seed}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf8")
    print(json.dumps({"status": policy["status"], "partition": args.partition,
                      "solver": res.get("status"), "k_hist": policy["k_hist"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
