#!/usr/bin/env python3
"""FIT-only measured cost model for cost-aware PC-FA.

All costs are measured with the runner FlopCounter and actual reference
actions; no theoretical step cost is substituted.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from tfgps_gpu_lib import (
    E0, FlopCounter, choose_device, cvar_terms, encoder_params, freeze_readout,
    make_audit, native_gradient, omega_from_counts, pcfa_fused_gradient,
    reference_gradient, set_deterministic, write_json,
)
from tfgps_gpu_runner import run_single_arm
from pivot_runner import get_endpoint


def measure_delta(fn):
    before = fn.counter.estimated_training_flops
    out = fn()
    after = fn.counter.estimated_training_flops
    return out, float(after - before)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--partitions", default="0,1")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--budget-factor", type=float, default=1.05)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k-grid", default="0,1,2,4")
    ap.add_argument("--variant", default="a100_fp32_fused")
    args = ap.parse_args()
    k_grid = sorted({int(v) for v in args.k_grid.split(",") if v.strip() != ""})
    if not k_grid or 0 not in k_grid:
        raise SystemExit("k-grid must contain 0")
    fused_pcfa = bool(int(os.environ.get("TFGPS_PCFA_FUSED", "0")))
    root = Path(args.root); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    caldir = out / "calibration"
    caldir.mkdir(parents=True, exist_ok=True)
    costdir = out / "cost_calibration"
    costdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for part in [int(x) for x in args.partitions.split(",") if x.strip()]:
        method = "raw"; ckpt_seed = 3407
        runner, base, head, x, labels, fit, held, counts, cal, ckpt, payload, device = \
            get_endpoint(method, ckpt_seed, part, args.gpu, args.batch_size, caldir)
        freeze_readout(base)
        params = encoder_params(base)
        counter = FlopCounter(); counter.attach(base)
        audit = make_audit()

        def snap():
            return (float(counter.estimated_training_flops),
                    float(counter.forward_flops),
                    int(audit["physical_encoder_forwards"]),
                    int(audit["physical_encoder_backwards"]))

        def delta(s):
            raw = float(counter.estimated_training_flops) - s[0]
            fwd = int(audit["physical_encoder_forwards"]) - s[2]
            bwd = int(audit["physical_encoder_backwards"]) - s[3]
            extra = max(bwd - fwd, 0)
            avg_fwd = ((float(counter.forward_flops) - s[1]) / float(max(fwd, 1)))
            return float(raw + 2.0 * extra * avg_fwd)

        # Full-batch native cost per sample.
        ii = np.asarray(fit[:int(args.batch_size)], dtype=np.int64)
        yy = torch.as_tensor(labels[ii], dtype=torch.long, device=device)
        before = snap()
        gn, _ = native_gradient(base, runner, x, ii, yy, params, device, counter, audit,
                                pass_kind="pcfa_cost_full_native")
        full_native_cost = delta(before)
        native_per_sample = full_native_cost / float(len(ii))
        # Per-class group costs, deterministically first <=4 FIT samples.
        class_costs = {}
        for c in range(int(base.n_classes)):
            pool = fit[labels[fit] == int(c)]
            take = np.asarray(pool[:min(4, len(pool))], dtype=np.int64)
            n = int(len(take))
            if n == 0:
                continue
            yy_c = torch.as_tensor(labels[take], dtype=torch.long, device=device)
            if fused_pcfa:
                # Measure the actual fused execution: one full native encoder
                # graph already paid by the batch, plus the incremental
                # reference-head action. The k=1 minus k=0 difference isolates
                # the detached-head solver step.
                b0 = snap()
                native_gradient(base, runner, x, take, yy_c, params, device, counter,
                                audit, pass_kind="pcfa_cost_class_full_native", denom=n)
                full_native_small_cost = delta(b0)
                active = [{"class": int(c), "mask": np.arange(n, dtype=np.int64),
                           "k": 0, "alpha": 1.0}]
                b1 = snap()
                pcfa_fused_gradient(base, runner, head, x, take, yy_c, method, params,
                                    device, counter, audit, active, weights=None, denom=n)
                fused0_cost = delta(b1)
                active[0]["k"] = 1
                b2 = snap()
                pcfa_fused_gradient(base, runner, head, x, take, yy_c, method, params,
                                    device, counter, audit, active, weights=None, denom=n)
                fused1_cost = delta(b2)
                # The shared encoder forward/backward is charged once by the
                # full native term. Only reference-head work is incremental.
                reference_action = max(fused0_cost - full_native_small_cost, 0.0)
                solver_step = max(fused1_cost - fused0_cost, 0.0)
                group_native_cost = 0.0
                ref0_cost = fused0_cost
                ref1_cost = fused1_cost
                cost_formula_variant = "fused_incremental_reference_head"
            else:
                b0 = counter.estimated_training_flops
                native_gradient(base, runner, x, take, yy_c, params, device, counter,
                                audit, pass_kind="pcfa_cost_group_native", denom=n)
                group_native_cost = float(counter.estimated_training_flops - b0)
                b1 = counter.estimated_training_flops
                reference_gradient(base, runner, head, x, take, yy_c, method, params,
                                   0, device, counter, audit, role="action", denom=n)
                ref0_cost = float(counter.estimated_training_flops - b1)
                b2 = counter.estimated_training_flops
                reference_gradient(base, runner, head, x, take, yy_c, method, params,
                                   1, device, counter, audit, role="action", denom=n)
                ref1_cost = float(counter.estimated_training_flops - b2)
                reference_action = group_native_cost + ref0_cost
                solver_step = ref1_cost - ref0_cost
                cost_formula_variant = "legacy_group_native_plus_reference"
            class_costs[str(c)] = {
                "support": n,
                "group_native_cost_per_sample": group_native_cost / n,
                "reference_k0_cost_per_sample": ref0_cost / n,
                "reference_action_cost_per_sample": reference_action / n,
                "solver_step_cost_per_sample": solver_step / n,
                "native_cost_per_sample": native_per_sample,
                "cost_formula_variant": cost_formula_variant,
            }
        counter.close()
        # Actual fixed-k matched-B02 arm cost on this partition.
        p = counts.astype(float) / counts.sum()
        pol = {"tau": 0.0, "kappa": 0.0, "k": 1, "alpha_base": 1.0,
               "q": p.tolist(), "omega": p.tolist(), "k_sequence": None,
               "matched_expected_budget": 0.2}
        fixed_res = run_single_arm(
            "fixed_k_matched_B02", base, runner, head, x, labels, fit, counts, cal,
            method, part, ckpt_seed, 3407, args.steps, args.batch_size, device,
            "pilot", costdir, len(held), int(payload.get("epoch", -1)), str(ckpt),
            policy_override=pol, output_prefix="PCFA_COST_CAL")
        total_steps_batch = float(args.steps * args.batch_size)
        fixed_flops = float(fixed_res["estimated_training_flops"])
        fixed_wall = float(fixed_res["wall_clock_training_seconds"])
        fixed_per_sample_flops = fixed_flops / total_steps_batch
        fixed_per_sample_wall = fixed_wall / total_steps_batch
        q = omega_from_counts(counts, 0.75)
        m = float(args.batch_size)
        r_fixed = (np.asarray(cal["sigma"], dtype=float) ** 2) / (m * np.maximum(q, 1e-15))
        mean_risk_fixed = float(np.sum(q * r_fixed))
        cvar_fixed, _, _ = cvar_terms(r_fixed, 0.8)
        rows.append({
            "partition": part, "checkpoint_seed": ckpt_seed, "method": method,
            "checkpoint": str(ckpt), "calibration": str(caldir / f"CALIBRATION_{method}_p{part}_s{ckpt_seed}.json"),
            "q": q.tolist(), "omega": q.tolist(), "q_sum": float(q.sum()), "m": int(args.batch_size),
            "native_cost_per_sample_flops": native_per_sample,
            "class_costs": class_costs,
            "fixed_k": {
                "total_flops": fixed_flops, "total_wall_seconds": fixed_wall,
                "cuda_training_seconds": fixed_res.get("cuda_training_seconds"),
                "warmup_seconds": fixed_res.get("warmup_seconds"),
                "warmup_cuda_seconds": fixed_res.get("warmup_cuda_seconds"),
                "cuda_warmup_enabled": fixed_res.get("cuda_warmup_enabled"),
                "pcfa_fused": fixed_res.get("pcfa_fused"),
                "per_sample_flops": fixed_per_sample_flops,
                "per_sample_wall_seconds": fixed_per_sample_wall,
                "steps": int(args.steps), "batch_size": int(args.batch_size),
                "action_counts": fixed_res["execution_manifest"]["observed"],
                "state_path": fixed_res.get("state_path"),
                "result_status": fixed_res.get("status"),
            },
            "budget": {
                "factor": float(args.budget_factor),
                "per_sample_flops": float(args.budget_factor * fixed_per_sample_flops),
                "per_sample_wall_seconds": float(args.budget_factor * fixed_per_sample_wall),
            },
            "mean_risk_fixed_k": mean_risk_fixed,
            "cvar_fixed_k": float(cvar_fixed),
            "measured_action_counts": fixed_res["execution_manifest"]["observed"],
        })
    result = {
        "status": "EXECUTED",
        "scope": "FIT-only measured native/reference/solver/fixed-k cost calibration; no HELD sample indexed",
        "held_reads": 0,
        "registered_config": {
            "cost_formula": "c_c(k,alpha)=native_cost_c + k*solver_cost_c + 1[alpha>0]*reference_cost_c",
            "implementation_variant": "fused_single_encoder_graph" if fused_pcfa else "legacy_per_group_encoder_passes",
            "reference_cost_c": (
                "measured fused reference-head incremental cost per sample"
                if fused_pcfa else
                "measured group native + reference-action incremental cost per sample"),
            "solver_cost_c": "measured k=1 minus k=0 reference-head cost per sample",
            "cost_source": "runner FlopCounter estimated_training_flops and actual reference-action counts",
            "budget": "1.05 * measured fixed-k matched-B02 per-sample FLOPs",
            "time_protocol": "same TFGPS_CUDA_WARMUP=1 forward+backward warm-up for fixed-k and fused PC-FA; CUDA events separate from end-to-end wall",
            "q": "omega(counts,tau=.75)", "m": int(args.batch_size),
            "mean_risk_constraint": "sum_c q_c r_c <= fixed-k mean risk",
            "tail_objective": "uniform-class CVaR gamma=.8",
            "k_grid": list(k_grid), "alpha_grid": [0.25, 0.5, 0.75, 1.0],
            "variant": args.variant,
            "solver_step_extrapolation": "measured k=1 minus k=0 solver-step cost, linear in k",
        },
        "execution_manifest": {"rows": rows, "gpu": int(args.gpu),
                               "device": str(choose_device(args.gpu))},
        "coverage_status": {
            "status": "proxy_only_unverified",
            "note": "empirical FIT cost model and risk proxies; actual matched arm costs are checked separately",
        },
    }
    write_json(out / "COST_MODEL.json", result)
    print(json.dumps({"status": "EXECUTED", "partitions": len(rows),
                      "fixed_k_flops": [r["fixed_k"]["total_flops"] for r in rows],
                      "budgets": [r["budget"]["per_sample_flops"] for r in rows]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
