#!/usr/bin/env python3
"""Pivot run: importance-corrected trust TF-GPS + corrected kappa-based K-MFAPS.

Writes only under the new pivot output directory.  FIT is used for all
selection/calibration; HELD is never indexed here.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from tfgps_gpu_lib import (
    ALPHA_GRID, KM_BUDGET, KM_K_GRID, KM_TAIL_FRACTION, TAU_GRID,
    choose_device, choose_kmfaps_policy_kappa, choose_mean_policy,
    counts_and_prior, set_deterministic, write_json,
)
from tfgps_gpu_runner import (
    fit_reference_head, load_endpoint, run_single_arm, serialize_calibration,
)
from tfgps_gpu_lib import calibrate_endpoint_with_head, E0

CONTROLS = ("native_balanced", "effective_number", "logit_adjustment",
            "fixed_k_matched_B02")
TRUSTS = ("weighted_trust_d005", "weighted_trust_d010", "weighted_trust_d020")
TRUST_DELTAS = {"weighted_trust_d005": 0.05, "weighted_trust_d010": 0.10,
                "weighted_trust_d020": 0.20}


def load_calibration(path: Path):
    d = json.loads(path.read_text())
    cal = {
        "counts": np.asarray(d["counts"]), "prior": np.asarray(d["prior"]),
        "A": np.asarray(d["A"]), "sigma": np.asarray(d["sigma"]),
        "sigma_raw": np.asarray(d["sigma_raw"]), "rho": np.asarray(d["rho"]),
        "support": np.asarray(d["support"]),
        "low_support": np.asarray(d["low_support"], dtype=bool),
        "cos1": np.asarray(d["cos1"]), "err1": np.asarray(d["err1"]),
        "ref_l2": np.asarray(d["ref_l2"]),
        "tail_indices": d["tail_indices"], "rho_rows": d["rho_rows"],
        "audit": d["audit"],
        "counter": type("C", (), {"estimated_training_flops": d.get("calibration_flops", 0.0)})(),
        "coverage": d["coverage_status"],
        "wall": d.get("calibration_wall_clock_seconds", 0.0),
        "calibration_flops": d.get("calibration_flops", 0.0),
    }
    return cal


def get_endpoint(method, ckpt_seed, partition, gpu, batch_size, caldir):
    device = choose_device(gpu)
    set_deterministic(ckpt_seed)
    runner, base, ckpt, payload, x, labels, fit, held, counts = load_endpoint(
        method, ckpt_seed, partition, device)
    head, probe_loss, probe_n = fit_reference_head(base, runner, x, labels, fit, method, device)
    cal_path = Path(caldir) / f"CALIBRATION_{method}_p{partition}_s{ckpt_seed}.json"
    if cal_path.exists():
        cal = load_calibration(cal_path)
    else:
        t = time.perf_counter()
        cal = calibrate_endpoint_with_head(base, runner, head, x, labels, fit, method,
                                           device, batch_size=batch_size,
                                           max_per_class=4, seed=ckpt_seed)
        cal["wall"] = time.perf_counter() - t
        write_json(cal_path, serialize_calibration(cal, method, ckpt, ckpt_seed,
                                                   partition, len(fit), len(held)))
    cal["held_n_not_used"] = int(len(held))
    cal["checkpoint_probe_loss"] = float(probe_loss)
    cal["checkpoint_probe_n"] = int(probe_n)
    return runner, base, head, x, labels, fit, held, counts, cal, ckpt, payload, device


def policy_for(arm, counts, cal, batch_size, training_seed):
    c = len(counts)
    p = counts.astype(float) / counts.sum()
    if arm == "native_balanced":
        q = np.full(c, 1.0 / c)
        return {"tau": 0.0, "kappa": 0.0, "k": 0, "alpha_base": 0.0,
                "q": q.tolist(), "omega": q.tolist()}
    if arm == "effective_number":
        beta = 0.9999
        en = (1.0 - beta) / (1.0 - np.power(beta, np.maximum(counts, 1).astype(float)))
        en = en / en.mean()
        return {"tau": 0.0, "kappa": 0.0, "k": 0, "alpha_base": 0.0,
                "q": p.tolist(), "omega": p.tolist(),
                "effective_number_weights": en.tolist(), "beta": beta}
    if arm == "logit_adjustment":
        return {"tau": 0.0, "kappa": 0.0, "k": 0, "alpha_base": 0.0,
                "q": p.tolist(), "omega": p.tolist(),
                "logit_adjustment_scale": 1.0,
                "log_prior": np.log(np.maximum(p, 1e-15)).tolist()}
    if arm == "fixed_k_matched_B02":
        return {"tau": 0.0, "kappa": 0.0, "k": 1, "alpha_base": 1.0,
                "q": p.tolist(), "omega": p.tolist(),
                "k_sequence": None, "matched_expected_budget": float(KM_BUDGET)}
    if arm in TRUSTS:
        pol, _ = choose_mean_policy(counts, cal["A"], cal["rho"], cal["sigma"],
                                    batch_size=batch_size)
        pol["delta"] = float(TRUST_DELTAS[arm])
        return pol
    if arm == "kmfaps":
        return choose_kmfaps_policy_kappa(
            counts, cal["A"], cal["sigma"], cal["rho"], cal["rho_rows"],
            batch_size=batch_size, tau=0.75, budget=KM_BUDGET,
            k_grid=KM_K_GRID, beta=0.0, gamma_tail=KM_TAIL_FRACTION)
    raise ValueError(f"unknown pivot arm {arm}")


def prefix_for(arm, mode):
    if arm == "kmfaps":
        base = "PIVOT_KMFAPS"
    elif arm in TRUSTS:
        base = "PIVOT_TRUST"
    else:
        base = "PIVOT_CONTROL"
    return base + ("_PILOT" if mode == "smoke" else "_MATCHED")


def run_arms(mode, arms, method, ckpt_seed, partition, training_seeds, steps,
             batch_size, gpu, outdir, caldir):
    runner, base, head, x, labels, fit, held, counts, cal, ckpt, payload, device = \
        get_endpoint(method, ckpt_seed, partition, gpu, batch_size, caldir)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    phase = "pilot" if mode == "smoke" else "matched"
    policy_cache = {}
    for arm in arms:
        if arm == "fixed_k_matched_B02":
            continue
        policy_cache[arm] = policy_for(arm, counts, cal, batch_size, 3407)
    summaries = []
    for ts in training_seeds:
        for arm in arms:
            if arm == "kmfaps":
                pol = policy_for(arm, counts, cal, batch_size, ts)
                if not pol.get("valid_nondegenerate"):
                    b = {
                        "status": "UNVERIFIED", "boundary": "BOUNDARY_KMFAPS",
                        "scope": "K-MFAPS precheck failed; no training executed",
                        "held_reads": 0, "method": method, "partition": int(partition),
                        "checkpoint_seed": int(ckpt_seed),
                        "registered_config": {
                            "kappa_definition": "A/(2*sigma-A)", "b_definition": "(kappa+beta*sigma)^2",
                            "beta": 0.0, "budget": KM_BUDGET, "k_grid": list(KM_K_GRID),
                            "rho_probe": "FIT-only k=1,2,4",
                        },
                        "execution_manifest": {
                            "policy": pol, "reason": pol.get("boundary_reason"),
                            "rho_identifiable_fraction": pol.get("rho_identifiable_fraction"),
                            "k_unique": pol.get("k_discrete_unique"),
                            "k_dispersion": pol.get("k_dispersion"),
                        },
                        "coverage_status": {
                            "status": "proxy_only_unverified",
                            "note": "K-MFAPS boundary; not a main result",
                        },
                    }
                    write_json(Path(outdir) / "BOUNDARY_KMFAPS.json", b)
                    print(json.dumps({"status": "BOUNDARY_KMFAPS", "policy": pol}, ensure_ascii=False))
                    return summaries
            elif arm == "fixed_k_matched_B02":
                pol = policy_for(arm, counts, cal, batch_size, ts)
            else:
                pol = copy.deepcopy(policy_cache[arm])
            prefix = prefix_for(arm, mode)
            r = run_single_arm(
                arm, base, runner, head, x, labels, fit, counts, cal, method,
                partition, ckpt_seed, ts, steps, batch_size, device, phase,
                outdir, len(held), int(payload.get("epoch", -1)), str(ckpt),
                cvar_slsqp_check=False, policy_override=pol, output_prefix=prefix)
            fname = f"{prefix}_{method}_p{partition}_s{ckpt_seed}_{arm}" + (f"_train{ts}" if phase == "matched" else "")
            summaries.append({
                "arm": arm, "training_seed": int(ts), "status": r.get("status"),
                "validation": (r.get("execution_manifest") or {}).get("status"),
                "fit_macro_f1_native": (r.get("fit_metrics_native_head") or {}).get("macro_f1"),
                "fit_tail10_native": (r.get("fit_metrics_native_head") or {}).get("tail_recall_worst10"),
                "held_reads": r.get("held_reads"),
                "max_importance_weight": r.get("max_importance_weight"),
                "wall": r.get("wall_clock_training_seconds"),
                "flops": r.get("estimated_training_flops"),
                "output": str(Path(outdir) / f"{fname}.json"),
            })
            print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["smoke", "matched", "kmfaps"])
    ap.add_argument("--method", default="raw")
    ap.add_argument("--checkpoint-seed", type=int, default=3407)
    ap.add_argument("--partition", type=int, default=0)
    ap.add_argument("--training-seeds", default="3407")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--calibration-dir", required=True)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    if args.mode == "smoke":
        arms = list(CONTROLS) + list(TRUSTS)
        seeds = [3407]
    elif args.mode == "matched":
        arms = list(CONTROLS) + list(TRUSTS)
        seeds = [int(x) for x in args.training_seeds.split(",") if x.strip()]
    else:
        arms = ["kmfaps"]
        seeds = [int(x) for x in args.training_seeds.split(",") if x.strip()]
    summaries = run_arms(args.mode, arms, args.method, args.checkpoint_seed,
                         args.partition, seeds, args.steps, args.batch_size,
                         args.gpu, outdir, args.calibration_dir)
    print(json.dumps({"status": "COMPLETE", "mode": args.mode,
                      "partition": args.partition, "n": len(summaries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
