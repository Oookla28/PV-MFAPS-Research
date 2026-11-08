#!/usr/bin/env python3
"""Balanced Softmax + PV-MFAPS plugin runner (PlantSPD, FIT-only training).

Runs the registered plugin arms with a shared backbone, checkpoint, optimizer,
encoder-update count, batch size and measured budget accounting.  HELD is never
indexed here: every artifact writes held_reads=0.  The one-time HELD evaluation
happens later through the standard freeze path.

Arms:
  balanced_softmax, bs_fixed_k_matched_B02, bs_pcfa,
  native_plus_pv_mfaps, two_forward_native, native_balanced,
  effective_number, logit_adjustment
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

from tfgps_gpu_lib import write_json
from tfgps_gpu_runner import make_policy_for_arm, run_single_arm
from pivot_runner import get_endpoint

ARMS = ("balanced_softmax", "bs_fixed_k_matched_B02", "bs_pcfa",
        "native_plus_pv_mfaps", "two_forward_native", "native_balanced",
        "effective_number", "logit_adjustment")


def pv_policy(counts, action_policy, balanced):
    p = counts.astype(float) / counts.sum()
    k = [int(v) for v in action_policy["k_discrete"]]
    a = [float(v) for v in action_policy["alpha_discrete"]]
    if len(k) != len(counts) or len(a) != len(counts):
        raise SystemExit("action policy length does not match class count")
    if any((kk == 0) and (aa > 0) for kk, aa in zip(k, a)):
        raise SystemExit("action policy violates k=0 => alpha=0")
    policy = {"tau": 0.75, "kappa": 0.0, "k": int(max(k)), "alpha_base": float(np.mean(a)),
            "q": p.tolist(), "omega": p.tolist(),
            "k_discrete": k, "alpha_discrete": a,
            "k_unique": sorted(set(k)), "k_dispersion": float(np.std(k)),
            "valid_nondegenerate": bool(len(set(k)) > 1),
            "feasible": True,
            "cost_per_sample": float(action_policy.get("certificate", {}).get("actual_cost") or 0.0),
            "fixed_k_cost_per_sample": float(action_policy.get("fixed_k_cost_per_sample") or 0.0),
            "budget_per_sample": float(action_policy.get("fixed_k_cost_per_sample") or 0.0) * 1.05,
            "mean_risk": float(action_policy.get("certificate", {}).get("macro_risk") or 0.0),
            "mean_risk_bar": 1e99,
            "registered_config": {"k_grid": action_policy["registered_config"]["k_grid"],
                                  "inner_budget": None, "tail_cvar_gamma": 0.8,
                                  "batch_size": None, "variant": "a100_fp32_fused",
                                  "source": "PV_MFAPS_ACTION_POLICY (UNVERIFIED_OFFLINE)"}}
    if balanced:
        policy.update({"balanced_softmax": True, "longtail_loss": "balanced_softmax",
                       "log_prior": np.log(np.maximum(counts.astype(float), 1e-15)).tolist()})
    return policy


def bs_fixed_policy(counts, cal, seed, batch_size):
    """Use the registered fixed-k action semantics, then add BS logits.

    The arm name is intentionally preserved in the artifact while the policy
    is constructed from the tested fixed_k_matched_B02 contract.
    """
    pol = make_policy_for_arm("fixed_k_matched_B02", counts, cal, batch_size, seed)
    pol.update({"balanced_softmax": True, "longtail_loss": "balanced_softmax",
                "log_prior": np.log(np.maximum(counts.astype(float), 1e-15)).tolist()})
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--partitions", default="0,1")
    ap.add_argument("--seeds", default="3407,3408,3409")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--action-policy-dir", default=None)
    args = ap.parse_args()
    root = Path(args.root); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    caldir = out / "calibration"
    caldir.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise SystemExit(f"unsupported plugin arms: {unknown}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    summaries = []
    for part in [int(p) for p in args.partitions.split(",") if p.strip()]:
        runner, base, head, x, labels, fit, held, counts, cal, ckpt, payload, device = \
            get_endpoint("raw", 3407, part, args.gpu, args.batch_size, caldir)
        apath = Path(args.action_policy_dir or out / "action_policy") / f"PV_MFAPS_ACTION_POLICY_p{part}_s3407.json"
        action_policy = json.loads(apath.read_text()) if apath.exists() else None
        if any(a in arms for a in ("bs_pcfa", "native_plus_pv_mfaps")) and action_policy is None:
            raise SystemExit(f"missing action policy for partition {part}: {apath}")
        for ts in seeds:
            for arm in arms:
                if arm == "bs_pcfa":
                    pol = pv_policy(counts, action_policy, balanced=True)
                elif arm == "native_plus_pv_mfaps":
                    pol = pv_policy(counts, action_policy, balanced=False)
                elif arm == "bs_fixed_k_matched_B02":
                    pol = bs_fixed_policy(counts, cal, ts, args.batch_size)
                else:
                    pol = make_policy_for_arm(arm, counts, cal, args.batch_size, ts)
                r = run_single_arm(
                    arm, base, runner, head, x, labels, fit, counts, cal, "raw",
                    part, 3407, ts, args.steps, args.batch_size, device,
                    "matched", out / "matched", len(held), int(payload.get("epoch", -1)),
                    str(ckpt), policy_override=pol, output_prefix="BS_PLUGIN",
                    pcfa_fused=True if arm in ("bs_pcfa", "native_plus_pv_mfaps") else None)
                row = {
                    "arm": arm, "partition": part, "training_seed": ts,
                    "status": r.get("status"),
                    "validation": (r.get("execution_manifest") or {}).get("status"),
                    "fit_macro_f1": (r.get("fit_metrics_native_head") or {}).get("macro_f1"),
                    "fit_tail10": (r.get("fit_metrics_native_head") or {}).get("tail_recall_worst10"),
                    "fit_fixed_macro_f1": (r.get("fit_metrics_fixed_head") or {}).get("macro_f1"),
                    "fit_fixed_tail10": (r.get("fit_metrics_fixed_head") or {}).get("tail_recall_worst10"),
                    "flops": r.get("estimated_training_flops"),
                    "wall": r.get("wall_clock_training_seconds"),
                    "cuda": r.get("cuda_training_seconds"),
                    "pcfa_fused": r.get("pcfa_fused"),
                    "file": str(out / "matched" / f"BS_PLUGIN_raw_p{part}_s3407_{arm}_train{ts}.json"),
                }
                summaries.append(row)
                write_json(out / "rows" / f"BS_PLUGIN_SUMMARY_p{part}_s{ts}_{arm}.json",
                           {"status": "EXECUTED", "kind": "BS_PLUGIN_ROW", "held_reads": 0,
                            "row": row})
                print(json.dumps(row, ensure_ascii=False), flush=True)
    ok = sum(1 for s in summaries if s["status"] == "EXECUTED")
    manifest = {
        "status": "EXECUTED" if ok == len(summaries) and summaries else "INVALID_CONFIG",
        "kind": "BS_PCFA_PLUGIN_MANIFEST",
        "scope": "Balanced Softmax + PV-MFAPS plugin FIT-only matched matrix; HELD untouched",
        "held_reads": 0,
        "registered_config": {"arms": arms, "partitions": [int(p) for p in args.partitions.split(",") if p.strip()],
                              "seeds": seeds, "steps": args.steps, "batch_size": args.batch_size,
                              "variant": "a100_fp32_fused", "budget_accounting": "measured FLOPs and wall"},
        "execution_manifest": {"n_rows": len(summaries), "n_executed": ok,
                               "n_invalid": len(summaries) - ok, "rows": summaries},
        "coverage_status": {"status": "fit_only",
                            "note": "plugin efficacy requires the post-freeze HELD evaluation; none here"},
    }
    write_json(out / "BS_PCFA_PLUGIN_MANIFEST.json", manifest)
    print(json.dumps({"status": manifest["status"], "n": len(summaries), "executed": ok}, ensure_ascii=False))
    if manifest["status"] != "EXECUTED":
        sys.exit(2)


if __name__ == "__main__":
    main()
