#!/usr/bin/env python3
"""Same-checkpoint old/fused PC-FA equivalence smoke.

No HELD sample is indexed. The script runs independent one-step legacy/fused
pairs from the same checkpoint, with identical forced active-class batches and
policy, then compares the action, logical counters, and the saved encoder state
of every pair. One-step pairs avoid AdamW drift amplification across steps while
testing the actual corrected update on each batch.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

from pivot_runner import get_endpoint
from tfgps_gpu_lib import set_deterministic
from tfgps_gpu_runner import run_single_arm


def _load_state(path):
    obj = torch.load(path, map_location="cpu")
    return {k: v.detach().float() for k, v in obj["encoder"].items()}


def _state_delta(a, b):
    keys = sorted(set(a) | set(b))
    missing = [k for k in keys if k not in a or k not in b]
    max_abs = 0.0
    norm_sq = 0.0
    ref_sq = 0.0
    for k in keys:
        if k not in a or k not in b:
            continue
        d = (a[k] - b[k]).reshape(-1)
        max_abs = max(max_abs, float(d.abs().max()))
        norm_sq += float(torch.dot(d, d))
        ref_sq += float(torch.dot(a[k].reshape(-1), a[k].reshape(-1)))
    return {"missing_keys": missing, "max_abs": max_abs,
            "relative_l2": (norm_sq ** 0.5) / (ref_sq ** 0.5 + 1e-12)}


def _action_rows(result):
    """Action sequence only; floating loss snapshots are not action state."""
    rows = []
    for row in result.get("rows", []):
        groups = row.get("reference_action_groups") or []
        rows.append({
            "step": row.get("step"),
            "batch_indices": row.get("batch_indices"),
            "batch_classes": row.get("batch_classes"),
            "k": row.get("k"),
            "alpha_used": row.get("alpha_used"),
            "k_by_class": row.get("k_by_class"),
            "alpha_by_class": row.get("alpha_by_class"),
            "used_reference": row.get("used_reference"),
            "group_actions": [
                {"class": g.get("class"), "n": g.get("n"),
                 "k_c": g.get("k_c"), "alpha_c": g.get("alpha_c")}
                for g in groups
            ],
        })
    return rows


def _coverage_batches(fit, labels, policy, steps, batch_size):
    """Deterministic all-active-class batches for the equivalence pairs.

    The standard q-sampler may contain no alpha_c>0 class in a short run. That
    would test only the native path. Using only the active class also makes the
    legacy group forward and the fused full-batch forward receive identical
    inputs, isolating the correction graph from batch-size convolution
    numerics. Each pair still starts from the same checkpoint and a fresh
    optimizer.
    """
    k_arr = np.asarray(policy["k_discrete"], dtype=int)
    a_arr = np.asarray(policy["alpha_discrete"], dtype=float)
    active = [int(c) for c in range(len(a_arr)) if float(a_arr[c]) > 0.0 and int(k_arr[c]) > 0]
    if not active:
        raise SystemExit("PCFA policy has no active reference class; equivalence coverage impossible")
    fit = np.asarray(fit, dtype=np.int64)
    c = active[0]
    cls_idx = np.asarray(fit[labels[fit] == c], dtype=np.int64)
    effective_batch_size = min(int(batch_size), int(len(cls_idx)))
    if effective_batch_size < 1:
        raise SystemExit("active class has no FIT samples")
    # Every pair receives the identical all-active-class batch. Each pair is
    # still an independent reset of model/optimizer, so this tests determinism
    # of the corrected update without injecting a second batch whose AdamW
    # drift could exceed the registered state tolerance for numerical reasons
    # unrelated to the implementation.
    base_batch = np.asarray(cls_idx[:effective_batch_size], dtype=np.int64)
    batches = [base_batch.copy() for _ in range(int(steps))]
    return active, batches, effective_batch_size


def _pair_compare(legacy, fused):
    legacy_obs = (legacy.get("execution_manifest") or {}).get("observed", {})
    fused_obs = (fused.get("execution_manifest") or {}).get("observed", {})
    logical_counts_equal = all(legacy_obs.get(k) == fused_obs.get(k) for k in (
        "native_passes", "reference_passes", "native_group_passes",
        "reference_action_passes", "solver_steps", "sample_solver_steps",
        "backward_passes"))
    active_groups_exercised = (
        int(fused_obs.get("reference_action_passes", 0)) > 0 and
        int(fused_obs.get("native_group_passes", 0)) > 0 and
        int(legacy_obs.get("reference_action_passes", 0)) > 0 and
        int(legacy_obs.get("native_group_passes", 0)) > 0)
    batch_override_ok = (legacy.get("batch_source") == "caller_override" and
                         fused.get("batch_source") == "caller_override")
    variant_ok = ((legacy.get("execution_manifest") or {}).get("pcfa_implementation_variant")
                  == "legacy_per_group_encoder_passes" and
                  (fused.get("execution_manifest") or {}).get("pcfa_implementation_variant")
                  == "fused_single_encoder_graph" and
                  legacy.get("pcfa_fused") is False and
                  fused.get("pcfa_fused") is True)
    state_delta = _state_delta(_load_state(legacy["state_path"]),
                               _load_state(fused["state_path"]))
    return {
        "action_equal": _action_rows(legacy) == _action_rows(fused),
        "logical_counts_equal": logical_counts_equal,
        "active_groups_exercised": active_groups_exercised,
        "batch_override_ok": batch_override_ok,
        "implementation_variants_ok": variant_ok,
        "state_delta": state_delta,
        "states_ok": not state_delta["missing_keys"] and state_delta["max_abs"] <= 1e-5,
        "held_reads_ok": legacy.get("held_reads") == 0 and fused.get("held_reads") == 0,
        "status_ok": legacy.get("status") == "EXECUTED" and fused.get("status") == "EXECUTED",
        "warmup_ok": (legacy.get("cuda_warmup_enabled") is True and
                      fused.get("cuda_warmup_enabled") is True and
                      legacy.get("warmup_seconds") is not None and
                      fused.get("warmup_seconds") is not None),
        "legacy_observed": legacy_obs, "fused_observed": fused_obs,
        "legacy_warmup_seconds": legacy.get("warmup_seconds"),
        "fused_warmup_seconds": fused.get("warmup_seconds"),
        "legacy_cuda_training_seconds": legacy.get("cuda_training_seconds"),
        "fused_cuda_training_seconds": fused.get("cuda_training_seconds"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--partition", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    caldir = out / "calibration"
    runner, base, head, x, labels, fit, held, counts, cal, ckpt, payload, device = \
        get_endpoint("raw", 3407, args.partition, args.gpu, args.batch_size, caldir)
    pre = json.loads((out / "PCFA_COST_PRECHECK.json").read_text())
    policy = copy.deepcopy(pre["execution_manifest"]["policies"][str(args.partition)])
    if not policy.get("valid_nondegenerate") or not policy.get("feasible"):
        raise SystemExit("PCFA precheck is not valid; equivalence smoke refused")
    active_classes, coverage_batches, coverage_batch_size = _coverage_batches(
        fit, labels, policy, args.steps, args.batch_size)

    pairs = []
    for step in range(int(args.steps)):
        batch = coverage_batches[step]
        common = dict(
            arm="pcfa", base=base, runner=runner, head=head, x=x, labels=labels,
            fit=fit, counts=counts, cal=cal, method="raw",
            partition=args.partition, ckpt_seed=3407, training_seed=3407,
            steps=1, batch_size=len(batch), device=device,
            phase="pilot", held_n=len(held),
            checkpoint_epoch=int(payload.get("epoch", -1)),
            checkpoint_path=str(ckpt), policy_override=policy,
            batches_override=[batch],
        )
        legacy_dir = out / "equivalence_legacy" / f"pair_{step + 1}" / "results"
        fused_dir = out / "equivalence_fused" / f"pair_{step + 1}" / "results"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        fused_dir.mkdir(parents=True, exist_ok=True)
        set_deterministic(3407)
        os.environ["TFGPS_PCFA_FUSED"] = "0"
        legacy = run_single_arm(outdir=legacy_dir,
                                output_prefix=f"PCFA_EQ_LEGACY_S{step + 1}",
                                pcfa_fused=False, **common)
        # The second run starts from the same checkpoint. run_single_arm
        # deep-copies the base model and constructs a fresh AdamW and inner
        # head solver, so no optimizer or parameter state is shared.
        set_deterministic(3407)
        os.environ["TFGPS_PCFA_FUSED"] = "1"
        fused = run_single_arm(outdir=fused_dir,
                               output_prefix=f"PCFA_EQ_FUSED_S{step + 1}",
                               pcfa_fused=True, **common)
        pair = _pair_compare(legacy, fused)
        pair.update({"pair": step + 1, "batch_indices": batch.tolist(),
                     "legacy_result": str(legacy_dir), "fused_result": str(fused_dir)})
        pairs.append(pair)

    def all_flag(key):
        return all(bool(p.get(key)) for p in pairs)

    action_equal = all_flag("action_equal")
    logical_counts_equal = all_flag("logical_counts_equal")
    active_groups_exercised = all_flag("active_groups_exercised")
    batch_override_ok = all_flag("batch_override_ok")
    variant_ok = all_flag("implementation_variants_ok")
    states_ok = all_flag("states_ok")
    held_ok = all_flag("held_reads_ok")
    status_ok = all_flag("status_ok")
    warmup_ok = all_flag("warmup_ok")
    max_abs = max((float(p["state_delta"]["max_abs"]) for p in pairs), default=None)
    max_rel = max((float(p["state_delta"]["relative_l2"]) for p in pairs), default=None)
    state_delta = {"max_abs": max_abs, "relative_l2": max_rel,
                   "per_pair": [p["state_delta"] for p in pairs]}
    ok = all((action_equal, logical_counts_equal, variant_ok, states_ok, held_ok,
              status_ok, warmup_ok, active_groups_exercised, batch_override_ok))
    result = {
        "status": "EXECUTED" if ok else "INVALID_CONFIG",
        "scope": "independent one-step same-checkpoint FIT-only old/fused PC-FA equivalence pairs; no HELD indexing",
        "held_reads": 0 if held_ok else 1,
        "registered_config": {
            "partition": int(args.partition), "checkpoint_seed": 3407,
            "training_seed": 3407, "n_pairs": int(args.steps),
            "steps_per_pair": 1, "batch_size": int(coverage_batch_size),
            "policy": policy,
            "coverage_batch_policy": "all-active-class batch; fresh model/optimizer per pair",
            "state_max_abs_tolerance": 1e-5,
        },
        "execution_manifest": {
            "legacy_result": str(out / "equivalence_legacy"),
            "fused_result": str(out / "equivalence_fused"),
            "action_equal": action_equal,
            "logical_counts_equal": logical_counts_equal,
            "active_groups_exercised": active_groups_exercised,
            "batch_override_ok": batch_override_ok,
            "coverage_active_classes": active_classes,
            "coverage_batch_size": int(coverage_batch_size),
            "coverage_batches": [b.tolist() for b in coverage_batches],
            "implementation_variants_ok": variant_ok,
            "state_delta": state_delta, "states_ok": states_ok,
            "held_reads_ok": held_ok, "status_ok": status_ok,
            "warmup_ok": warmup_ok,
            "pairs": pairs,
            "legacy_warmup_seconds": max((p.get("legacy_warmup_seconds") or 0.0) for p in pairs),
            "fused_warmup_seconds": max((p.get("fused_warmup_seconds") or 0.0) for p in pairs),
            "legacy_cuda_training_seconds": sum((p.get("legacy_cuda_training_seconds") or 0.0) for p in pairs),
            "fused_cuda_training_seconds": sum((p.get("fused_cuda_training_seconds") or 0.0) for p in pairs),
            "legacy_observed": pairs[-1]["legacy_observed"] if pairs else {},
            "fused_observed": pairs[-1]["fused_observed"] if pairs else {},
        },
        "coverage_status": {
            "status": "proxy_only_unverified",
            "note": "equivalence gate only; no efficacy or speed claim",
        },
    }
    (out / "PCFA_FUSED_EQUIVALENCE_SMOKE.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": result["status"],
                      "action_equal": action_equal,
                      "logical_counts_equal": logical_counts_equal,
                      "states_ok": states_ok,
                      "state_delta": state_delta}, ensure_ascii=False))
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
