#!/usr/bin/env python3
"""TF-GPS PlantSPD GPU runner.

One endpoint process loads the frozen checkpoint, calibrates FIT-only class
proxies and a fixed affine reference head, then executes registered arms.
Every arm pre-registers its action/count contract and is checked after
execution.  HELD is never indexed here: training JSONs always write
held_reads=0 and held_n_not_used only.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tfgps_gpu_lib import (
    ALPHA_GRID, BATCH_SIZE, E0, GAMMA_TAIL, INNER_LR, KAPPA_GRID, K_GRID, KM_BUDGET,
    KM_K_GRID, KM_TAIL_FRACTION, LAMBDA_COST, REFERENCE_L2, TAU_GRID, WEIGHT_CLIP,
    clone_head,
    FlopCounter, assign_grads, build_registered_config, calibrate_endpoint_with_head,
    choose_cvar_policy, choose_device, choose_kmfaps_policy, choose_mean_policy,
    class_indices_from_fit, counts_and_prior, encoder_params, eval_fixed_head_fit,
    eval_native_fit, freeze_readout, hash_file, importance_weights,
    make_batches_class, make_batches_empirical, make_audit, mix_grads,
    native_gradient, nearest_grid, omega_from_counts, pcfa_fused_gradient,
    reference_gradient,
    set_deterministic, state_to_cpu, static_policy_check, sum_grads, trust_alpha,
    validate_execution, vec, waterfill, write_json,
)

PILOT_ARMS = (
    "native_balanced", "mean_tfgps", "cvar_tfgps",
    "weighted_trust_d005", "weighted_trust_d010", "weighted_trust_d020",
    "kmfaps", "old_rfaps",
)
MATCHED_ARMS = (
    "native", "uniform_balanced", "effective_number", "logit_adjustment",
    "fixed_k_1", "fixed_k_matched_B02", "old_rfaps", "mean_tfgps",
    "cvar_tfgps", "kmfaps", "two_forward_native",
)
TRUST_DELTAS = {"weighted_trust_d005": 0.05, "weighted_trust_d010": 0.10,
                "weighted_trust_d020": 0.20}


def native_affine_init(model, runner, method):
    with torch.no_grad():
        proto = model.prototype_factors @ model.prototype_factors.transpose(-1, -2) + 1e-4 * model.identity
        if method == "log":
            proto = runner.stable_logm(proto)
        scale = model.distance_scales[1 if method == "log" else 0]
        temp = model.log_temperature.exp()
        branch_w = 2.0 * proto.reshape(2, model.n_classes, -1) / scale[:, None, None] / temp
        branch_b = -(proto.square().sum((-2, -1))) / scale[:, None] / temp
        init_w = 0.5 * branch_w.permute(1, 0, 2).reshape(model.n_classes, -1)
        init_b = 0.5 * branch_b.sum(0)
    return init_w.detach().cpu(), init_b.detach().cpu()


def fit_reference_head(base, runner, x, labels, fit, method, device, probe_limit=256):
    probe = np.asarray(fit[:int(probe_limit)], dtype=np.int64)
    z, y = E0.collect_features(runner, base, x, probe, device, 8)
    init_w, init_b = native_affine_init(base, runner, method)
    head, probe_loss = E0.fit_affine(z.detach().cpu(), y.detach().cpu(), int(base.n_classes),
                                     init_w, init_b)
    head = head.to(device)
    return head, float(probe_loss), int(len(probe))


def serialize_calibration(cal, method, checkpoint, seed, partition, fit_n, held_n):
    return {
        "status": "EXECUTED",
        "kind": "PIVOT_CALIBRATION",
        "scope": "FIT-only classwise feedback proxies; no HELD sample indexed",
        "method": method, "checkpoint_seed": int(seed), "partition": int(partition),
        "fit_n": int(fit_n), "held_n_not_used": int(held_n), "held_reads": 0,
        "counts": cal["counts"].tolist(), "prior": cal["prior"].tolist(),
        "A": cal["A"].tolist(), "sigma": cal["sigma"].tolist(),
        "sigma_raw": cal["sigma_raw"].tolist(), "rho": cal["rho"].tolist(),
        "support": cal["support"].tolist(),
        "low_support": cal["low_support"].tolist(),
        "cos1": cal["cos1"].tolist(), "err1": cal["err1"].tolist(),
        "ref_l2": cal["ref_l2"].tolist(),
        "tail_indices": cal["tail_indices"], "rho_rows": cal["rho_rows"],
        "audit": {k: v for k, v in cal["audit"].items() if k != "pass_kinds"},
        "calibration_flops": float(cal["counter"].estimated_training_flops),
        "calibration_wall_clock_seconds": float(cal.get("wall", 0.0)),
        "coverage_status": cal["coverage"],
        "checkpoint": str(checkpoint),
        "registered_config": {
            "class_probe_max_per_class": 4, "rho_probe_k": [1, 2, 4],
            "low_support_threshold": 8, "score": "classwise gradient residual proxy",
        },
        "execution_manifest": {
            "calibration_passes": cal["audit"],
            "class_probe_source": "FIT only; no HELD sample indexed",
            "rho_rows": len(cal["rho_rows"]),
        },
    }


def load_endpoint(method, seed, partition, device):
    runner, base, ckpt, payload = E0.load_checkpoint(method, seed, partition, device=str(device))
    base.eval()
    for module in base.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    x, labels, fit, held = E0.load_data(partition)
    E0.labels_global = labels
    fit = np.asarray(fit, dtype=np.int64)
    held = np.asarray(held, dtype=np.int64)
    counts = np.bincount(labels[fit], minlength=int(base.n_classes))
    if len(fit) != 4184 or len(held) != 1046 or int(base.n_classes) != 69:
        raise RuntimeError(f"BLOCKED_ENV: FIT/HELD/classes mismatch fit={len(fit)} held={len(held)} classes={base.n_classes}")
    if int(counts.min()) != 5 or int(counts.max()) != 344:
        raise RuntimeError(f"BLOCKED_ENV: count contract mismatch min={counts.min()} max={counts.max()}")
    return runner, base, ckpt, payload, x, labels, fit, held, counts


def make_policy_for_arm(arm, counts, cal, batch_size, seed):
    if arm == "native_balanced":
        q = np.full(len(counts), 1.0 / len(counts))
        return {"tau": 0.0, "kappa": 0.0, "k": 0, "alpha_base": 0.0,
                "q": q.tolist(), "omega": (np.full(len(counts), 1.0 / len(counts))).tolist()}
    if arm == "native":
        p = counts.astype(float) / counts.sum()
        return {"tau": 0.0, "kappa": 0.0, "k": 0, "alpha_base": 0.0,
                "q": p.tolist(), "omega": p.tolist()}
    if arm == "uniform_balanced":
        q = np.full(len(counts), 1.0 / len(counts))
        return {"tau": 1.0, "kappa": 0.0, "k": 0, "alpha_base": 0.0,
                "q": q.tolist(), "omega": q.tolist()}
    if arm == "effective_number":
        p = counts.astype(float) / counts.sum()
        beta = 0.9999
        en = (1.0 - beta) / (1.0 - np.power(beta, np.maximum(counts, 1).astype(float)))
        en = en / en.mean()
        return {"tau": 0.0, "kappa": 0.0, "k": 0, "alpha_base": 0.0,
                "q": p.tolist(), "omega": p.tolist(), "effective_number_weights": en.tolist(),
                "beta": beta}
    if arm == "logit_adjustment":
        p = counts.astype(float) / counts.sum()
        return {"tau": 0.0, "kappa": 0.0, "k": 0, "alpha_base": 0.0,
                "q": p.tolist(), "omega": p.tolist(),
                "logit_adjustment_scale": 1.0, "log_prior": np.log(np.maximum(p, 1e-15)).tolist()}
    if arm in ("balanced_softmax", "bs_pcfa", "native_plus_pv_mfaps"):
        p = counts.astype(float) / counts.sum()
        policy = {"tau": 0.0, "kappa": 0.0, "k": 0, "alpha_base": 0.0,
                  "q": p.tolist(), "omega": p.tolist()}
        if arm in ("balanced_softmax", "bs_pcfa"):
            policy.update({"balanced_softmax": True,
                           "longtail_loss": "balanced_softmax",
                           "log_prior": np.log(np.maximum(counts.astype(float), 1e-15)).tolist()})
        if arm in ("bs_pcfa", "native_plus_pv_mfaps"):
            policy.update({"k_discrete": [0] * len(counts),
                           "alpha_discrete": [0.0] * len(counts),
                           "valid_nondegenerate": True, "feasible": True,
                           "cost_per_sample": 0.0, "budget_per_sample": 1.0,
                           "mean_risk": 0.0, "mean_risk_bar": 1e99})
        return policy
    if arm == "fixed_k_1":
        p = counts.astype(float) / counts.sum()
        return {"tau": 0.0, "kappa": 0.0, "k": 1, "alpha_base": 1.0,
                "q": p.tolist(), "omega": p.tolist(), "k_sequence": [1] * 8}
    if arm == "fixed_k_matched_B02":
        p = counts.astype(float) / counts.sum()
        rng = np.random.default_rng(int(seed) + 991)
        k_seq = (rng.random(8) < float(KM_BUDGET)).astype(int).tolist()
        return {"tau": 0.0, "kappa": 0.0, "k": 1, "alpha_base": 1.0,
                "q": p.tolist(), "omega": p.tolist(), "k_sequence": k_seq,
                "matched_expected_budget": float(np.mean(k_seq))}
    if arm == "bs_fixed_k_matched_B02":
        p = counts.astype(float) / counts.sum()
        rng = np.random.default_rng(int(seed) + 991)
        k_seq = (rng.random(8) < float(KM_BUDGET)).astype(int).tolist()
        return {"tau": 0.0, "kappa": 0.0, "k": 1, "alpha_base": 1.0,
                "q": p.tolist(), "omega": p.tolist(), "k_sequence": k_seq,
                "matched_expected_budget": float(np.mean(k_seq)),
                "balanced_softmax": True, "longtail_loss": "balanced_softmax",
                "log_prior": np.log(np.maximum(counts.astype(float), 1e-15)).tolist()}
    if arm == "two_forward_native":
        p = counts.astype(float) / counts.sum()
        return {"tau": 0.0, "kappa": 0.0, "k": 0, "alpha_base": 0.0,
                "q": p.tolist(), "omega": p.tolist()}
    if arm == "old_rfaps":
        p = counts.astype(float) / counts.sum()
        return {"tau": 0.0, "kappa": 0.0, "k": 0, "alpha_base": 0.0,
                "q": p.tolist(), "omega": p.tolist(),
                "sigma_proxy": float(np.median(cal["sigma"])),
                "rho_proxy": float(np.median(cal["rho"])),
                "cache_protocol": "probe_every_2_cold_start_1_drift_0.5"}
    if arm in ("mean_tfgps",) + tuple(TRUST_DELTAS):
        pol, candidates = choose_mean_policy(counts, cal["A"], cal["rho"], cal["sigma"],
                                             batch_size=batch_size)
        if arm in TRUST_DELTAS:
            pol["delta"] = float(TRUST_DELTAS[arm])
        return pol
    if arm == "cvar_tfgps":
        pol, candidates = choose_cvar_policy(counts, cal["A"], cal["rho"], cal["sigma"],
                                             batch_size=batch_size)
        return pol
    if arm == "kmfaps":
        pol = choose_kmfaps_policy(counts, cal["A"], cal["rho"], cal["sigma"],
                                   batch_size=batch_size)
        rows = cal.get("rho_rows", [])
        frac = float(np.mean([bool(r.get("identifiable")) for r in rows])) if rows else 0.0
        pol["rho_identifiable_fraction"] = frac
        if frac < 1.0:
            pol["valid_nondegenerate"] = False
            pol["boundary_reason"] = "rho_not_fully_identifiable"
        return pol
    raise ValueError(f"unknown arm {arm}")


def run_single_arm(arm, base, runner, head, x, labels, fit, counts, cal, method,
                   partition, ckpt_seed, training_seed, steps, batch_size, device,
                   phase, outdir, held_n, checkpoint_epoch, checkpoint_path,
                   cvar_slsqp_check=False, policy_override=None, output_prefix=None,
                   pcfa_fused=None, batches_override=None):
    prefix = output_prefix or ("TFGPS_GPU_PILOT" if phase == "pilot" else "TFGPS_GPU_MATCHED")
    tag = f"{method}_p{partition}_s{ckpt_seed}_{arm}" + (f"_train{training_seed}" if phase == "matched" else "")
    if policy_override is not None:
        policy = copy.deepcopy(policy_override)
    else:
        policy = make_policy_for_arm(arm, counts, cal, batch_size, training_seed)
    fused = (os.environ.get("TFGPS_PCFA_FUSED", "0") == "1") if pcfa_fused is None else bool(pcfa_fused)
    if arm in ("pcfa", "bs_pcfa", "native_plus_pv_mfaps"):
        policy["pcfa_fused"] = fused
    n_classes = len(counts)
    q = np.asarray(policy.get("q", counts.astype(float) / counts.sum()), dtype=float)
    q = np.maximum(q, float(policy.get("kappa", 0.0)) / n_classes)
    q /= q.sum()
    policy["q"] = q.tolist()
    omega = np.asarray(policy.get("omega", counts.astype(float) / counts.sum()), dtype=float)
    omega = omega / omega.sum()
    policy["omega"] = omega.tolist()

    # Optional independent CVaR epigraph check on the selected action.
    if arm == "cvar_tfgps" and cvar_slsqp_check:
        k = int(policy["k"])
        alpha = float(policy["alpha_base"])
        b = (cal["A"] ** 2) * np.power(np.clip(cal["rho"], 0, 0.999999), 2 * k)
        from tfgps_gpu_lib import cvar_slsqp
        check = cvar_slsqp(alpha, b, cal["sigma"], batch_size,
                           float(policy["kappa"]), float(policy["gamma"]), q)
        policy["slsqp_check"] = check
        if check.get("available") and check.get("success"):
            policy["slsqp_gap_final"] = float(policy["risk"] - check["objective"])

    # Pre-register the batch/action sequence.
    empirical = arm in ("native", "effective_number", "logit_adjustment", "balanced_softmax",
                        "bs_fixed_k_matched_B02", "fixed_k_1",
                        "fixed_k_matched_B02", "two_forward_native", "old_rfaps")
    if empirical:
        batches = make_batches_empirical(fit, steps, batch_size, training_seed + 1301)
    else:
        batches = make_batches_class(fit, labels, q, steps, batch_size,
                                     training_seed + 1301, n_classes)
    batch_source = "registered_sampler"
    if batches_override is not None:
        batches = [np.asarray(b, dtype=np.int64) for b in batches_override]
        if len(batches) != int(steps):
            raise ValueError("batches_override length must equal steps")
        batch_source = "caller_override"
    if arm == "fixed_k_1":
        policy["k_sequence"] = [1] * steps
    if arm == "fixed_k_matched_B02":
        rng = np.random.default_rng(training_seed + 991)
        policy["k_sequence"] = (rng.random(steps) < float(KM_BUDGET)).astype(int).tolist()
    if arm == "kmfaps":
        # Pre-registered class-level actions; all class groups get a reference
        # action when alpha_base>0, with k_c inner steps (k_c may be zero).
        policy["k_sequence"] = None

    registered = build_registered_config(arm, steps, policy, batch_size, batches, n_classes, labels=labels)
    importance_audit = None
    if arm in TRUST_DELTAS:
        identity_residual = abs(float(np.sum(q * (omega / np.maximum(q, 1e-12))) - 1.0))
        importance_audit = {
            "required": True,
            "native_loss_weighted": True,
            "reference_loss_weighted": True,
            "same_weight_tensor_native_reference": True,
            "weight_formula": "omega_c/q_c applied to every sample loss",
            "q_sampling": "class-first draw from q",
            "identity_residual": identity_residual,
            "identity_equation": "sum_c q_c*(omega_c/q_c)=sum_c omega_c=1",
            "identity_status": "PASS" if identity_residual < 1e-10 else "FAIL",
            "weight_clip": float(WEIGHT_CLIP),
        }
        registered["importance_correction"] = importance_audit
        if identity_residual >= 1e-10:
            errors = [f"importance identity residual {identity_residual}"]
        else:
            errors = []
    else:
        errors = []
    errors += static_policy_check(policy, q, n_classes, arm)
    if errors:
        result = {
            "status": "INVALID_CONFIG", "kind": "TFGPS_GPU_ARM",
            "scope": f"{phase} FIT-only; static policy check failed",
            "held_reads": 0, "arm": arm, "method": method, "partition": int(partition),
            "checkpoint_seed": int(ckpt_seed), "training_seed": int(training_seed),
            "registered_config": registered, "policy": policy,
            "execution_manifest": {"preflight_status": "FAIL", "status": "FAIL", "errors": errors},
            "coverage_status": cal["coverage"]["status"], "error": "static_policy_check",
        }
        write_json(outdir / f"INVALID_{prefix}_{tag}.json", result)
        return result

    print(json.dumps({"event": "ARM_PREFLIGHT", "arm": arm, "tag": tag,
                      "registered_config": registered,
                      "static_policy_check": "PASS" if not errors else "FAIL",
                      "output_prefix": prefix}, ensure_ascii=False), flush=True)

    model = copy.deepcopy(base).to(device)
    freeze_readout(model)
    params = encoder_params(model)
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    log_prior = None
    if arm in ("logit_adjustment", "balanced_softmax", "bs_pcfa", "bs_fixed_k_matched_B02"):
        log_prior = torch.as_tensor(policy["log_prior"], dtype=torch.float32, device=device)
    cuda_warmup = bool(int(os.environ.get("TFGPS_CUDA_WARMUP", "0")))
    warmup_seconds = None
    warmup_cuda_seconds = None
    warmup_mode = None
    if cuda_warmup and torch.cuda.is_available():
        # Explicit forward+backward warm-up of the arm's actual execution
        # path. It must not update the model or optimizer state and is excluded
        # from both steady-state FLOPs and wall-clock training time.
        warmup_ii = np.asarray(batches[0], dtype=np.int64)
        warmup_yy = torch.as_tensor(labels[warmup_ii], dtype=torch.long, device=device)
        warmup_start = time.perf_counter()
        w_start_event = torch.cuda.Event(enable_timing=True)
        w_end_event = torch.cuda.Event(enable_timing=True)
        w_start_event.record()
        warmup_counter = FlopCounter()  # not attached: warm-up is outside FLOPs
        warmup_audit = make_audit()
        if arm in ("pcfa", "bs_pcfa", "native_plus_pv_mfaps") and fused:
            k_disc = np.asarray(policy["k_discrete"], dtype=int)
            a_disc = np.asarray(policy["alpha_discrete"], dtype=float)
            active = [int(c) for c in range(len(a_disc)) if float(a_disc[c]) > 0]
            if active:
                c0 = active[0]
                pcfa_fused_gradient(
                    model, runner, head, x, warmup_ii, warmup_yy, method, params,
                    device, warmup_counter, warmup_audit,
                    [{"class": c0, "mask": np.arange(len(warmup_ii), dtype=np.int64),
                      "k": int(k_disc[c0]), "alpha": float(a_disc[c0])}],
                    weights=None, denom=len(warmup_ii), log_prior=log_prior)
                warmup_mode = "pcfa_fused_correction_forward_backward"
            else:
                native_gradient(model, runner, x, warmup_ii, warmup_yy, params,
                                device, warmup_counter, warmup_audit,
                                pass_kind="warmup_native", log_prior=log_prior)
                warmup_mode = "native_forward_backward"
        elif (arm.startswith("fixed_k_") or arm == "bs_fixed_k_matched_B02") and int(policy.get("k", 0)) > 0:
            native_gradient(model, runner, x, warmup_ii, warmup_yy, params,
                            device, warmup_counter, warmup_audit,
                            pass_kind="warmup_native")
            reference_gradient(model, runner, head, x, warmup_ii, warmup_yy,
                               method, params, int(policy.get("k", 1)), device,
                               warmup_counter, warmup_audit, role="action")
            warmup_mode = "fixed_k_native_plus_reference_forward_backward"
        else:
            native_gradient(model, runner, x, warmup_ii, warmup_yy, params,
                            device, warmup_counter, warmup_audit,
                            pass_kind="warmup_native", log_prior=log_prior)
            warmup_mode = "native_forward_backward"
        model.zero_grad(set_to_none=True)
        w_end_event.record()
        torch.cuda.synchronize(device)
        warmup_cuda_seconds = float(w_start_event.elapsed_time(w_end_event) / 1000.0)
        warmup_seconds = float(time.perf_counter() - warmup_start)
    counter = FlopCounter()
    counter.attach(model)
    audit = make_audit()
    rows = []
    t0 = time.perf_counter()
    cuda_start_event = None
    cuda_end_event = None
    if torch.cuda.is_available():
        cuda_start_event = torch.cuda.Event(enable_timing=True)
        cuda_end_event = torch.cuda.Event(enable_timing=True)
        cuda_start_event.record()
    max_iw = 1.0
    clipped_frac = 0.0
    mean_iw = 1.0
    # Old R-FAPS cache state.
    cached_kappa = None
    cached_gref_norm = None
    cached_native_loss = None
    last_estimate_step = -1
    probe_gradient = None
    reference_probe_count = 0
    reference_action_count = 0

    for step, ii in enumerate(batches):
        ii = np.asarray(ii, dtype=np.int64)
        yy = torch.as_tensor(labels[ii], dtype=torch.long, device=device)
        batch_labels = labels[ii]
        row = {"step": step + 1, "batch_indices": ii.tolist(),
               "batch_classes": sorted(set(int(c) for c in batch_labels.tolist())),
               "batch_counts": np.bincount(batch_labels, minlength=n_classes).tolist()}
        weights = None
        if arm == "effective_number":
            en = np.asarray(policy["effective_number_weights"], dtype=float)
            w = en[batch_labels]
            weights = torch.as_tensor(w, dtype=torch.float32, device=device)
            max_iw = max(max_iw, float(w.max())); mean_iw = float(w.mean())
        elif arm in ("native", "native_balanced", "uniform_balanced", "logit_adjustment", "balanced_softmax", "fixed_k_1",
                     "fixed_k_matched_B02", "two_forward_native", "old_rfaps"):
            weights = None
        else:
            weights, wmax_raw, wclip_frac, wmean = importance_weights(
                batch_labels, omega, q, device, clip=WEIGHT_CLIP)
            max_iw = max(max_iw, float(wmax_raw)); clipped_frac = max(clipped_frac, float(wclip_frac))
            mean_iw = float(wmean)
        row["importance_weight_max_batch"] = float(weights.max().item()) if weights is not None else 1.0
        if arm in TRUST_DELTAS:
            assert weights is not None, "trust arm must apply importance weights to native and reference losses"

        # Latency-optimized PC-FA path. It preserves the registered gradient
        # formula while reusing the full encoder graph for class corrections.
        if arm in ("pcfa", "bs_pcfa", "native_plus_pv_mfaps") and fused:
            k_arr = np.asarray(policy["k_discrete"], dtype=int)
            a_arr = np.asarray(policy["alpha_discrete"], dtype=float)
            active = []
            for c in row["batch_classes"]:
                ac = float(a_arr[int(c)])
                if ac <= 0:
                    continue
                active.append({
                    "class": int(c),
                    "mask": np.flatnonzero(batch_labels == c),
                    "k": int(k_arr[int(c)]),
                    "alpha": ac,
                })
            g, nloss, group_meta = pcfa_fused_gradient(
                model, runner, head, x, ii, yy, method, params, device,
                counter, audit, active, weights=weights, denom=len(ii), log_prior=log_prior)
            row.update({
                "native_loss": float(nloss),
                "reference_loss": float(sum(x["reference_loss"] for x in group_meta)) if group_meta else None,
                "k": int(max((x["k_c"] for x in group_meta), default=0)),
                "alpha_used": float(np.mean([x["alpha_c"] for x in group_meta])) if group_meta else 0.0,
                "alpha_by_class": {int(c): float(a_arr[int(c)]) for c in row["batch_classes"]},
                "k_by_class": {int(c): int(k_arr[int(c)]) for c in row["batch_classes"]},
                "used_reference": bool(group_meta),
                "reference_action_groups": group_meta,
                "native_group_passes": len(group_meta),
                "pcfa_fused": True,
            })
            assign_grads(params, g); opt.step(); rows.append(row); continue

        if arm == "two_forward_native":
            gn, nloss = native_gradient(model, runner, x, ii, yy, params, device, counter, audit,
                                        pass_kind="two_forward_native_1")
            gn2, nloss2 = native_gradient(model, runner, x, ii, yy, params, device, counter, audit,
                                          pass_kind="two_forward_native_2")
            g = tuple((a + b) / 2.0 for a, b in zip(gn, gn2))
            row.update({"native_loss": 0.5 * (nloss + nloss2), "k": 0, "alpha_used": 0.0,
                        "alpha_base": 0.0, "used_reference": False})
            assign_grads(params, g); opt.step()
            rows.append(row); continue

        if arm == "old_rfaps":
            gn, nloss = native_gradient(model, runner, x, ii, yy, params, device, counter, audit,
                                        pass_kind="old_rfaps_native")
            drift = None
            if cached_native_loss is not None:
                drift = abs(float(nloss) - float(cached_native_loss)) / (abs(float(cached_native_loss)) + 1e-8)
            due = cached_kappa is None or (step - last_estimate_step) >= 2
            if drift is not None and drift > 0.5:
                due = True
            probe_used = False
            action_used = False
            probe_reused = False
            skipped = False
            ref_loss = None
            if due:
                gp, probe_loss, _ = reference_gradient(
                    model, runner, head, x, ii, yy, method, params, 1, device,
                    counter, audit, role="probe")
                reference_probe_count += 1
                probe_used = True
                vn, vp = vec(gn), vec(gp)
                cached_kappa = float(torch.linalg.vector_norm(vn - vp) /
                                     (torch.linalg.vector_norm(vp) + 1e-8))
                cached_gref_norm = float(torch.linalg.vector_norm(vp))
                cached_native_loss = float(nloss)
                probe_gradient = gp
                last_estimate_step = step
                ref_loss = probe_loss
            if step == 0 and reference_probe_count == 1:
                k, alpha = 1, 1.0
                g = probe_gradient
                probe_reused = True
            else:
                # Empirical R-FAPS action from the cached mismatch.
                gnorm = max(float(cached_gref_norm or 1.0), 1e-8)
                a_norm = max(float(cached_kappa or 1e-4), 1e-4)
                s_norm = float(policy["sigma_proxy"]) / gnorm
                rho_p = min(max(float(policy["rho_proxy"]), 0.9), 0.999999)
                best = None
                for kk in K_GRID:
                    for aa in ALPHA_GRID:
                        if kk == 0 and aa > 0:
                            continue
                        if kk > 0 and aa == 0:
                            continue
                        risk = (1 - aa) ** 2 * a_norm ** 2 * rho_p ** (2 * kk) + aa ** 2 * s_norm ** 2
                        cost = 0.05 * kk + (0.25 if aa > 0 else 0.0)
                        val = risk + cost
                        if best is None or val < best[0]:
                            best = (val, kk, aa)
                _, k, alpha = best
                if k == 0:
                    g = gn
                    skipped = True
                elif probe_gradient is not None and k == 1:
                    g = mix_grads(gn, probe_gradient, alpha)
                    probe_reused = True
                else:
                    gr, ref_loss, _ = reference_gradient(
                        model, runner, head, x, ii, yy, method, params, k, device,
                        counter, audit, role="action")
                    reference_action_count += 1
                    action_used = True
                    g = mix_grads(gn, gr, alpha)
            row.update({"native_loss": float(nloss), "reference_loss": ref_loss,
                        "k": int(k), "alpha_base": float(alpha), "alpha_used": float(alpha),
                        "used_reference": bool(k > 0 and alpha > 0),
                        "reference_probe_used": probe_used,
                        "reference_action_used": action_used,
                        "probe_reused_for_action": probe_reused,
                        "skipped_reference": skipped, "drift": drift})
            assign_grads(params, g); opt.step(); rows.append(row); continue

        # All remaining arms start from a native pass.
        gn, nloss = native_gradient(model, runner, x, ii, yy, params, device, counter, audit,
                                    weights=weights, pass_kind=f"{arm}_native",
                                    log_prior=log_prior)
        row.update({"native_loss": float(nloss), "k": 0, "alpha_used": 0.0,
                    "alpha_base": 0.0, "used_reference": False,
                    "reference_used_for_trust": False})

        if arm in ("pcfa", "bs_pcfa", "native_plus_pv_mfaps"):
            k_arr = np.asarray(policy["k_discrete"], dtype=int)
            a_arr = np.asarray(policy["alpha_discrete"], dtype=float)
            groups = []
            group_meta = []
            g = list(gn)
            for c in row["batch_classes"]:
                ac = float(a_arr[int(c)])
                if ac <= 0:
                    continue
                mask = np.flatnonzero(batch_labels == c)
                gi = ii[mask]
                gy = yy[mask]
                gw = weights[mask] if weights is not None else None
                kc = int(k_arr[int(c)])
                gnc, nl_g = native_gradient(model, runner, x, gi, gy, params, device,
                                            counter, audit, weights=gw, pass_kind="pcfa_group_native",
                                            denom=len(ii), log_prior=log_prior)
                audit["native_group_passes"] += 1
                grc, rl_g, _ = reference_gradient(
                    model, runner, head, x, gi, gy, method, params, kc, device,
                    counter, audit, weights=gw, role="action", denom=len(ii), log_prior=log_prior)
                audit["sample_solver_steps"] += int(kc) * int(len(gi))
                g = [g[j] + ac * (grc[j] - gnc[j]) for j in range(len(g))]
                groups.append(int(c))
                group_meta.append({"class": int(c), "n": int(len(gi)), "k_c": kc,
                                   "alpha_c": ac, "group_native_loss": float(nl_g),
                                   "reference_loss": float(rl_g)})
            row.update({"reference_loss": float(sum(x["reference_loss"] for x in group_meta)) if group_meta else None,
                        "k": int(max((x["k_c"] for x in group_meta), default=0)),
                        "alpha_used": float(np.mean([x["alpha_c"] for x in group_meta])) if group_meta else 0.0,
                        "alpha_by_class": {int(c): float(a_arr[int(c)]) for c in row["batch_classes"]},
                        "k_by_class": {int(c): int(k_arr[int(c)]) for c in row["batch_classes"]},
                        "used_reference": bool(group_meta),
                        "reference_action_groups": group_meta,
                        "native_group_passes": len(group_meta),
                        "pcfa_fused": False})
            assign_grads(params, tuple(g)); opt.step(); rows.append(row); continue

        if arm == "kmfaps":
            alpha_base = float(policy["alpha_base"])
            k_disc = np.asarray(policy["k_discrete"], dtype=int)
            row["alpha_base"] = alpha_base
            row["k_by_class"] = {int(c): int(k_disc[c]) for c in row["batch_classes"]}
            if alpha_base > 0:
                groups = []
                group_meta = []
                for c in row["batch_classes"]:
                    mask = np.flatnonzero(batch_labels == c)
                    gi = ii[mask]
                    gy = yy[mask]
                    gw = weights[mask] if weights is not None else None
                    kc = int(k_disc[c])
                    gr_g, rl_g, _ = reference_gradient(
                        model, runner, head, x, gi, gy, method, params, kc, device,
                        counter, audit, weights=gw, role="action", denom=len(ii), log_prior=log_prior)
                    groups.append(gr_g)
                    audit["sample_solver_steps"] += int(kc) * int(len(gi))
                    group_meta.append({"class": int(c), "n": int(len(gi)), "k_c": kc,
                                       "reference_loss": float(rl_g)})
                gr = sum_grads(groups)
                alpha_used = alpha_base
                g = mix_grads(gn, gr, alpha_used)
                row.update({"reference_loss": float(sum(x["reference_loss"] for x in group_meta)),
                            "k": int(max(k_disc[c] for c in row["batch_classes"])),
                            "alpha_used": float(alpha_used), "used_reference": True,
                            "reference_action_groups": group_meta})
            else:
                g = gn
                row.update({"reference_loss": None, "used_reference": False,
                            "reference_action_groups": []})
            assign_grads(params, g); opt.step(); rows.append(row); continue

        # Global policy arms.
        k = int(policy.get("k", 0))
        if arm in ("fixed_k_matched_B02", "bs_fixed_k_matched_B02"):
            k = int(policy["k_sequence"][step])
        alpha_base = float(policy.get("alpha_base", 0.0))
        alpha_used = alpha_base if k > 0 else 0.0
        gr = None
        ref_loss = None
        if k > 0 and alpha_base > 0:
            gr, ref_loss, _ = reference_gradient(
                model, runner, head, x, ii, yy, method, params, k, device,
                counter, audit, weights=weights, role="action", log_prior=log_prior)
            audit["sample_solver_steps"] += int(k) * int(len(ii))
            if arm in TRUST_DELTAS:
                alpha_used, cos, rel_step = trust_alpha(gn, gr, alpha_base,
                                                        delta=float(policy["delta"]))
                row["cos_native_reference"] = float(cos)
                row["relative_reference_step"] = float(rel_step)
                row["reference_used_for_trust"] = True
                row["used_reference"] = bool(alpha_used > 0)
            else:
                row["used_reference"] = True
            g = mix_grads(gn, gr, alpha_used)
        else:
            g = gn
        row.update({"k": int(k), "alpha_base": float(alpha_base),
                    "alpha_used": float(alpha_used),
                    "reference_loss": None if ref_loss is None else float(ref_loss),
                    "used_reference": bool(row.get("used_reference", False))})
        assign_grads(params, g); opt.step(); rows.append(row)

    cuda_training_seconds = None
    if torch.cuda.is_available():
        cuda_end_event.record()
        torch.cuda.synchronize(device)
        cuda_training_seconds = float(cuda_start_event.elapsed_time(cuda_end_event) / 1000.0)
    wall_train = time.perf_counter() - t0
    counter.close()
    raw_flops = float(counter.estimated_training_flops)
    physical_forwards = int(audit.get("physical_encoder_forwards", 0))
    physical_backwards = int(audit.get("physical_encoder_backwards", 0))
    extra_backwards = max(physical_backwards - physical_forwards, 0)
    avg_forward_flops = (float(counter.forward_flops) / float(max(physical_forwards, 1)))
    extra_backward_flops = 2.0 * float(extra_backwards) * avg_forward_flops
    estimated_training_flops = raw_flops + extra_backward_flops
    validation = validate_execution(registered, audit, rows, arm)
    alpha_vals = [float(r.get("alpha_used", 0.0)) for r in rows]
    alpha_summary = {
        "min": float(min(alpha_vals)) if alpha_vals else None,
        "mean": float(sum(alpha_vals) / len(alpha_vals)) if alpha_vals else None,
        "max": float(max(alpha_vals)) if alpha_vals else None,
        "nonzero_steps": int(sum(1 for x in alpha_vals if x > 1e-12)),
    }
    # FIT-only evaluation of the deployed encoder with the fixed reference head.
    t_eval = time.perf_counter()
    fit_metrics_fixed = eval_fixed_head_fit(model, runner, x, labels, fit, head, method, device, counts)
    log_prior_eval = None
    if arm == "logit_adjustment":
        log_prior_eval = torch.as_tensor(policy["log_prior"], dtype=torch.float32, device=device)
    fit_metrics_native = eval_native_fit(model, runner, x, labels, fit, device, counts,
                                         log_prior=log_prior_eval)
    wall_eval = time.perf_counter() - t_eval
    status = "EXECUTED" if validation["status"] == "PASS" else "INVALID_CONFIG"
    state_path = None
    if status == "EXECUTED":
        state_path = outdir.parent / "states" / f"{tag}.pt"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "encoder": state_to_cpu(model),
            "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
            "head_in_features": int(head.in_features),
            "head_out_features": int(head.out_features),
            "method": method, "partition": int(partition),
            "checkpoint_seed": int(ckpt_seed), "training_seed": int(training_seed),
            "arm": arm, "phase": phase,
            "policy": policy, "q": q.tolist(), "omega": omega.tolist(),
            "fit_counts": counts.tolist(),
        }, state_path)
    result = {
        "status": status,
        "kind": "TFGPS_GPU_ARM",
        "scope": f"{phase}: FIT-only selection/training; HELD not indexed",
        "held_reads": 0,
        "held_n_not_used": int(held_n),
        "arm": arm, "method": method, "phase": phase, "partition": int(partition),
        "checkpoint_seed": int(ckpt_seed), "training_seed": int(training_seed),
        "steps": int(steps), "batch_size": int(batch_size),
        "action_space": {
            "k_grid": (policy.get("registered_config") or {}).get("k_grid"),
            "k_max": int(max(policy.get("k_discrete") or [policy.get("k", 0)])),
            "k_unique": sorted({int(v) for v in (policy.get("k_discrete") or [policy.get("k", 0)])}),
            "inner_budget": (policy.get("registered_config") or {}).get("inner_budget"),
            "gamma": (policy.get("registered_config") or {}).get("tail_cvar_gamma", policy.get("gamma")),
            "batch_size": int(batch_size),
            "variant": (policy.get("registered_config") or {}).get("variant")
                       or os.environ.get("TFGPS_VARIANT", "a100_fp32_fused"),
        },
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint_epoch),
        "fit_n": int(len(fit)), "counts": counts.tolist(),
        "count_min": int(counts.min()), "count_max": int(counts.max()),
        "head_tail_ratio": float(counts.max() / max(counts.min(), 1)),
        "omega": omega.tolist(), "q": q.tolist(),
        "max_importance_weight": float(max_iw),
        "importance_weight_clip": float(WEIGHT_CLIP),
        "importance_weight_clipped_fraction": float(clipped_frac),
        "mean_importance_weight_observed": float(mean_iw),
        "policy": policy,
        "registered_config": registered,
        "execution_manifest": {
            "preflight_status": "PASS",
            "status": validation["status"], "errors": validation["errors"],
            "pcfa_implementation_variant": (
                "fused_single_encoder_graph" if arm in ("pcfa", "bs_pcfa", "native_plus_pv_mfaps") and fused
                else "legacy_per_group_encoder_passes" if arm in ("pcfa", "bs_pcfa", "native_plus_pv_mfaps")
                else None),
            "pcfa_fused": bool(arm in ("pcfa", "bs_pcfa", "native_plus_pv_mfaps") and fused),
            "batch_source": batch_source,
            "observed": {k: v for k, v in audit.items() if k != "pass_kinds"},
            "pass_kinds": audit["pass_kinds"],
            "alpha_used": alpha_summary,
            "importance_correction": importance_audit,
            "cuda_warmup": cuda_warmup,
            "warmup_seconds": warmup_seconds,
            "warmup_cuda_seconds": warmup_cuda_seconds,
            "warmup_mode": warmup_mode,
            "cuda_training_seconds": cuda_training_seconds,
            "numerics": {
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_enabled": bool(torch.backends.cudnn.enabled),
            },
        },
        "coverage_status": {
            "status": cal["coverage"]["status"],
            "low_support_classes": cal["coverage"]["low_support_classes"],
            "low_support_count": cal["coverage"]["low_support_count"],
            "sigma_source": cal["coverage"]["pooled_sigma_source"],
            "cvar_solver_verified": (bool(policy.get("solver_verified")) if arm == "cvar_tfgps" else None),
            "note": cal["coverage"]["note"],
        },
        "cvar_solver_verified": (bool(policy.get("solver_verified")) if arm == "cvar_tfgps" else None),
        "rows": rows,
        "fit_metrics_fixed_head": fit_metrics_fixed,
        "fit_metrics_native_head": fit_metrics_native,
        "class_proxy_summary": {
            "A": cal["A"].tolist(), "rho": cal["rho"].tolist(),
            "sigma": cal["sigma"].tolist(), "support": cal["support"].tolist(),
            "low_support": cal["low_support"].tolist(),
            "cos1": cal["cos1"].tolist(), "err1": cal["err1"].tolist(),
        },
        "pcfa_fused": bool(arm in ("pcfa", "bs_pcfa", "native_plus_pv_mfaps") and fused),
        "pcfa_implementation_variant": (
            "fused_single_encoder_graph" if arm in ("pcfa", "bs_pcfa", "native_plus_pv_mfaps") and fused
            else "legacy_per_group_encoder_passes" if arm in ("pcfa", "bs_pcfa", "native_plus_pv_mfaps")
            else None),
        "batch_source": batch_source,
        "wall_clock_training_seconds": float(wall_train),
        "cuda_training_seconds": cuda_training_seconds,
        "wall_clock_fit_eval_seconds": float(wall_eval),
        "warmup_seconds": warmup_seconds,
        "warmup_cuda_seconds": warmup_cuda_seconds,
        "warmup_mode": warmup_mode,
        "cuda_warmup_enabled": cuda_warmup,
        "numerics": {
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        },
        "estimated_training_flops": float(estimated_training_flops),
        "estimated_training_flops_counter": float(raw_flops),
        "physical_extra_backwards": int(extra_backwards),
        "extra_backward_flops_estimate": float(extra_backward_flops),
        "flop_accounting": "3*(forward+head) + 2*extra_backward*mean_forward_flops",
        "calibration_cost": {
            "wall_clock_seconds": float(cal.get("wall", 0.0) or 0.0),
            "flops": float(cal.get("calibration_flops", 0.0) or 0.0),
            "note": "shared endpoint-level FIT calibration cost; not added to per-arm training FLOPs",
        },
        "state_path": str(state_path) if state_path else None,
        "precursor_label": "UNWEIGHTED_PRECURSOR" if arm == "old_rfaps" else None,
    }
    outpath = outdir / f"{prefix}_{tag}.json"
    if status != "EXECUTED":
        outpath = outdir / f"INVALID_{prefix}_{tag}.json"
    write_json(outpath, result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="raw", choices=["raw", "log"])
    ap.add_argument("--checkpoint-seed", type=int, default=3407)
    ap.add_argument("--partition", type=int, default=0)
    ap.add_argument("--training-seeds", default="3407")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--phase", default="pilot", choices=["pilot", "matched"])
    ap.add_argument("--arms", default="")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--calibration-dir", required=True)
    ap.add_argument("--cvar-slsqp-check", action="store_true")
    ap.add_argument("--policy-json", default="",
                    help="Optional JSON policy override for the selected arm (FIT-only; no HELD reads).")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    caldir = Path(args.calibration_dir); caldir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.gpu)
    set_deterministic(args.checkpoint_seed)
    runner, base, ckpt, payload, x, labels, fit, held, counts = load_endpoint(
        args.method, args.checkpoint_seed, args.partition, device)
    head, probe_loss, probe_n = fit_reference_head(base, runner, x, labels, fit, args.method, device)
    cal_path = caldir / f"CALIBRATION_{args.method}_p{args.partition}_s{args.checkpoint_seed}.json"
    if cal_path.exists():
        cal_json = json.loads(cal_path.read_text())
        cal = {
            "counts": np.asarray(cal_json["counts"]), "prior": np.asarray(cal_json["prior"]),
            "A": np.asarray(cal_json["A"]), "sigma": np.asarray(cal_json["sigma"]),
            "sigma_raw": np.asarray(cal_json["sigma_raw"]), "rho": np.asarray(cal_json["rho"]),
            "support": np.asarray(cal_json["support"]),
            "low_support": np.asarray(cal_json["low_support"], dtype=bool),
            "cos1": np.asarray(cal_json["cos1"]), "err1": np.asarray(cal_json["err1"]),
            "ref_l2": np.asarray(cal_json["ref_l2"]),
            "tail_indices": cal_json["tail_indices"], "rho_rows": cal_json["rho_rows"],
            "audit": cal_json["audit"],
            "counter": type("C", (), {"estimated_training_flops": cal_json.get("calibration_flops", 0.0)})(),
            "coverage": cal_json["coverage_status"],
            "wall": cal_json.get("calibration_wall_clock_seconds", 0.0),
            "calibration_flops": cal_json.get("calibration_flops", 0.0),
        }
    else:
        _t_cal = time.perf_counter()
        cal = calibrate_endpoint_with_head(base, runner, head, x, labels, fit, args.method,
                                           device, batch_size=args.batch_size,
                                           max_per_class=4, seed=args.checkpoint_seed)
        cal["wall"] = time.perf_counter() - _t_cal
        write_json(cal_path, serialize_calibration(cal, args.method, ckpt,
                                                   args.checkpoint_seed, args.partition,
                                                   len(fit), len(held)))
    cal["held_n_not_used"] = int(len(held))
    cal["checkpoint_probe_loss"] = float(probe_loss)
    cal["checkpoint_probe_n"] = int(probe_n)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not arms:
        arms = list(PILOT_ARMS if args.phase == "pilot" else MATCHED_ARMS)
    train_seeds = [int(s.strip()) for s in str(args.training_seeds).split(",") if s.strip()]
    policy_cache = {}
    for arm in arms:
        if arm in ("fixed_k_matched_B02",):
            continue
        policy_cache[arm] = make_policy_for_arm(arm, counts, cal, args.batch_size, 3407)
    if args.policy_json:
        policy_path = Path(args.policy_json)
        if not policy_path.exists():
            raise FileNotFoundError(f"policy JSON not found: {policy_path}")
        override = json.loads(policy_path.read_text(encoding="utf8"))
        target_arm = str(override.get("arm", "bs_pcfa"))
        if target_arm not in arms:
            raise ValueError(f"policy arm {target_arm} not present in --arms={arms}")
        if str(override.get("status", "EXECUTED")) in ("INVALID_CONFIG", "BLOCKED_ENV"):
            raise ValueError(f"refusing invalid policy status: {override.get('status')}")
        policy_cache[target_arm] = override
    summary = []
    for ts in train_seeds:
        for arm in arms:
            try:
                r = run_single_arm(arm, base, runner, head, x, labels, fit, counts, cal,
                                   args.method, args.partition, args.checkpoint_seed, ts,
                                   args.steps, args.batch_size, device, args.phase, outdir,
                                   int(len(held)), int(payload.get("epoch", -1)), str(ckpt),
                                   cvar_slsqp_check=args.cvar_slsqp_check,
                                   policy_override=policy_cache.get(arm))
                prefix = "TFGPS_GPU_PILOT" if args.phase == "pilot" else "TFGPS_GPU_MATCHED"
                name = f"{prefix}_{args.method}_p{args.partition}_s{args.checkpoint_seed}_{arm}"
                if args.phase == "matched":
                    name += f"_train{ts}"
                if r["status"] != "EXECUTED":
                    name = "INVALID_" + name
                write_json(outdir / f"{name}.json", r)
                summary.append({"arm": arm, "training_seed": ts, "status": r["status"],
                                "validation": r["execution_manifest"]["status"],
                                "fit_macro_f1_proxy": r["fit_metrics_fixed_head"]["macro_f1"],
                                "fit_tail10": r["fit_metrics_fixed_head"]["tail_recall_worst10"],
                                "q_max": float(np.max(np.asarray(r["q"]))),
                                "max_iw": r["max_importance_weight"]})
                print(json.dumps(summary[-1], ensure_ascii=False))
            except Exception as exc:
                err = {"status": "INVALID_CONFIG", "kind": "TFGPS_GPU_ARM",
                       "scope": f"{args.phase}: FIT-only", "held_reads": 0,
                       "arm": arm, "training_seed": ts, "error": repr(exc),
                       "registered_config": {}, "execution_manifest": {"status": "FAIL"},
                       "coverage_status": cal["coverage"]["status"]}
                name = f"INVALID_TFGPS_GPU_{args.phase.upper()}_{args.method}_p{args.partition}_s{args.checkpoint_seed}_{arm}_train{ts}"
                write_json(outdir / f"{name}.json", err)
                print(json.dumps({"arm": arm, "status": "INVALID_CONFIG", "error": repr(exc)}))
    print(json.dumps({"status": "COMPLETE", "method": args.method,
                      "partition": args.partition, "checkpoint_seed": args.checkpoint_seed,
                      "arms": len(summary), "outdir": str(outdir)}))


if __name__ == "__main__":
    main()
