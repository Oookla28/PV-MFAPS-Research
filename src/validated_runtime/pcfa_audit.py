#!/usr/bin/env python3
"""Deterministic cost-aware PC-FA audit and summary (stdlib only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception as exc:
        return {"_parse_error": repr(exc), "_path": str(path)}


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def mean_std(vals):
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)), len(xs)


def collect(out):
    arms = []
    for phase in ("pilot", "matched"):
        d = out / phase
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            obj = load_json(p); obj["_file"] = str(p); obj["_phase"] = phase
            arms.append(obj)
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--audit-md", required=True)
    ap.add_argument("--audit-json", required=True)
    ap.add_argument("--skip-state-existence", action="store_true")
    args = ap.parse_args()
    root = Path(args.root); out = Path(args.out)
    cm = load_json(out / "COST_MODEL.json")
    smoke = load_json(out / "PCFA_COST_SMOKE.json")
    pre = load_json(out / "PCFA_COST_PRECHECK.json")
    eq = load_json(out / "PCFA_FUSED_EQUIVALENCE_SMOKE.json")
    freeze = load_json(out / "FIT_FREEZE_MANIFEST.json")
    held = load_json(out / "HELD_ONE_TIME_EVAL.json")
    env_path = out / "PCFA_GPU_ENV_MANIFEST.json"
    if not env_path.exists():
        env_path = out / "PIVOT_GPU_ENV_MANIFEST.json"
    if not env_path.exists():
        env_path = root / "out/pivot_2026-11-05/PIVOT_GPU_ENV_MANIFEST.json"
    env = load_json(env_path)
    arms = collect(out)
    errors = []; checks = []
    for name, obj, want in (("COST_MODEL", cm, "EXECUTED"), ("PCFA_COST_SMOKE", smoke, "EXECUTED"),
                            ("PCFA_COST_PRECHECK", pre, "EXECUTED"),
                            ("PCFA_FUSED_EQUIVALENCE_SMOKE", eq, "EXECUTED"),
                            ("FIT_FREEZE", freeze, "EXECUTED"), ("HELD", held, "EXECUTED")):
        if obj.get("status") == want:
            checks.append(f"{name} {want}")
        else:
            errors.append(f"{name} status {obj.get('status')}")
    if env.get("status") != "EXECUTED":
        errors.append(f"environment manifest status {env.get('status')}")
    else:
        checks.append(f"environment manifest {env_path}")
        vis = (env.get("execution_manifest") or {}).get("visible_devices")
        if vis != [0, 1]:
            errors.append(f"visible_devices {vis} != [0,1]")
        else:
            checks.append("environment visible_devices=[0,1]")
    eqm = eq.get("execution_manifest") or {}
    eq_ok = (
        eq.get("status") == "EXECUTED" and eq.get("held_reads") == 0 and
        eqm.get("action_equal") is True and
        eqm.get("logical_counts_equal") is True and
        eqm.get("active_groups_exercised") is True and
        eqm.get("batch_override_ok") is True and
        eqm.get("implementation_variants_ok") is True and
        eqm.get("states_ok") is True and
        eqm.get("held_reads_ok") is True and
        eqm.get("status_ok") is True and
        eqm.get("warmup_ok") is True
    )
    if eq_ok:
        checks.append("real-checkpoint old/fused equivalence gate PASS")
    else:
        errors.append("PCFA_FUSED_EQUIVALENCE_SMOKE gate failed")
    cm_reg = cm.get("registered_config") or {}
    cm_rows = (cm.get("execution_manifest") or {}).get("rows") or []
    cm_variant_ok = cm_reg.get("implementation_variant") == "fused_single_encoder_graph"
    cm_class_variant_ok = all(
        all(v.get("cost_formula_variant") == "fused_incremental_reference_head"
            for v in (r.get("class_costs") or {}).values())
        for r in cm_rows)
    cm_timing_ok = all(
        (r.get("fixed_k") or {}).get("cuda_warmup_enabled") is True and
        (r.get("fixed_k") or {}).get("warmup_seconds") is not None and
        (r.get("fixed_k") or {}).get("cuda_training_seconds") is not None
        for r in cm_rows)
    if cm_variant_ok and cm_class_variant_ok and cm_timing_ok:
        checks.append("fused cost model regenerated with fused class costs and fixed-k warm-up/CUDA timing")
    else:
        errors.append("COST_MODEL is not a fresh fused measurement with fixed-k timing")
    invalid = [a for a in arms if a.get("status") != "EXECUTED"]
    executed = [a for a in arms if a.get("status") == "EXECUTED"]
    for a in arms:
        for key in ("status", "scope", "held_reads", "registered_config", "execution_manifest", "coverage_status"):
            if key not in a:
                errors.append(f"{a.get('_file')} missing {key}")
    for a in executed:
        if a.get("held_reads") != 0:
            errors.append(f"{a.get('_file')} FIT held_reads != 0")
        if (a.get("execution_manifest") or {}).get("status") != "PASS":
            errors.append(f"{a.get('_file')} self-check not PASS")
        if a.get("arm") == "pcfa":
            pol = a.get("policy", {})
            if not pol.get("valid_nondegenerate"):
                errors.append(f"{a.get('_file')} PC-FA degenerate")
            if not pol.get("feasible"):
                errors.append(f"{a.get('_file')} PC-FA infeasible")
            if pol.get("cost_per_sample", 1e99) > pol.get("budget_per_sample", 0.0) + 1e-9:
                errors.append(f"{a.get('_file')} PC-FA model cost exceeds budget")
            if not a.get("state_path"):
                errors.append(f"{a.get('_file')} missing state_path")
            elif not args.skip_state_existence and not Path(a["state_path"]).exists():
                errors.append(f"{a.get('_file')} state missing")
            if not a.get("pcfa_fused") or (
                    (a.get("execution_manifest") or {}).get("pcfa_implementation_variant")
                    != "fused_single_encoder_graph"):
                errors.append(f"{a.get('_file')} PC-FA matched arm is not the fused variant")
        if a.get("arm") in ("pcfa", "fixed_k_matched_B02"):
            if a.get("cuda_warmup_enabled") is not True:
                errors.append(f"{a.get('_file')} CUDA warm-up not enabled")
            if a.get("warmup_seconds") is None or a.get("cuda_training_seconds") is None:
                errors.append(f"{a.get('_file')} missing warm-up/CUDA timing")
    if invalid:
        checks.append(f"invalid artifacts excluded: {len(invalid)}")
    else:
        checks.append("no invalid artifacts")
    matched_names = {a.get("arm") for a in executed if a.get("_phase") == "matched"}
    required = {"native_balanced", "effective_number", "fixed_k_matched_B02", "kmfaps", "pcfa"}
    missing = sorted(required - matched_names)
    if missing:
        errors.append(f"missing matched arms: {missing}")
    else:
        checks.append("required matched arms present")
    # Held native metrics.
    held_by = {}
    for r in (held or {}).get("results", []):
        held_by[(r.get("phase"), r.get("method"), r.get("partition"),
                 r.get("checkpoint_seed"), r.get("training_seed"), r.get("arm"))] = r
    groups = defaultdict(list)
    for a in executed:
        groups[(a.get("_phase"), a.get("arm"))].append(a)
    rows = []
    for (phase, arm), items in sorted(groups.items()):
        def hval(key, sub):
            return mean_std([held_by.get((x.get("_phase"), x.get("method"), x.get("partition"),
                                          x.get("checkpoint_seed"), x.get("training_seed"),
                                          x.get("arm")), {}).get(key, {}).get(sub) for x in items])
        fit = mean_std([x.get("fit_metrics_native_head", {}).get("macro_f1") for x in items])
        tail = mean_std([x.get("fit_metrics_native_head", {}).get("tail_recall_worst10") for x in items])
        hm = hval("held_native_head", "macro_f1"); ht = hval("held_native_head", "tail_recall_worst10")
        wall = mean_std([x.get("wall_clock_training_seconds") for x in items])
        fl = mean_std([x.get("estimated_training_flops") for x in items])
        obs = [(x.get("execution_manifest") or {}).get("observed", {}) for x in items]
        def omean(key):
            return mean_std([o.get(key) for o in obs])[0]
        rows.append({"phase": phase, "arm": arm, "n": len(items), "n_held": hm[2],
                     "fit_macro": fit[0], "fit_tail": tail[0],
                     "held_macro": hm[0], "held_tail": ht[0],
                     "wall": wall[0], "flops": fl[0],
                     "ref_actions": omean("reference_action_passes"),
                     "solver_steps": omean("solver_steps"),
                     "sample_solver_steps": omean("sample_solver_steps"),
                     "native_group_passes": omean("native_group_passes")})
    by = {(r["phase"], r["arm"]): r for r in rows}
    # Actual cost ratios PC-FA vs fixed-k, per partition/seed.
    matched_pc = [a for a in executed if a.get("_phase") == "matched" and a.get("arm") == "pcfa"]
    matched_fx = [a for a in executed if a.get("_phase") == "matched" and a.get("arm") == "fixed_k_matched_B02"]
    fx = {(a.get("partition"), a.get("training_seed")): a for a in matched_fx}
    ratios = []
    for a in matched_pc:
        b = fx.get((a.get("partition"), a.get("training_seed")))
        if not b:
            errors.append(f"no fixed-k pair for pcfa {a.get('_file')}")
            continue
        pcfa_cuda = a.get("cuda_training_seconds")
        fixed_cuda = b.get("cuda_training_seconds")
        ratios.append({
            "partition": a.get("partition"), "training_seed": a.get("training_seed"),
            "pcfa_flops": a.get("estimated_training_flops"), "fixed_k_flops": b.get("estimated_training_flops"),
            "flops_ratio": float(a.get("estimated_training_flops") / max(b.get("estimated_training_flops", 1e-30), 1e-30)),
            "pcfa_wall": a.get("wall_clock_training_seconds"), "fixed_k_wall": b.get("wall_clock_training_seconds"),
            "wall_ratio": float(a.get("wall_clock_training_seconds") / max(b.get("wall_clock_training_seconds", 1e-30), 1e-30)),
            "pcfa_cuda": pcfa_cuda, "fixed_k_cuda": fixed_cuda,
            "cuda_ratio": (float(pcfa_cuda / max(fixed_cuda, 1e-30))
                           if pcfa_cuda is not None and fixed_cuda is not None else None),
        })
    max_flops_ratio = max([r["flops_ratio"] for r in ratios], default=None)
    mean_flops_ratio = mean_std([r["flops_ratio"] for r in ratios])[0]
    max_wall_ratio = max([r["wall_ratio"] for r in ratios], default=None)
    mean_wall_ratio = mean_std([r["wall_ratio"] for r in ratios])[0]
    cuda_ratios = [r["cuda_ratio"] for r in ratios if r.get("cuda_ratio") is not None]
    max_cuda_ratio = max(cuda_ratios, default=None)
    mean_cuda_ratio = mean_std(cuda_ratios)[0] if cuda_ratios else None
    pc = by.get(("matched", "pcfa")); fxrow = by.get(("matched", "fixed_k_matched_B02"))
    km = by.get(("matched", "kmfaps"))
    gates = []
    tail_up = False
    flops_ok = False
    wall_ok = False
    cost_ok = False
    model_cost_ok = False
    if not pc or not pc.get("held_macro") or not fxrow:
        gates.append("PROXY_BUDGET_ONLY: PC-FA matched evidence missing")
        gates.append("IMPLEMENTATION_OPTIMIZATION_UNVERIFIED: matched fusion evidence missing")
    else:
        flops_ok = (max_flops_ratio is not None and max_flops_ratio <= 1.05 + 1e-9)
        wall_ok = (max_wall_ratio is not None and max_wall_ratio <= 1.05 + 1e-9)
        cost_ok = bool(flops_ok and wall_ok)
        model_cost_ok = all(
            (a.get("policy", {}).get("cost_per_sample", 1e99) <= a.get("policy", {}).get("budget_per_sample", 0.0) + 1e-9)
            for a in matched_pc)
        if not model_cost_ok:
            gates.append("PROXY_BUDGET_ONLY: PC-FA model cost exceeds 1.05x measured fixed-k budget")
        if not flops_ok:
            gates.append("PROXY_BUDGET_ONLY: actual measured FLOPs exceeds 1.05x fixed-k")
        if not wall_ok:
            gates.append("IMPLEMENTATION_OPTIMIZATION_UNVERIFIED: fused end-to-end wall-clock exceeds 1.05x fixed-k")
        kuniq = [a.get("policy", {}).get("k_unique") for a in matched_pc]
        if any(not u or len(u) <= 1 for u in kuniq):
            gates.append("DEGENERATE_ALLOCATION: k_c is constant")
        tail_up = pc["held_tail"] is not None and fxrow["held_tail"] is not None and pc["held_tail"] > fxrow["held_tail"]
        macro_ok = pc["held_macro"] is not None and fxrow["held_macro"] is not None and pc["held_macro"] >= fxrow["held_macro"]
        if tail_up and not macro_ok:
            gates.append("TAIL_MACRO_TRADEOFF: HELD tail-10% up but Macro-F1 below fixed-k")
        if not tail_up:
            gates.append("ALGORITHM_BOUNDARY: HELD tail gate not passed")
        if tail_up and macro_ok and cost_ok and model_cost_ok:
            gates.append("ALGORITHM_CLAIM_REOPENED: equivalence PASS, wall-clock <=1.05x, HELD tail up and Macro-F1 not below fixed-k")
            gates.append("PARTIAL_ALGORITHM_RESULT: tail-10% up, Macro-F1 not below fixed-k, measured cost <=1.05x")
    if pc and fxrow:
        if not tail_up:
            gates.append("NO_TAIL_GAIN: PC-FA HELD tail-10% is not above fixed-k; the unique main question is answered no.")
        if (pc.get("fit_tail") is not None and fxrow.get("fit_tail") is not None
                and pc["fit_tail"] > fxrow["fit_tail"] and not tail_up):
            gates.append("FIT_TAIL_UP_HELD_NONREPRO: FIT tail-10% gain did not reproduce on HELD.")
    sm = smoke.get("execution_manifest") or {}
    smoke_fused = sm.get("pcfa_implementation_variant") == "fused_single_encoder_graph"
    if (sm.get("action_count_match") is True and sm.get("flops_ratio") is not None
            and smoke_fused):
        checks.append(f"fused smoke action counts match; measured/model FLOPs ratio {sm.get('flops_ratio'):.4f}")
    else:
        errors.append("PCFA_COST_SMOKE action-count, fused-variant, or cost consistency missing")
    if not gates:
        gates.append("NO_GATE_TRIGGERED: no cost-aware PC-FA advantage; report boundary, not SOTA")
    # Summary.
    md = ["# Cost-aware PC-FA Summary", "",
          "Status: deterministic artifact summary; semantic review unavailable.", "",
          "FIT metrics are training-set proxies; HELD is a single post-freeze read. Budget is measured, not proxy.",
          "", "## Matched arms", "",
          "| arm | n | FIT Macro-F1 | FIT tail-10% | HELD Macro-F1 | HELD tail-10% | wall s | FLOPs | ref actions | solver |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["phase"] != "matched":
            continue
        def f(x, d=4): return "n/a" if x is None else f"{x:.{d}f}"
        md.append(f"| {r['arm']} | {r['n']} | {f(r['fit_macro'])} | {f(r['fit_tail'])} | "
                  f"{f(r['held_macro'])} | {f(r['held_tail'])} | {f(r['wall'],3)} | {f(r['flops'],0)} | "
                  f"{f(r['ref_actions'],1)} | {f(r['solver_steps'],1)} |")
    md += ["", "## Measured cost ratios (PC-FA / fixed-k-matched-B02)", ""]
    if ratios:
        md += ["| partition | seed | flops ratio | wall ratio | CUDA-event ratio |",
               "|---:|---:|---:|---:|---:|"]
        for r in ratios:
            cr = "n/a" if r.get("cuda_ratio") is None else f"{r['cuda_ratio']:.4f}"
            md.append(f"| {r['partition']} | {r['training_seed']} | {r['flops_ratio']:.4f} | "
                      f"{r['wall_ratio']:.4f} | {cr} |")
        cuda_txt = ("n/a" if max_cuda_ratio is None else f"{max_cuda_ratio:.4f}")
        md.append(f"\nmax FLOPs ratio **{max_flops_ratio:.4f}**, mean **{mean_flops_ratio:.4f}**, "
                  f"max wall ratio **{max_wall_ratio:.4f}**, mean **{mean_wall_ratio:.4f}**, "
                  f"max CUDA-event ratio **{cuda_txt}**.")
        md.append("\nEquivalence gate: real-checkpoint old/fused action sequence, logical counters, "
                  "and ≤1e-5 encoder-state delta.")
        md.append(f"Implementation optimization: **{'VERIFIED' if (eq_ok and flops_ok and wall_ok) else 'UNVERIFIED'}** "
                  f"(equivalence={eq_ok}, FLOPs≤1.05×={flops_ok}, wall≤1.05×={wall_ok}).")
    else:
        md.append("No paired PC-FA/fixed-k matched arms.")
    md += ["", "## Automated gates"] + [f"- {g}" for g in gates]
    if pc and fxrow:
        md += ["", "## Comparison", "",
               f"- PC-FA HELD Macro-F1 {pc['held_macro']:.4f}, tail-10% {pc['held_tail']:.4f}.",
               f"- fixed-k HELD Macro-F1 {fxrow['held_macro']:.4f}, tail-10% {fxrow['held_tail']:.4f}."]
        if km:
            md.append(f"- corrected K-MFAPS HELD Macro-F1 {km['held_macro']:.4f}, tail-10% {km['held_tail']:.4f} (cost context only).")
    md += ["", "## Evidence boundary",
           "- Cost model uses runner FlopCounter and actual reference-action counts, not theoretical step costs.",
           "- The old sum(q_c k_c)~0.2 proxy budget is not used as matched-compute evidence.",
           "- The final latency gate uses end-to-end wall-clock; CUDA events only explain the bottleneck.",
           "- FLOPs reduction alone is not treated as wall-clock success.",
           "- No SOTA, universal improvement, cross-task, generalization, or independent-review claim.",
           "- Semantic reviewer: REVIEW_UNAVAILABLE."]
    Path(args.summary).write_text("\n".join(md) + "\n", encoding="utf8")
    claims = {
        "status": "EXECUTED" if not errors else "INVALID_CONFIG",
        "scope": "deterministic cost-aware PC-FA claims/evidence audit; no semantic reviewer",
        "held_reads": int(held.get("held_reads", 0) or 0),
        "registered_config": {
            "budget_factor": 1.05,
            "cost_measured": True,
            "cost_formula": "native_cost_c + k*solver_cost_c + 1[alpha>0]*reference_cost_c",
            "mean_risk_constraint": "fixed-k mean risk",
            "tail_cvar_gamma": 0.8,
            "held_policy": "one-time after FIT freeze",
            "pcfa_implementation_variant": "fused_single_encoder_graph",
            "equivalence_state_max_abs_tolerance": 1e-5,
            "wall_clock_gate": "end-to-end wall-clock ratio <= 1.05",
            "held_tail_gate": "PC-FA HELD tail-10% above fixed-k",
        },
        "execution_manifest": {
            "checks": checks, "errors": errors,
            "n_artifacts": len(arms), "n_executed": len(executed), "n_invalid": len(invalid),
            "state_presence_checked": not bool(args.skip_state_existence),
            "summary": rows, "cost_ratios": ratios,
            "max_flops_ratio": max_flops_ratio, "mean_flops_ratio": mean_flops_ratio,
            "max_wall_ratio": max_wall_ratio, "mean_wall_ratio": mean_wall_ratio,
            "max_cuda_ratio": max_cuda_ratio, "mean_cuda_ratio": mean_cuda_ratio,
            "equivalence_status": eq.get("status"),
            "equivalence_gate_ok": eq_ok,
            "fused_implementation_variant": cm_reg.get("implementation_variant"),
            "implementation_optimization_status": (
                "VERIFIED" if (eq_ok and flops_ok and wall_ok) else "UNVERIFIED"),
            "algorithm_tail_gate_passed": bool(tail_up),
            "gates": gates, "smoke_status": smoke.get("status"),
            "review_status": "REVIEW_UNAVAILABLE",
            "review_note": "No independent semantic reviewer; this is not a substitute.",
        },
        "coverage_status": {
            "status": "proxy_only_unverified",
            "note": "FIT cost/risk proxies plus one-time HELD; actual cost ratios reported",
        },
        "classification": {
            "EXECUTED": [a.get("_file") for a in executed],
            "REGISTERED_NOT_EXECUTED": [
                "CIFAR10-LT/CIFAR100-LT/ImageNet-LT cross-domain runner",
                "Power(p)/learnable spectral geometry comparison",
                "independent second SPD project validation",
            ],
            "UNVERIFIED": ["generalization", "SOTA", "semantic review", "cross-task transfer"],
            "INVALID_CONFIG": [a.get("_file") for a in invalid],
            "BLOCKED_ENV": [] if env.get("status") == "EXECUTED" else ["environment manifest"],
        },
        "review_status": "REVIEW_UNAVAILABLE",
    }
    write_json(args.audit_json, claims)
    amd = ["# Cost-aware PC-FA Claims/Evidence Audit", "",
           f"Status: **{claims['status']}**", "",
           "This is a deterministic audit, not an independent semantic review.",
           "Semantic reviewer: **REVIEW_UNAVAILABLE**.", "", "## Checks"] + [f"- {c}" for c in checks]
    amd += ["", "## Errors"] + ([f"- {e}" for e in errors] if errors else ["- none"])
    amd += ["", "## Gates"] + [f"- {g}" for g in gates]
    amd += ["", "## Evidence classification"]
    for k, vals in claims["classification"].items():
        amd.append(f"- **{k}**: {len(vals)}")
        for v in vals[:20]:
            amd.append(f"  - `{v}`")
    amd += ["", "## Explicitly not claimed",
            "- SOTA, universal long-tail improvement, cross-task generalization, or independent review.",
            "- Proxy-budget gains as matched-compute evidence.",
            "- Cost-aware PC-FA as a main result unless the measured cost and held gates pass."]
    Path(args.audit_md).write_text("\n".join(amd) + "\n", encoding="utf8")
    print(json.dumps({"status": claims["status"], "errors": errors,
                      "executed": len(executed), "invalid": len(invalid),
                      "gates": gates, "max_flops_ratio": max_flops_ratio,
                      "max_wall_ratio": max_wall_ratio}, ensure_ascii=False))


if __name__ == "__main__":
    main()
