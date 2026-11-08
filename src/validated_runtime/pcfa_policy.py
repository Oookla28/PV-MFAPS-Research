#!/usr/bin/env python3
"""FIT-only cost-aware PC-FA allocation.

Objective: minimize uniform-class CVaR_gamma(r) subject to
  sum_c q_c c_c(k_c,alpha_c) <= 1.05 * measured fixed-k per-sample cost
  sum_c q_c r_c <= fixed-k mean risk
  k_c in {0,1,2,4}, alpha_c in {0.25,0.5,0.75,1}, k=0 => alpha=0.
The cost model is measured by the runner FlopCounter, not theoretical steps.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tfgps_gpu_lib import ALPHA_GRID, cvar_terms, write_json, omega_from_counts, KM_K_GRID

DEFAULT_K_GRID = tuple(int(v) for v in KM_K_GRID)
ACTION_ALPHAS = (0.25, 0.5, 0.75, 1.0)


def load_json(path):
    return json.loads(Path(path).read_text())


def normalize_k_grid(k_grid):
    if isinstance(k_grid, str):
        k_grid = [v for v in k_grid.split(",") if v.strip() != ""]
    grid = sorted({int(v) for v in k_grid})
    if not grid:
        raise ValueError("k_grid must not be empty")
    if grid[0] < 0:
        raise ValueError("k_grid must be non-negative")
    if 0 not in grid:
        raise ValueError("k_grid must contain 0 (native action)")
    if len(grid) < 2:
        raise ValueError("k_grid must contain at least one positive action")
    return grid


def choose_pcfa(cost_row, cal, gamma=0.8, budget_factor=1.05,
                k_grid=DEFAULT_K_GRID, inner_budget=None, variant=None):
    counts = np.asarray(cal["counts"], dtype=float)
    A = np.asarray(cal["A"], dtype=float)
    rho = np.asarray(cal["rho"], dtype=float)
    sigma = np.asarray(cal["sigma"], dtype=float)
    C = len(counts)
    q = np.asarray(cost_row["q"], dtype=float)
    q = q / q.sum()
    m = float(cost_row.get("m", 8))
    cc = cost_row["class_costs"]
    native = np.asarray([cc[str(c)]["native_cost_per_sample"] for c in range(C)], dtype=float)
    solver = np.asarray([cc[str(c)]["solver_step_cost_per_sample"] for c in range(C)], dtype=float)
    refact = np.asarray([cc[str(c)]["reference_action_cost_per_sample"] for c in range(C)], dtype=float)
    fixed_cost = float(cost_row["fixed_k"]["per_sample_flops"])
    budget = float(budget_factor * fixed_cost)
    Rbar = float(cost_row["mean_risk_fixed_k"])
    cvar_fixed = float(cost_row["cvar_fixed_k"])
    grid = normalize_k_grid(k_grid)
    actions = []
    for k in grid:
        if int(k) == 0:
            actions.append((0, 0.0))
        else:
            for a in ACTION_ALPHAS:
                actions.append((int(k), float(a)))
    # Per-class risk/cost tables.
    table = {}
    for c in range(C):
        rows = []
        for k, a in actions:
            risk = (1.0 - a) ** 2 * A[c] ** 2 * rho[c] ** (2 * k)
            risk += a ** 2 * sigma[c] ** 2 / (m * max(q[c], 1e-15))
            cost = native[c] + k * solver[c] + (refact[c] if a > 0 else 0.0)
            rows.append({"k": k, "alpha": a, "risk": float(risk), "cost": float(cost)})
        table[c] = rows
    # Lambda sweep spanning risk-dominated to cost-dominated.
    risks_med = float(np.median([max(r["risk"] for r in table[c]) for c in range(C)]))
    costs_med = float(np.median([max(q[c] * r["cost"], 1.0) for c in range(C) for r in table[c]]))
    lam_ref = risks_med / max(costs_med, 1e-30)
    lams = [0.0] + list(np.geomspace(max(lam_ref * 1e-7, 1e-30),
                                     max(lam_ref * 1e7, 1e-30), 90))
    fixed_risk = sigma ** 2 / (m * np.maximum(q, 1e-15))
    fixed_k = np.ones(C, dtype=int)
    fixed_a = np.ones(C, dtype=float)
    tail_n = max(1, int(math.ceil(0.2 * C)))
    tail_fixed = np.argsort(-fixed_risk)[:tail_n]
    candidates = []

    def add_candidate(k_arr, a_arr, ctype, lam):
        k_arr = np.asarray(k_arr, dtype=int)
        a_arr = np.asarray(a_arr, dtype=float)
        cvals = np.zeros(C); rvals = np.zeros(C)
        for c in range(C):
            row = next(x for x in table[c] if x["k"] == int(k_arr[c]) and abs(x["alpha"] - float(a_arr[c])) < 1e-12)
            cvals[c] = row["cost"]; rvals[c] = row["risk"]
        cost = float(np.sum(q * cvals))
        mean_risk = float(np.sum(q * rvals))
        k_budget = float(np.sum(q * k_arr))
        cvar, _, _ = cvar_terms(rvals, gamma)
        candidates.append({"k": k_arr, "alpha": a_arr, "cost": cost,
                           "mean_risk": mean_risk, "cvar": float(cvar),
                           "k_budget": k_budget,
                           "type": ctype, "lambda": float(lam),
                           "cost_by_class": cvals.tolist(), "risk_by_class": rvals.tolist()})

    # Native and fixed-k reference candidates.
    add_candidate(np.zeros(C, dtype=int), np.zeros(C), "native", 0.0)
    add_candidate(fixed_k, fixed_a, "fixed_k_measured", 0.0)
    candidates[-1]["cost"] = fixed_cost  # actual measured fixed-k cost
    for lam in lams:
        k_arr = np.zeros(C, dtype=int); a_arr = np.zeros(C, dtype=float)
        for c in range(C):
            best = min(table[c], key=lambda x: x["risk"] + lam * q[c] * x["cost"])
            k_arr[c] = best["k"]; a_arr[c] = best["alpha"]
        add_candidate(k_arr, a_arr, "all_class_lagrangian", lam)
        # Tail-focused: retain native for non-tail classes.
        k_t = np.zeros(C, dtype=int); a_t = np.zeros(C, dtype=float)
        for c in tail_fixed:
            best = min(table[c], key=lambda x: x["risk"] + lam * q[c] * x["cost"])
            k_t[c] = best["k"]; a_t[c] = best["alpha"]
        add_candidate(k_t, a_t, "tail_focused_lagrangian", lam)
    # Feasibility, nondegeneracy, choose minimum CVaR among feasible nondegenerate.
    inner_budget_value = float(inner_budget) if inner_budget is not None else None
    feasible = []
    for cand in candidates:
        cand["cost_ok"] = bool(cand["cost"] <= budget + 1e-9)
        cand["risk_ok"] = bool(cand["mean_risk"] <= Rbar + 1e-9)
        cand["k_budget_ok"] = bool(
            inner_budget_value is None or cand["k_budget"] <= inner_budget_value + 1e-9)
        cand["feasible"] = bool(cand["cost_ok"] and cand["risk_ok"] and cand["k_budget_ok"])
        uniq = sorted(set(int(v) for v in cand["k"].tolist()))
        cand["k_unique"] = uniq
        cand["k_dispersion"] = float(np.std(cand["k"]))
        cand["nondegenerate"] = bool(len(uniq) > 1 and cand["k_dispersion"] > 0.0)
        if cand["feasible"] and cand["nondegenerate"]:
            feasible.append(cand)
    if feasible:
        best = min(feasible, key=lambda x: (x["cvar"], x["cost"]))
        valid = True
        boundary = []
    else:
        # Fall back to the actual fixed-k candidate; it may be infeasible under
        # the registered k-budget, in which case the action space is a boundary.
        best = candidates[1]
        valid = False
        if inner_budget_value is not None and best["k_budget"] > inner_budget_value + 1e-9:
            reason = "BOUNDARY_ACTION_SPACE"
        elif not best["nondegenerate"]:
            reason = "DEGENERATE_ALLOCATION"
        else:
            reason = "PROXY_BUDGET_ONLY"
        boundary = [reason]
    k_arr = np.asarray(best["k"], dtype=int); a_arr = np.asarray(best["alpha"], dtype=float)
    if np.any((k_arr == 0) & (a_arr > 0.0)):
        raise ValueError("k=0 must imply alpha=0")
    uniq = sorted(set(int(v) for v in k_arr.tolist()))
    return {
        "tau": 0.75, "kappa": 0.0, "gamma": float(gamma), "k": int(max(k_arr)),
        "alpha_base": float(np.mean(a_arr)),
        "q": q.tolist(), "omega": q.tolist(), "q_sum": float(q.sum()), "m": int(m),
        "k_discrete": k_arr.tolist(), "alpha_discrete": a_arr.tolist(),
        "k_unique": uniq,
        "k_dispersion": float(np.std(k_arr)),
        "alpha_unique": sorted(set(float(v) for v in a_arr.tolist())),
        "cost_per_sample": float(best["cost"]),
        "budget_per_sample": float(budget),
        "fixed_k_cost_per_sample": fixed_cost,
        "cost_ratio": float(best["cost"] / max(fixed_cost, 1e-30)),
        "mean_risk": float(best["mean_risk"]),
        "mean_risk_bar": float(Rbar),
        "k_budget": float(best["k_budget"]),
        "inner_budget": inner_budget_value,
        "cvar": float(best["cvar"]),
        "cvar_fixed_k": cvar_fixed,
        "selected_lambda": float(best["lambda"]),
        "candidate_type": best["type"],
        "feasible": bool(best["feasible"]),
        "valid_nondegenerate": bool(valid),
        "boundary_reason": boundary,
        "tail_indices": tail_fixed.tolist(),
        "cost_by_class": best["cost_by_class"],
        "risk_by_class": best["risk_by_class"],
        "registered_config": {
            "cost_formula": "c_c(k,alpha)=native_cost_c+k*solver_cost_c+1[alpha>0]*reference_cost_c",
            "budget_factor": float(budget_factor),
            "budget_per_sample_flops": float(budget),
            "fixed_k_cost_per_sample_flops": fixed_cost,
            "mean_risk_constraint": float(Rbar),
            "tail_cvar_gamma": float(gamma),
            "q": "omega(counts,tau=.75)", "m": int(m),
            "k_grid": list(grid), "alpha_grid": list(ACTION_ALPHAS),
            "inner_budget": inner_budget_value,
            "inner_budget_constraint": "sum_c q_c k_c <= inner_budget",
            "variant": variant,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cost-model", required=True)
    ap.add_argument("--precheck", required=True)
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--partitions", default="0,1")
    ap.add_argument("--k-grid", default=",".join(str(v) for v in DEFAULT_K_GRID))
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--inner-budget", type=float, default=None)
    ap.add_argument("--budget-factor", type=float, default=1.05)
    ap.add_argument("--variant", default="a100_fp32_fused")
    args = ap.parse_args()
    grid = normalize_k_grid(args.k_grid)
    out = Path(args.out)
    cm = load_json(args.cost_model)
    rows = {}
    result_rows = []
    for part in [int(x) for x in args.partitions.split(",") if x.strip()]:
        row = next(r for r in cm["execution_manifest"]["rows"] if int(r["partition"]) == part)
        cal = load_json(row["calibration"])
        pol = choose_pcfa(row, cal, gamma=args.gamma, budget_factor=args.budget_factor,
                          k_grid=grid, inner_budget=args.inner_budget, variant=args.variant)
        rows[str(part)] = pol
        result_rows.append({
            "partition": part,
            "k_unique": pol["k_unique"], "k_dispersion": pol["k_dispersion"],
            "alpha_unique": pol["alpha_unique"],
            "cost_per_sample": pol["cost_per_sample"],
            "fixed_k_cost_per_sample": pol["fixed_k_cost_per_sample"],
            "cost_ratio": pol["cost_ratio"],
            "budget_per_sample": pol["budget_per_sample"],
            "mean_risk": pol["mean_risk"], "mean_risk_bar": pol["mean_risk_bar"],
            "cvar": pol["cvar"], "cvar_fixed_k": pol["cvar_fixed_k"],
            "selected_lambda": pol["selected_lambda"],
            "candidate_type": pol["candidate_type"],
            "feasible": pol["feasible"],
            "valid_nondegenerate": pol["valid_nondegenerate"],
            "boundary_reason": pol["boundary_reason"],
            "k_discrete": pol["k_discrete"], "alpha_discrete": pol["alpha_discrete"],
            "k_budget": pol["k_budget"], "inner_budget": pol["inner_budget"],
        })
    valid = all(bool(p["valid_nondegenerate"] and p["feasible"]) for p in rows.values())
    base = {
        "scope": "FIT-only cost-aware PC-FA allocation precheck; no HELD sample indexed",
        "held_reads": 0,
        "registered_config": {
            "budget_factor": float(args.budget_factor),
            "cost_formula": "c_c(k,alpha)=native_cost_c+k*solver_cost_c+1[alpha>0]*reference_cost_c",
            "mean_risk_constraint": "sum_c q_c r_c <= fixed-k mean risk",
            "inner_budget_constraint": "sum_c q_c k_c <= inner_budget",
            "tail_objective": f"uniform-class CVaR gamma={float(args.gamma):.2f}",
            "tail_cvar_gamma": float(args.gamma),
            "k_grid": list(grid), "alpha_grid": list(ACTION_ALPHAS),
            "inner_budget": args.inner_budget,
            "variant": args.variant,
            "q": "omega(counts,tau=.75)",
            "cost_model_implementation": (
                (cm.get("registered_config") or {}).get("implementation_variant")),
        },
        "coverage_status": {
            "status": "proxy_only_unverified",
            "note": "FIT cost/risk proxies; matched actual FLOPs/wall-clock checked separately",
        },
    }
    if not valid:
        reasons = sorted({r for p in result_rows for r in p["boundary_reason"]})
        bad = {
            **base,
            "status": "UNVERIFIED",
            "boundary_status": "BOUNDARY_ACTION_SPACE" if "BOUNDARY_ACTION_SPACE" in reasons else "PCFA_BOUNDARY",
            "execution_manifest": {"rows": result_rows, "policies": rows,
                                   "reason": "no feasible nondegenerate allocation within the registered "
                                             "measured fixed-k cost, risk and inner-k budget constraints",
                                   "boundary_reasons": reasons},
        }
        write_json(args.boundary, bad)
        print(json.dumps({"status": "UNVERIFIED", "rows": result_rows}, ensure_ascii=False))
        raise SystemExit(3)
    good = {
        **base,
        "status": "EXECUTED",
        "execution_manifest": {"rows": result_rows, "policies": rows,
                               "training_started": False},
    }
    write_json(args.precheck, good)
    print(json.dumps({"status": "EXECUTED", "rows": result_rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
