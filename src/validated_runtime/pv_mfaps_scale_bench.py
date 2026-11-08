#!/usr/bin/env python3
"""PV-MFAPS discrete-solver verification and large-class-cost benchmark.

CPU-only.  Verifies the finite-action DP against brute force, measures the
exact-DP time/memory/frontier curve for 69 and larger class counts, and
compares a sparse active-class DP and a marginal-gain greedy approximation
against the exact solution on the 69-class instance.  It never reads HELD and
never runs on the GPU used by the pilot.
"""
from __future__ import annotations

import argparse
import itertools
import json
import signal
import time
from pathlib import Path

import numpy as np

import pv_mfaps_dp as dp
import pv_mfaps_dp_solver as solver


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def make_instance(C, seed, tail_fraction=0.15):
    rng = np.random.default_rng(seed)
    n = np.maximum(np.round(np.geomspace(5, 344, C)).astype(float), 5)
    rho = np.clip(rng.normal(0.9995, 3e-4, size=C), 0.99, 0.99999)
    A = rng.uniform(0.2, 1.0, size=C)
    sigma = rng.uniform(0.05, 0.5, size=C)
    q = n / n.sum()
    order = np.argsort(n)
    k_tail = max(1, int(round(tail_fraction * C)))
    mask = np.zeros(C, dtype=bool); mask[order[:k_tail]] = True
    return dict(A=A, rho=rho, sigma=sigma, n=n, q=q, tail=mask)


def brute_force(A, rho, sigma, n, q, budget, k_grid, xi, mu, tail, tail_cap, macro_cap):
    C = len(A)
    weights = mu / C
    best = None
    for combo in itertools.product(k_grid, repeat=C):
        cost = 0.0; risks = []
        for c, k in enumerate(combo):
            r = float(dp.risk_envelope([A[c]], [rho[c]], [sigma[c]], [n[c]], k, xi)[0])
            risks.append(r)
            cost += q[c] * k
        if cost > budget + 1e-12:
            continue
        risks = np.asarray(risks)
        if macro_cap is not None and risks.mean() > macro_cap + 1e-12:
            continue
        if tail_cap is not None and tail.any() and risks[tail].max() > tail_cap + 1e-12:
            continue
        obj = float(weights * risks.sum())
        if best is None or obj < best[0] - 1e-15:
            best = (obj, combo, float(cost), float(risks.mean()), float(risks[tail].max()) if tail.any() else 0.0)
    return best


def solve(inst, budget, k_grid, alpha_grid, xi, mu, macro_cap, timeout=180):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    t0 = time.perf_counter()
    try:
        res = dp.solve_pv_mfaps_dp(
            rho=inst["rho"], A=inst["A"], sigma=inst["sigma"], n=inst["n"],
            mu=mu, q=inst["q"], budget=budget, k_grid=k_grid, alpha_grid=alpha_grid,
            xi=xi, tail_mask=inst["tail"], macro_cap=macro_cap, return_frontier_stats=True)
    except _Timeout:
        return {"status": "TIMEOUT", "seconds": timeout,
                "reason": "exact DP exceeded the registered time cap; computational boundary"}
    finally:
        signal.alarm(0)
    res["seconds"] = time.perf_counter() - t0
    return res


def greedy(inst, budget, k_grid, xi, mu, macro_cap, scale=1000):
    C = len(inst["A"]); q = inst["q"]
    risks = np.array([[float(dp.risk_envelope([inst["A"][c]], [inst["rho"][c]], [inst["sigma"][c]],
                                              [inst["n"][c]], k, xi)[0]) for k in k_grid] for c in range(C)])
    # Match the DP's conservative ceil quantisation so both solve the same
    # quantised-budget problem.
    ks = [0] * C
    cost_i = 0
    budget_i = int(np.floor(budget * scale + 1e-12))
    t0 = time.perf_counter()
    improved = True
    while improved:
        improved = False
        best = None
        for c in range(C):
            cur = k_grid.index(ks[c])
            for j in range(cur + 1, len(k_grid)):
                dcost_i = int(np.ceil(max(0.0, q[c] * k_grid[j]) * scale - 1e-12)) - \
                          int(np.ceil(max(0.0, q[c] * k_grid[ks[c]]) * scale - 1e-12))
                if cost_i + dcost_i > budget_i:
                    continue
                gain = risks[c, cur] - risks[c, j]
                if gain <= 1e-15:
                    continue
                score = gain / max(dcost_i, 1e-12)
                if best is None or score > best[0]:
                    best = (score, c, j, dcost_i, gain)
        if best is not None:
            _, c, j, dcost_i, _ = best
            ks[c] = k_grid[j]; cost_i += dcost_i; improved = True
    chosen = np.array([risks[c, k_grid.index(ks[c])] for c in range(C)])
    return {"status": "GREEDY", "seconds": time.perf_counter() - t0, "k": ks,
            "actual_cost": float(cost_i / scale), "quantized_cost": int(cost_i),
            "macro_risk": float(chosen.mean()),
            "objective": float(mu / C * chosen.sum()),
            "macro_cap_satisfied": macro_cap is None or chosen.mean() <= macro_cap + 1e-12,
            "k_hist": {int(k): int(sum(1 for x in ks if x == k)) for k in k_grid}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--xi", type=float, default=1.01)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    k_grid = (0, 1, 2, 4)
    alpha_grid = (0.25,)
    # 1. Brute-force exactness on small instances.
    exact = []
    for trial in range(6):
        inst = make_instance(4, args.seed + trial, tail_fraction=0.5)
        budget = args.budget
        res = solve(inst, budget, k_grid, alpha_grid, args.xi, args.mu, None, timeout=60)
        bf = brute_force(inst["A"], inst["rho"], inst["sigma"], inst["n"], inst["q"],
                         budget, k_grid, args.xi, args.mu, inst["tail"], None, None)
        ok = (bf is not None and res.get("status", "").startswith("OPTIMAL") and
              abs(res["certificate"]["objective"] - bf[0]) <= 1e-9 and
              abs(res["certificate"]["actual_cost"] - bf[2]) <= 1e-9)
        exact.append({"trial": trial, "dp_status": res.get("status"),
                      "dp_objective": (res.get("certificate") or {}).get("objective"),
                      "bf_objective": None if bf is None else bf[0],
                      "dp_cost": (res.get("certificate") or {}).get("actual_cost"),
                      "bf_cost": None if bf is None else bf[2],
                      "match": bool(ok)})
    # 2. Exact DP curve for several class counts.
    curve = []
    for C in (69, 200):
        inst = make_instance(C, args.seed)
        res = solve(inst, args.budget, k_grid, alpha_grid, args.xi, args.mu, None, timeout=args.timeout)
        curve.append({"C": C, "status": res.get("status"), "seconds": res.get("seconds"),
                      "object": res.get("certificate"),
                      "frontier_final": (res.get("diagnostics") or {}).get("frontier_sizes", [])[-1:] or None,
                      "k_hist": ({int(k): int(sum(1 for x in res.get("k", []) if x == k)) for k in k_grid}
                                 if res.get("status", "").startswith("OPTIMAL") else None)})
    # 3. Large C: sparse active-class DP + greedy fallback.
    inst_big = make_instance(1000, args.seed + 1)
    tail_idx = np.flatnonzero(inst_big["tail"])
    sub = {k: inst_big[k][tail_idx] for k in ("A", "rho", "sigma", "n", "q")}
    sub["q"] = sub["q"] / sub["q"].sum()
    sub["tail"] = np.ones(len(tail_idx), dtype=bool)
    sparse = solve(sub, args.budget, k_grid, alpha_grid, args.xi, args.mu, None, timeout=args.timeout)
    g_big = greedy(inst_big, args.budget, k_grid, args.xi, args.mu, None)
    # 4. Greedy vs exact on 69 classes.
    inst69 = make_instance(69, args.seed)
    exact69 = solve(inst69, args.budget, k_grid, alpha_grid, args.xi, args.mu, None, timeout=args.timeout)
    g69 = greedy(inst69, args.budget, k_grid, args.xi, args.mu, None)
    gap = None
    if exact69.get("status", "").startswith("OPTIMAL") and exact69["certificate"]["objective"] > 0:
        gap = float((g69["objective"] - exact69["certificate"]["objective"]) /
                    exact69["certificate"]["objective"])
    # 5. Stability bound with registered constants.
    stab = solver.stability_bound(np.array([5, 10, 25, 100, 344], dtype=float),
                                  lipschitz=1.0, grad_bound=1.0, smoothness=1.0,
                                  step_size=0.1, k_grid=k_grid)
    result = {
        "status": "EXECUTED",
        "kind": "PV_MFAPS_DISCRETE_SOLVER_BENCH",
        "scope": "CPU-only finite-action DP verification and large-class cost curve; no HELD",
        "held_reads": 0,
        "registered_config": {"seed": args.seed, "budget": args.budget, "mu": args.mu, "xi": args.xi,
                              "k_grid": list(k_grid), "alpha_grid": list(alpha_grid),
                              "budget_scale": 1000, "timeout_seconds": args.timeout,
                              "stability_constants": {"L": 1.0, "G": 1.0, "beta": 1.0, "eta": 0.1}},
        "execution_manifest": {
            "brute_force_exactness": {"all_match": bool(all(x["match"] for x in exact)), "rows": exact},
            "exact_dp_curve": curve,
            "large_class_1000": {
                "sparse_active_class": {"n_active": int(len(tail_idx)), "status": sparse.get("status"),
                                        "seconds": sparse.get("seconds"),
                                        "certificate": sparse.get("certificate")},
                "greedy_full": g_big,
                "note": "full exact DP on 1000 classes is registered as a computational boundary; "
                        "sparse tail-only DP and greedy are the auditable approximations",
            },
            "greedy_vs_exact_69": {"greedy_objective": g69["objective"],
                                   "exact_objective": (exact69.get("certificate") or {}).get("objective"),
                                   "relative_gap": gap, "greedy_seconds": g69["seconds"],
                                   "exact_seconds": exact69.get("seconds")},
            "stability_bound": {"n": [5, 10, 25, 100, 344], "k_grid": list(k_grid),
                                "values": stab.tolist()},
        },
        "coverage_status": {"status": "solver_only",
                            "note": "DP exactness/feasibility evidence only; no algorithm efficacy claim"},
    }
    write_json(out / "PV_MFAPS_DP_BENCH.json", result)
    md = ["# PV-MFAPS discrete solver benchmark", "",
          f"- brute-force all match: {result['execution_manifest']['brute_force_exactness']['all_match']}",
          f"- exact DP 69 classes: {curve[0]['status']} in {curve[0]['seconds']:.2f}s",
          f"- exact DP 200 classes: {curve[1]['status']} in {curve[1]['seconds']:.2f}s",
          f"- 1000-class sparse tail DP: {sparse.get('status')} ({len(tail_idx)} active classes)",
          f"- greedy vs exact 69 relative objective gap: {gap}",
          "", "Status: EXECUTED (solver evidence only)."]
    (out / "PV_MFAPS_DP_BENCH.md").write_text("\n".join(md) + "\n", encoding="utf8")
    print(json.dumps({"status": "EXECUTED", "brute_force_all_match": all(x["match"] for x in exact),
                      "dp69": curve[0]["status"], "dp200": curve[1]["status"],
                      "sparse1000": sparse.get("status"), "greedy_gap": gap}, ensure_ascii=False))


if __name__ == "__main__":
    main()
